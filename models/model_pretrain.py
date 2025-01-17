'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BERT_LIKE
from models.xgpt import GPT2Config, GPT2LMHeadModel
from  transformers.modeling_outputs import BaseModelOutput
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            # checkpoint = torch.hub.load_state_dict_from_url(
            #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            #     map_location="cpu", check_hash=True)
            checkpoint = torch.load('pretrained_models/deit_base_patch16_224-b5f2ef4d.pth', map_location='cpu')
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']     
        self.text_encoder_mode = config.get('mode','retrieval')  
        if 'bert' in text_encoder.lower():
            self.text_encoder_type='bert'
            bert_config = BertConfig.from_json_file(config['bert_config'])
            if self.text_encoder_mode == 'generation':
                bert_config.is_encoder_decoder = True
                bert_config.mask_token_id = tokenizer.mask_token_id
            self.text_encoder, loading_info = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config, output_loading_info=True)   
        elif 'gpt' in text_encoder.lower():
            self.text_encoder_type='gpt'
            assert self.text_encoder_mode=='generation', self.text_encoder_mode
            gpt_config = GPT2Config.from_pretrained(config['gpt2_config']) #fusion_layer
            if 'fusion_layer' in config:
                gpt_config.fusion_layer = config['fusion_layer']
                print('Overwrite GPT2Config.fusion_layer to ', gpt_config.fusion_layer)
            self.text_encoder, loading_info = GPT2LMHeadModel.from_pretrained(text_encoder, config=gpt_config, output_loading_info=True)
        else:
            raise ValueError   

        print(loading_info)    
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)     

        # create momentum models
        if self.text_encoder_mode!='generation':
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
            self.vision_proj_m = nn.Linear(vision_width, embed_dim)
            self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
            self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.vision_proj,self.vision_proj_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_proj,self.text_proj_m],
                            ]
        
            self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
    

    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        if self.text_encoder_mode=='generation':
            if self.text_encoder_type == 'bert':
                #1. Mask text_ids 
                input_ids = text.input_ids.clone()
                labels = input_ids.clone()
                probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
                input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                            probability_matrix = probability_matrix)  
                if  self.text_encoder_type=='bert':
                    text_output = self.text_encoder.bert(input_ids, attention_mask = text.attention_mask,    
                                                    is_decoder = True, #important!                  
                                                    return_dict = True, mode = 'text')    
                # elif self.text_encoder_type=='gpt':
                #     text_output = self.text_encoder.transformer(input_ids, attention_mask=text.attention_mask,
                #                         return_dict=True, mode='text')
                text_embeds = text_output.last_hidden_state
                
                #we don't use [CLS] embedding here to compute text_feat, instead we use pooling (watch out for the padding token)
                #text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)    
                #text_feat = F.normalize(self.text_proj(text_embeds.mean(dim=1)),dim=-1)  
                pooled_embeds = torch.sum(text_embeds*text.attention_mask[:,:,None], dim=1)/torch.sum(text.attention_mask,dim=-1, keepdims=True) #B,L,D *B,L,1 -> B,L,D ->B,D/B,1
                text_feat = F.normalize(self.text_proj(pooled_embeds),dim=-1)   #B,D

                mlm_output = self.text_encoder(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            is_decoder = True, #!! important
                                            labels = labels, mode='fusion'  
                                            ) 
                            
                loss_mlm = mlm_output.loss 
                sim_i2t = image_feat @ text_feat.t() / self.temp  #(B,N)
                sim_t2i = text_feat @ image_feat.t() / self.temp 
                sim_targets = torch.zeros(sim_i2t.size()).to(image.device) #(B, N_t) (N_t==N_i?)
                sim_targets.fill_diagonal_(1)   
                loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
                loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 
                loss_ita = (loss_i2t+loss_t2i)/2
                
                loss_itm = torch.zeros_like(loss_ita)
                return loss_mlm, loss_ita, loss_itm
            
            elif self.text_encoder_type == 'gpt': #no-need to mask
                labels = text.input_ids.clone()

                labels[text.input_ids==self.tokenizer.pad_token_id] = -100 # We only compute loss on masked tokens 
                # print('input_ids', text.input_ids)
                # print('labels', labels)
                # input()
                if self.text_encoder.config.fusion_layer!=0:
                    text_output = self.text_encoder.transformer(text.input_ids,
                                        return_dict=True, mode='text')
                    text_embeds = text_output.last_hidden_state
                    
                    #we use [EOS] embedding here to compute text_feat
                    eos_pos = torch.sum(text.attention_mask, dim=1) #debug
                    #print(text.input_ids, text.attention_mask, eos_pos)
                    text_feat = F.normalize(self.text_proj(text_embeds[torch.arange(text_embeds.shape[0]),eos_pos-1,:]),dim=-1)    

                    mlm_output = self.text_encoder(inputs_embeds = text_embeds,  #encoder_embeds?->inputs_embeds
                                                attention_mask = text.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,      
                                                return_dict = True,
                                                labels = labels, mode='fusion'  
                                                ) 
                                
                    loss_mlm = mlm_output.loss 
                    sim_i2t = image_feat @ text_feat.t() / self.temp  #(B,N)
                    sim_t2i = text_feat @ image_feat.t() / self.temp 
                    sim_targets = torch.zeros(sim_i2t.size()).to(image.device) #(B, N_t) (N_t==N_i?)
                    sim_targets.fill_diagonal_(1)   
                    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
                    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 
                    loss_ita = (loss_i2t+loss_t2i)/2
                else:
                    mlm_output = self.text_encoder(
                                                text.input_ids, 
                                                attention_mask = text.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,      
                                                return_dict = True,
                                                labels = labels, mode='multi_modal'  
                                                )  
                    loss_mlm = mlm_output.loss  
                    loss_ita = torch.zeros_like(loss_mlm)
                loss_itm = torch.zeros_like(loss_mlm)    
                return loss_mlm, loss_ita, loss_itm                  
                

        
        
        elif self.text_encoder_mode=='retrieval':
            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
                
            # get momentum features
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image) 
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  #B,D
                image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)   #D,N_i                                    
                text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                                    return_dict = True, mode = 'text')    
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
                text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1) #D, N_t

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp  # B, D @ D,N_t  (B, N_t)
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp # B, D @ D, N_i   (B, N_i)

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device) #(B, N_t) (N_t==N_i?)
                sim_targets.fill_diagonal_(1)          

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

            sim_i2t = image_feat @ text_feat_all / self.temp  #(B,N)
            sim_t2i = text_feat @ image_feat_all / self.temp 
                                
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

            loss_ita = (loss_i2t+loss_t2i)/2

            self._dequeue_and_enqueue(image_feat_m, text_feat_m)

            ###=================================###
            # forward the positve image-text pair
            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )            
            with torch.no_grad():
                bs = image.size(0)          
                weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1) #(B,B)
                weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)

                weights_i2t.fill_diagonal_(0) #do not choose the positive sample
                weights_t2i.fill_diagonal_(0)

            # select a negative image for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()  #B 
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)    #B,

            # select a negative text for each image
            text_embeds_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_embeds_neg.append(text_embeds[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   #B,D
            text_atts_neg = torch.stack(text_atts_neg,dim=0)      #B

            text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)      #2B, d
            text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

            image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0) #2B,D
            image_atts_all = torch.cat([image_atts,image_atts],dim=0)

            output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, #2B,N
                                            attention_mask = text_atts_all,
                                            encoder_hidden_states = image_embeds_all,
                                            encoder_attention_mask = image_atts_all,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )                         

            vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
            vl_output = self.itm_head(vl_embeddings)            

            itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                                dim=0).to(image.device)
            loss_itm = F.cross_entropy(vl_output, itm_labels)     
            
            ##================= MLM ========================##                
            input_ids = text.input_ids.clone()
            labels = input_ids.clone()

            probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
            input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                        probability_matrix = probability_matrix) 
            
            with torch.no_grad():
                logits_m = self.text_encoder_m(input_ids, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds_m,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            return_logits = True,   
                                            )   
            mlm_output = self.text_encoder(input_ids, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        labels = labels,   
                                        soft_labels = F.softmax(logits_m,dim=-1),
                                        alpha = alpha
                                        )                           
            loss_mlm = mlm_output.loss        

            return loss_mlm, loss_ita, loss_itm  

        

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat) #gather from all processes
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0] #n_gpu*B

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

