import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

#from models.model_retrieval import ALBEF
from models.model_pretrain import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from glob import glob
from tqdm import tqdm

@torch.no_grad()
def test_img2poem(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_forward_size = text_bs*100 #25600
    text_embeds_list_file = os.path.join(args.output_dir, 'text_embeds_list.bin')
    text_embeds_tensor_file = os.path.join(args.output_dir, 'text_embeds_tensor.bin')
    if os.path.isfile(text_embeds_list_file):
        print(f'Load text features from {text_embeds_list_file}')
        text_embeds = torch.load(text_embeds_list_file, map_location=args.device)
        assert len(text_embeds)
    else:
        print(f'Computing text features, #{num_text} ...')
        text_embeds = []
        for i in tqdm(range(0, num_text, text_forward_size)):
            texts_forward = texts[i: min(num_text, i+text_forward_size)]
            for i in range(0, len(texts_forward), text_bs):
                text = texts_forward[i: min(num_text, i+text_bs)]
                text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
                text_output = model.text_encoder.bert(text_input.input_ids, attention_mask = text_input.attention_mask, return_dict=True, mode='text')  
                text_feat = text_output.last_hidden_state
                text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))   #B,D     
                text_embeds.extend([e for e in text_embed])   
        print('Saving as '+text_embeds_list_file)
        torch.save(text_embeds, text_embeds_list_file)
        #text_embeds_tensor = torch.stack(text_embeds, dim=0)
        # print('Saving as '+text_embeds_tensor_file)
        # torch.save(text_embeds_tensor, text_embeds_tensor_file)

    print(f'Retrieve TOP-K={config["test_k"]} for each image ...')
    img2results = {}
    for imgname, filename, image in data_loader: 
        print(filename)
        results = {}
        image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat[:,0,:])   #1,D         
        image_embed = F.normalize(image_embed,dim=-1)    
        sims_matrix = []  
        for i in range(0, num_text, text_forward_size):
            text_embeds_forward = text_embeds[i: min(num_text, i+text_forward_size)]
            text_embeds_forward = torch.stack(text_embeds_forward, dim=0)
            sims_matrix_ = image_embed @ text_embeds_forward.t() #F
            sims_matrix.append(sims_matrix_[0])
        sims_matrix = torch.cat(sims_matrix)
        topk_sim, topk_idx = torch.topk(sims_matrix, k=config['test_k'])  #N, 
        topk_sim, topk_idx = topk_sim.tolist(), topk_idx.tolist() #K
        results['topk'] = [
            {'id':txt_id, 'text':texts[txt_id], 'sim':sim}
            for sim, txt_id in zip(topk_sim, topk_idx)
        ]
        topk_texts = [texts[txt_id] for txt_id in topk_idx]
        #re-forward (topk_texts)
        #mode = 'fusion'
        topk_text_input = tokenizer(topk_texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        topk_text_output = model.text_encoder.bert(topk_text_input.input_ids, attention_mask = topk_text_input.attention_mask, return_dict=True, mode='text')  
        topk_text_feat = topk_text_output.last_hidden_state
        image_feats = torch.tile(image_feat, dims=(len(topk_texts),1,1)) #1,L,D -> K,L,D
        image_atts = torch.ones(image_feats.size()[:-1],dtype=torch.long).to(image.device)
        output_fusion = model.text_encoder.bert(encoder_embeds = topk_text_feat, 
                                        attention_mask = topk_text_input.attention_mask,
                                        encoder_hidden_states = image_feats,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )   
        rerank_score = model.itm_head(output_fusion.last_hidden_state[:,0,:])[:,1] #K
        rerank_idx = torch.argsort(rerank_score*-1).tolist()
        results['rerank'] = [
            {'id': idx, 'text': texts[topk_idx[idx]], 'rerank_score': rerank_score[idx].item()}
            for idx in rerank_idx
        ]
        img2results[filename[0]] = results
        with open(os.path.join(args.output_dir, f'{imgname[0]}_results.json'),'w') as f:
            json.dump(results, f)
        print(f'Save as {os.path.join(args.output_dir, f"{imgname[0]}_results.json")}')
    with open(os.path.join(args.output_dir, f'results.json'),'w') as f:
        json.dump(img2results, f)    
    return


def main(args, config):
    device = torch.device(args.device)


    #### Dataset #### 
    print("Creating retrieval dataset")
    config['test_image_dir'], config['text_file'] = args.test_image_dir, args.text_file
    test_dataset = create_dataset('re_img2poem', config)  
    samplers = [None]
    
    test_loader = create_loader([test_dataset],samplers,
                                batch_size=[1],
                                num_workers=[4], is_trains=[False], collate_fns=[None])[0]
       
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if os.path.isfile(args.checkpoint): 
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        '''
        used for model_retrieval.ALBEF
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]               
        '''
        msg = model.load_state_dict(state_dict,strict=True)  
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
    else:
        print(f'{args.checkpoint} does not exist')
    
    model = model.to(device)   
    
    test_img2poem(args, model, test_loader, tokenizer, device, config)
    return
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_subdir', default='output/Retrieval_flickr')        
    parser.add_argument('--test_image_dir', default='../CLIP_distillation/data/images_evaluation/')
    parser.add_argument('--text_file', default='../others/ancient_poems/ancient_poems_short_all_txtonly.json') 
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', default='latest')
    parser.add_argument('--test_k', default=50, type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config['test_k'] = args.test_k
    args.output_dir = os.path.join('output', os.path.basename(args.config).split('.')[0], args.output_subdir)
    args.checkpoint_dir = os.path.join('output', os.path.basename(args.config).split('.')[0])
    print(args.checkpoint_dir)
    if args.checkpoint == 'latest':
        args.checkpoint = sorted(glob(os.path.join(args.checkpoint_dir, 'checkpoint_*.pth')))[-1]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
