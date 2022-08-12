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
from  transformers.modeling_outputs import BaseModelOutput

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from glob import glob
from tqdm import tqdm
import wandb

@torch.no_grad()
def test_img2poem_gen(args, model, config, tokenizer, device, epoch=None, wandb_run=None):
    model.eval()
    datasets = [create_dataset('gen_img2poem', config)]
    samplers = [None]
    data_loader = create_loader(datasets,samplers,batch_size=[1], num_workers=[4], is_trains=[False], collate_fns=[None])[0]
    filename2result, total_ppl = {}, []
    for i, (imgnames, filenames, images, references) in tqdm(enumerate(data_loader)):
        images = images.to(device,non_blocking=True) 
        image_embeds = model.visual_encoder(images) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
        encoder_outputs= BaseModelOutput(last_hidden_state=image_embeds, attentions=image_atts)
        generate_results = model.text_encoder.generate(
            encoder_outputs=encoder_outputs, 
            encoder_attention_mask = image_atts, encoder_hidden_states=image_embeds, 
            num_beams=config['generation_cfg']['beam_size'], 
            num_return_sequences=config['generation_cfg']['beam_size'],
            max_length=config['generation_cfg']['max_length'], 
            bos_token_id=tokenizer.vocab[tokenizer.cls_token], eos_token_id=tokenizer.vocab[tokenizer.sep_token],
            mode = 'multi_modal', use_cache=False
            )
        output = tokenizer.batch_decode(generate_results,skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output = [o.replace(" ","") for o in output]

        filename2result[filenames[0]] = output
        if references[0].lower()!='null': 
            text_input = tokenizer(references, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)
            #[[101,t1,t2,...,511,102]]
            predict_logits = []
            for pos in range(1, text_input.input_ids.shape[1]): #
                if model.text_encoder_type == 'bert':
                    input_ids = torch.ones([1,pos+1], dtype=torch.long, device=device)*tokenizer.vocab[tokenizer.mask_token]
                    input_ids[:,:pos] = text_input.input_ids[:,:pos]
                    output = model.text_encoder(
                        input_ids=input_ids,
                        encoder_attention_mask = image_atts, encoder_hidden_states=image_embeds, 
                        is_decoder=True, #!
                        mode='multi_modal')
                elif model.text_encoder_type == 'gpt':
                    input_ids = text_input.input_ids[:,:pos]
                    output = model.text_encoder(
                        input_ids=input_ids,
                        encoder_attention_mask = image_atts, encoder_hidden_states=image_embeds, 
                        mode='multi_modal')
                predict_logits.append(output.logits[0,-1,:]) #bs-1
            predict_logits = torch.stack(predict_logits, dim=0) #L,D 
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # -100 index = padding token
            labels = text_input.input_ids[0,1:] #L
            xe = loss_fct(predict_logits.view(-1, model.text_encoder.config.vocab_size), labels.view(-1))
            total_ppl.append(xe) #L
    if total_ppl!=[]:
        total_ppl = torch.cat(total_ppl, dim=0)
        num_tokens = total_ppl.shape[0]  
        total_ppl = torch.pow(2, torch.mean(total_ppl))
        print('#Num_tokens={} PPL={:.2f}'.format(num_tokens,total_ppl))
    if wandb_run!=None and total_ppl!=[]:
        wandb.log({'eval/ppl': total_ppl})
    output_dir_result = os.path.join(args.output_dir, 'generation_results')
    os.makedirs(output_dir_result, exist_ok=True)
    output_file = os.path.join(output_dir_result, 
                               f'imgfile2results_epoch{epoch}.json' if epoch!=None else 'imgfile2results.json')
    with open(output_file,'w') as f:
        json.dump(filename2result, f)

def main(args, config):
    device = torch.device(args.device)
    #### Dataset #### 
    print("Creating generation dataset")
    config['test_image_dir'], config['test_poem_reference'] = args.test_image_dir, args.test_poem_reference
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder, add_sep=(config.get('mode','retrieval')=='generation'))

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if os.path.isfile(args.checkpoint): 
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict, epoch = checkpoint['model'], checkpoint['epoch']
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        msg = model.load_state_dict(state_dict,strict=True)  
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
    else:
        print(f'{args.checkpoint} does not exist')
    
    model = model.to(device)   
    test_img2poem_gen(args, model, config, tokenizer, device, epoch=epoch)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_subdir', default='output/Retrieval_flickr')        
    parser.add_argument('--test_image_dir', default='data/generation_eval/images/')
    parser.add_argument('--test_poem_reference', default='data/generation_eval/debug_img2text.pkl') 
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', default='latest')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.output_dir = os.path.join('output', os.path.basename(args.config).replace('.yaml',''), args.output_subdir)
    args.checkpoint_dir = os.path.join('output', os.path.basename(args.config).replace('.yaml',''))
    if args.checkpoint == 'latest':
        checkpoint_list = glob(os.path.join(args.checkpoint_dir, 'checkpoint_*.pth'))
        checkpoint_list = sorted(checkpoint_list, key=lambda c: int(os.path.basename(c).replace('checkpoint_','')[:-4]))
        args.checkpoint = checkpoint_list[-1]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #print(args.output_dir, os.path.isdir(args.output_dir))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
