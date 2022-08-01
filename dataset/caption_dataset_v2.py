import json, pickle
import os
import random, base64, io

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

#from dataset.utils import pre_caption #
from dataset.read_tsv import TSVFile

def img_from_base64(imagestring, color=True):
    img_str = base64.b64decode(imagestring)
    try:
        if color:
            r = Image.open(io.BytesIO(img_str)).convert('RGB')
            return r
        else:
            r = Image.open(io.BytesIO(img_str)).convert('L')
            return r
    except:
        return None

class pretrain_dataset_v2(Dataset):
    def __init__(self, ann_file, transform, max_words=30):  
        self.text_file = ann_file['text_file']   
        self.id2text = json.load(open(self.text_file,'r'))
        assert '.tsv' in ann_file['img_file']
        self.tsv = TSVFile(ann_file['img_file'])
        self.total_num = self.tsv.num_rows()
        assert '.pkl' in ann_file['img2text']
        self.imgid2textid = pickle.load(open(ann_file['img2text'],'rb'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return self.total_num
    
    def get_image_item(self, i):
        #t0 = time.time()
        row = self.tsv.seek(i)

        #print('seek', time.time()-t0)
        #t0 = time.time()

        image = img_from_base64(row[-1])
        #print('img from base64', time.time()-t0)
        #t0 = time.time()
        return image  

    def __getitem__(self, index):    
        
        image = self.get_image_item(index)
        image = self.transform(image)

        txt_id = self.imgid2textid[index]
        caption = self.id2text[txt_id] #to-preprocess ? or tokenize
        # if type(ann['caption']) == list:
        #     caption = pre_caption(random.choice(ann['caption']), self.max_words)
        # else:
        #     caption = pre_caption(ann['caption'], self.max_words)
      
        # image = Image.open(ann['image']).convert('RGB')   
                
        return index, image, caption
            