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

from glob import glob
from collections import defaultdict

def is_image(filename):
    for ext in ['.png', '.jpg', '.jpeg']:
        if ext in filename:
            return True
    return False
class re_img2poem_test_dataset(Dataset):
    def __init__(self, test_img_dir, text_file, transform):
        self.img2path = []
        self.transform = transform
        for filename in os.listdir(test_img_dir):
            if is_image(filename):
                imgname = os.path.basename(filename).split('.')[0]
                self.img2path.append([imgname, os.path.join(test_img_dir, filename)])
        self.text = json.load(open(text_file, 'r'))

    def __len__(self):
        return len(self.img2path)

    def __getitem__(self, index): 
        imgname, filename = self.img2path[index]
        image = Image.open(filename).convert('RGB')   
        image = self.transform(image)     
        return imgname, filename, image

class gen_img2poem_test_dataset(Dataset):
    def __init__(self, test_img_dir, test_poem_reference, transform):
        self.img2path = []
        self.transform = transform
        for filename in os.listdir(test_img_dir):
            if is_image(filename):
                imgname = os.path.basename(filename).split('.')[0]
                self.img2path.append([imgname, os.path.join(test_img_dir, filename)])
        if test_poem_reference==None:
            self.test_poem_reference = defaultdict(lambda :'null')
        else:
            test_poem_reference = pickle.load(open(test_poem_reference, 'rb'))
            self.test_poem_reference = defaultdict(lambda :'null', test_poem_reference)

    def __len__(self):
        return len(self.img2path)

    def __getitem__(self, index): 
        imgname, filename = self.img2path[index]
        image = Image.open(filename).convert('RGB')   
        image = self.transform(image)     
        return imgname, filename, image, self.test_poem_reference[imgname]

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
    def __init__(self, ann_file, transform, sample_caption=False, max_words=30):  
        self.text_file = ann_file['text_file']   
        self.id2text = json.load(open(self.text_file,'r'))
        assert '.tsv' in ann_file['img_file']
        self.tsv = TSVFile(ann_file['img_file'])
        self.total_num = self.tsv.num_rows()
        assert '.pkl' in ann_file['img2text']
        self.imgid2textids = pickle.load(open(ann_file['img2text'],'rb'))
        if type(self.imgid2textids[0])!=list:
            self.imgid2textids = [[top1] for top1 in self.imgid2textids]
        self.transform = transform
        self.max_words = max_words
        self.sample_caption = sample_caption
        
        
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

        if self.sample_caption:
            txt_id = random.choice(self.imgid2textids[index])
        else:
            txt_id = self.imgid2textids[index][0]

        caption = self.id2text[txt_id]
        # if type(ann['caption']) == list:
        #     caption = pre_caption(random.choice(ann['caption']), self.max_words)
        # else:
        #     caption = pre_caption(ann['caption'], self.max_words)
      
        # image = Image.open(ann['image']).convert('RGB')   
                
        return index, image, caption
            