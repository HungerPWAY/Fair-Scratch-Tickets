import numpy as np
import os
import cv2
import pandas as pd
import tarfile
import tqdm


import os
from os.path import join
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from utils.utils import list_files
from natsort import natsorted
import random
import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import DataLoader
import copy

ATTRS_NAME = "LFW/lfw_attributes.txt"  # http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
IMAGES_NAME = "LFW/lfw-deepfunneled.tgz"  # http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
RAW_IMAGES_NAME = "LFW/lfw.tgz"  # http://vis-www.cs.umass.edu/lfw/lfw.tgz


def decode_image_from_raw_bytes(raw_bytes):
    """
    convert the raw matrix into image and change the color system to RGB
    """
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_lfw_dataset(
        use_raw=False,
        dx=16, dy=16,
        dimx=256, dimy=256):

    # read attrs
    df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])
    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))

    # read photos
    all_photos = []
    photo_ids = []
    
    # tqdm in used to show progress bar while reading the data in a notebook here, you can change 
    # tqdm_notebook to use it outside a notebook
    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in tqdm.tqdm_notebook(f.getmembers()):
            #only process image files from the compressed data
            if m.isfile() and m.name.endswith(".jpg"):
                # prepare image
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                # crop only faces and resize it
                #img = img[dy:-dy, dx:-dx]
                
                img = cv2.resize(img, (dimx, dimy))
                img = img[dx:-dx,dy:-dy]
                #print(img.shape)
                # parse person and append it to the collected data
                fname = os.path.split(m.name)[-1]
                fname_splitted = fname[:-4].replace('_', ' ').split()
                person_id = ' '.join(fname_splitted[:-1])
                photo_number = int(fname_splitted[-1])
                if (person_id, photo_number) in imgs_with_attrs:
                    all_photos.append(img)
                    photo_ids.append({'person': person_id, 'imagenum': photo_number})

    photo_ids = pd.DataFrame(photo_ids)
    all_photos = np.stack(all_photos).astype('uint8')

    # preserve photo_ids order!
    all_attrs = photo_ids.merge(df_attrs, on=('person', 'imagenum')).drop(["person", "imagenum"], axis=1)

    return all_photos, all_attrs.values.astype(np.float32), list(all_attrs)


class LFWDataset(VisionDataset):


    def __init__(self, split=None, transform=None, target_transform=None):
        
        super(LFWDataset, self).__init__(root=None, transform=transform,
                                             target_transform=target_transform)
        if split is not None:
            self.split = split
        else:
            raise ValueError("Split is None!!!")
        
        images, attrs_value, self.attrs_type = load_lfw_dataset(False)
        pos_idx = np.where(attrs_value[:,self.attrs_type.index("Male")]>0)[0]
        neg_idx= np.where(attrs_value[:,self.attrs_type.index("Male")]<0)[0]
        attrs_value[attrs_value < 0] = 0
        attrs_value[attrs_value > 0] = 1 
        self.target_type = "Smiling"
        random.seed(1)
        random.shuffle(pos_idx)
        random.seed(1)
        random.shuffle(neg_idx)
        num_all_images = len(images)
        

        all_list = np.array(list(range(len(images))))
        random.seed(0)
        random.shuffle(all_list)

        
        if split == 'train':
            all_idx = all_list[:6000]
        elif split == 'val':
            all_idx = all_list[6000:6000+3600]
        else:
            all_idx = all_list[6000+3600:]
        

        '''
        if split == 'train':
            #split_ratio = 6000/num_all_iamges
            split_pos_idx = pos_idx[:4647]
            split_neg_idx = neg_idx[:1353]
        elif split == 'val':
            #split_ratio = 3600/num_all_iamges
            split_pos_idx = pos_idx[4647:4647+2788]
            split_neg_idx = neg_idx[1353:1353+812]
        else:
            #split_ratio = (num-3600/num_all_iamges
            split_pos_idx = pos_idx[4647+2788:]
            split_neg_idx = neg_idx[1353+812:]

        Num_fem = len(split_neg_idx)
        Num_male = len(split_pos_idx)
        Num_fem_sens = np.sum(attrs_value[split_neg_idx,self.attrs_type.index(self.target_type)]>0)
        Num_male_sens = np.sum(attrs_value[split_pos_idx,self.attrs_type.index(self.target_type)]>0)
        #print(split_pos_idx)
        all_idx = list(split_pos_idx)
        #print(all_idx, len(all_idx))
        all_idx.extend(list(split_neg_idx))
        
        random.seed(1)
        random.shuffle(all_idx)
        '''
        print(len(all_idx))
        self.images=images[all_idx,:,:,:]
        self.attrs_value = attrs_value[all_idx,:]
        
        Num_fem = np.sum(self.attrs_value[:,0]==0)
        Num_male = np.sum(self.attrs_value[:,0]>0)
        
        Num_fem_sens = np.sum(self.attrs_value[self.attrs_value[:,0]==0,self.attrs_type.index(self.target_type)]>0)
        Num_male_sens = np.sum( self.attrs_value[self.attrs_value[:,0] > 0,self.attrs_type.index(self.target_type)]>0)
        

        self.Nmatrix = np.array([[ Num_fem- Num_fem_sens, Num_fem_sens],
                                   [Num_male - Num_male_sens, Num_male_sens]])
        
        print(self.Nmatrix)

    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        #image = Image.open(image_path, mode='r').convert('RGB')

        image = self.images[idx,:]
        if self.transform:
            image = self.transform(image)

        return image, self.attrs_value[idx,self.attrs_type.index(self.target_type)], self.attrs_value[idx,self.attrs_type.index("Male")]


class LFW:
    def __init__(self,args):
        super(LFW, self).__init__()
        use_cuda = torch.cuda.is_available()

            # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform_list = [
                        transforms.ToTensor(),
                        normalize
                        ]
        train_transform = transforms.Compose(transform_list)
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = LFWDataset(split = 'train', transform=train_transform)
        self.train_Nmatrix = train_dataset.Nmatrix
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        val_dataset = LFWDataset(split = 'val', transform=test_transform)
        self.val_Nmatrix = val_dataset.Nmatrix
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

        test_dataset = LFWDataset(split = 'test', transform=test_transform)
        self.test_Nmatrix = test_dataset.Nmatrix
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    

