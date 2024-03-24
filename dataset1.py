import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None,train = True):

        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        
   
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
      
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = []
      
        mask.append(cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        

        img_normalized = img.astype('float32')
        img = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized)-np.min(img_normalized))) 
        img = img.transpose(2, 0, 1)
       
        mask = mask.astype('float32') / 255.
        mask = mask.transpose(2, 0, 1)
        
      
        
        return img, mask, {'img_id': img_id}
