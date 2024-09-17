import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop, RandomVerticalFlip
from PIL import Image
import cv2
import numpy as np

def ImageTransform(loadSize):
    return {"train": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
    ]), "test": Compose([
        ToTensor(),
    ]), "train_gt": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
    ])}

class DocData(Dataset):
    def __init__(self, path_img, path_gt, loadSize, mode=1):
        super().__init__()
        self.path_gt = path_gt
        self.path_img = path_img
        self.data_gt = sorted([f for f in os.listdir(path_gt) if self.is_image_file(f)])
        self.data_img = sorted([f for f in os.listdir(path_img) if self.is_image_file(f)])
        
        # Debug: Print number of images and list contents of the directories
        print(f"Number of ground truth images in {path_gt}: {len(self.data_gt)}")
        print(f"Number of input images in {path_img}: {len(self.data_img)}")
        print(f"Contents of {path_img}: {os.listdir(path_img)}")
        
        if len(self.data_gt) != len(self.data_img):
            raise ValueError("The number of images and ground truth images must be the same.")
        
        self.mode = mode
        if mode == 1:
            self.ImgTrans = (ImageTransform(loadSize)["train"], ImageTransform(loadSize)["train_gt"])
        else:
            self.ImgTrans = ImageTransform(loadSize)["test"]

    def __len__(self):
        return len(self.data_gt)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.path_gt, self.data_gt[idx])
        img_path = os.path.join(self.path_img, self.data_img[idx])

        gt = cv2.imread(gt_path, 0)
        img = cv2.imread(img_path, 0)

        # Debug: Check if the images are loaded correctly
        if gt is None:
            raise ValueError(f"Ground truth image not found or could not be loaded: {gt_path}")
        if img is None:
            raise ValueError(f"Input image not found or could not be loaded: {img_path}")

        if self.mode == 1:
            gt = Image.fromarray(np.uint8(gt))
            img = Image.fromarray(np.uint8(img))

            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.ImgTrans[0](img)
            torch.manual_seed(seed)
            gt = self.ImgTrans[1](gt)
        else:
            img = Image.fromarray(np.uint8(img))
            gt = Image.fromarray(np.uint8(gt))
            img = self.ImgTrans(img)
            gt = self.ImgTrans(gt)
        
        name = self.data_img[idx]
        return img, gt, name

    def is_image_file(self, filename):
        IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        return filename.lower().endswith(IMG_EXTENSIONS)
