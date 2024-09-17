import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop, RandomVerticalFlip
from PIL import Image
import cv2
import numpy as np

def ImageTransform(loadSize):
    return {
        "train": Compose([
            RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
            RandomAffine(10, fill=255),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ToTensor(),
        ]),
        "test": Compose([
            ToTensor(),
        ]),
        "train_gt": Compose([
            RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
            RandomAffine(10, fill=255),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ToTensor(),
        ])
    }

class DocData(Dataset):
    def __init__(self, path_img, path_gt, loadSize, mode=1):
        super().__init__()
        self.path_gt = path_gt
        self.path_img = path_img
        self.data_gt = sorted([f for f in os.listdir(path_gt) if self.is_image_file(f)])
        self.data_img = sorted([f for f in os.listdir(path_img) if self.is_image_file(f)])

        # Debug statements
        print(f"Ground truth path: {self.path_gt}")
        print(f"Image path: {self.path_img}")
        print(f"Found {len(self.data_gt)} ground truth images")
        print(f"Found {len(self.data_img)} images")

        if not self.data_gt or not self.data_img:
            raise ValueError("No images found in the specified directories.")

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

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if gt is None or img is None:
            raise ValueError(f"Error loading images: {img_path} or {gt_path}")

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
        IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')
        return filename.lower().endswith(IMG_EXTENSIONS)
