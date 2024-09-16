import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import time

def convert_grayscale_to_rgb(grayscale_image):
    if grayscale_image.ndim == 2:
        color_image = np.repeat(grayscale_image[:, :, np.newaxis], 3, axis=2)
        return color_image
    return grayscale_image

def ssim_metric(img1, img2):
    img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)
    if img2.ndim == 2:
        img2 = np.repeat(img2[:, :, np.newaxis], 3, axis=2)
    return ssim(img1, img2, channel_axis=2, data_range=img2.max() - img2.min())


def f_measure(true, pred, beta=1):
    true_positive = np.sum((true == 1) & (pred == 1))
    false_positive = np.sum((true == 0) & (pred == 1))
    false_negative = np.sum((true == 1) & (pred == 0))
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-8)
    return f1_score * 100


import time

def calculate_fps(generator, test_images):
    start_time = time.time()
    for img in test_images:
        _ = generator.predict(img.reshape(1, 256, 256, 1))
    end_time = time.time()
    total_time = end_time - start_time
    fps = len(test_images) / total_time
    return fps


def drd(true, pred):
    true = true.astype(np.uint8)
    pred = pred.astype(np.uint8)
    return (np.sum(np.abs(true - pred)) / (np.sum(true) + 1e-8))
    
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))


def split2(dataset, size, h, w):
    newdataset = []
    nsize1 = 256
    nsize2 = 256
    for i in range(size):
        im = dataset[i]
        for ii in range(0, h, nsize1):
            for iii in range(0, w, nsize2):
                newdataset.append(im[ii:ii + nsize1, iii:iii + nsize2, :])
    return np.array(newdataset)

def merge_image2(splitted_images, h, w):
    image = np.zeros((h, w, 1))
    nsize1 = 256
    nsize2 = 256
    ind = 0
    for ii in range(0, h, nsize1):
        for iii in range(0, w, nsize2):
            image[ii:ii + nsize1, iii:iii + nsize2, :] = splitted_images[ind]

            ind += 1
    return np.array(image)

def getPatches(watermarked_image, clean_image, mystride):
    watermarked_patches = []
    clean_patches = []
    h = ((watermarked_image.shape[0] // 256) + 1) * 256
    w = ((watermarked_image.shape[1] // 256) + 1) * 256
    image_padding = np.ones((h, w))
    image_padding[:watermarked_image.shape[0], :watermarked_image.shape[1]] = watermarked_image
    for j in range(0, h - 256, mystride):
        for k in range(0, w - 256, mystride):
            watermarked_patches.append(image_padding[j:j + 256, k:k + 256])
    h = ((clean_image.shape[0] // 256) + 1) * 256
    w = ((clean_image.shape[1] // 256) + 1) * 256
    image_padding = np.ones((h, w)) * 255
    image_padding[:clean_image.shape[0], :clean_image.shape[1]] = clean_image
    for j in range(0, h - 256, mystride):
        for k in range(0, w - 256, mystride):
            clean_patches.append(image_padding[j:j + 256, k:k + 256] / 255)
    return np.array(watermarked_patches).reshape(-1, 256, 256, 1), np.array(clean_patches).reshape(-1, 256, 256, 1)
