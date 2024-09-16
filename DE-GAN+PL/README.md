Here’s a README that focuses exclusively on the **GAN-based** model, as requested:

---

# DE-GAN+PL: Document Image Enhancement Using Generative Adversarial Networks with Perceptual Loss

This repository contains the implementation of **DE-GAN+PL**, a GAN-based model enhanced with **perceptual loss** to improve the quality of degraded document images. This model builds upon the original **DE-GAN** architecture, incorporating perceptual loss to better preserve structural details and enhance text readability, making it especially effective for tasks like Optical Character Recognition (OCR).

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Contact](#contact)

## Project Overview
Document images often suffer from various forms of degradation, including noise, stains, and fading, which impair readability and OCR accuracy. **DE-GAN+PL** is an enhanced GAN model designed to address these challenges by focusing on both pixel accuracy and structural integrity.

- **Perceptual Loss**: The key enhancement in **DE-GAN+PL** is the addition of perceptual loss, calculated using a pre-trained **VGG19** network. This loss ensures that high-level features such as text clarity and document layout are preserved during the enhancement process.
- **Metrics**: The model’s performance is evaluated using **PSNR**, **SSIM**, **F-measure**, **Pseudo-F-measure (Fps)**, and **OCR accuracy**, focusing on both local pixel fidelity and global structural consistency.

Example results demonstrating the effectiveness of **DE-GAN+PL**:
![results_comparison](insert-your-image-path)

## Prerequisites
The required software and libraries are:
* Python >= 3.6
* TensorFlow >= 2.0.0
* numpy >= 1.18.5
* matplotlib >= 3.1.3
* scikit-image >= 0.17.2
* imageio >= 2.9.0
* tqdm >= 4.47.0
* pandas >= 1.1.0

## Usage

### Training the Model

To train the **DE-GAN+PL** model, use the following command:

```bash
python train.py --model de-gan --data_dir <path_to_data> --epochs <num_epochs>
```
- `--data_dir`: Specify the path to your training and validation datasets.
- `--epochs`: Number of training epochs (default is 100).

### Model Evaluation

To evaluate the trained model and enhance document images:

```bash
python predict.py --input_folder <folder_path> --output_folder <output_path>
```
- `--input_folder`: Path to the folder containing degraded document images.
- `--output_folder`: Path to save the enhanced document images.

The model automatically loads pre-trained weights if available.

## Contact
For any questions or inquiries, feel free to [email](karimpour.f@gmail.com).


