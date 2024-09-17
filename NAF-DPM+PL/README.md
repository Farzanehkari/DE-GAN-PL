
# NAF-DPM+PL

This repository contains the **NAF-DPM+PL** model for document image enhancement using a diffusion model combined with **perceptual loss**. This approach significantly improves document image quality, especially in tasks like Optical Character Recognition (OCR).

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Contact](#contact)

## Project Overview

The **NAF-DPM+PL** model builds on the original **NAF-DPM** (Nonlinear Activation-Free Diffusion Probabilistic Model) and enhances it by incorporating **perceptual loss** for improved structural consistency and text readability in document images.

The original **NAF-DPM** model can be found [here]([https://github.com/author/NAF-DPM](https://github.com/ispamm/NAF-DPM/tree/main)). This modified version enhances document binarization tasks by maintaining both pixel-level accuracy and perceptual quality.

## Prerequisites

To run this model, you will need:

- Python >= 3.6
- PyTorch >= 1.8.1
- torchvision >= 0.9.1
- numpy >= 1.21.6
- pandas >= 1.2.4
- tqdm >= 4.50.2
- scikit-learn >= 0.24.2
- pyiqa >= 0.3.0
- opencv-python >= 4.5.3.56
- yaml >= 5.4

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Training the Model

To train the **NAF-DPM+PL** model, use the following command:

```bash
python main.py --config Binarization/conf.yml
```

- `--config`: Specify the path to your configuration file.

The model supports various configurations for training, including data paths, loss functions, and diffusion steps, all specified in the `config.yml` file.

## Model Evaluation

For evaluation, you can run the test process using the pretrained models:

```bash
python main.py --config Binarization/conf.yml --mode test
```

Ensure that the testing paths for ground truth and test images are specified in the configuration file. The model will save the evaluation metrics and results in the specified output directory.

## Contact

If you have any questions, feel free to reach out via email: karimpour.f@gmail.com

