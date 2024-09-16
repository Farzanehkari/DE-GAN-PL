import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import imageio
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.metrics import structural_similarity as ssim
import random
from tensorflow.keras.metrics import BinaryAccuracy
from utils import *
from utils import calculate_fps

from models import *
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print(f"\nEpoch {epoch+1}: Learning rate is {lr.numpy()}")

print_lr = PrintLR()

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)




def initialize_loss_arrays(epochs):
    return [np.nan] * epochs

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


input_size = (256, 256, 1)

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.bmp'):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(np.array(img.convert('L')).reshape(256, 256, 1) / 255.0)  # Normalize to [0, 1]
    return images

    

results = {
    "Epoch": [],
    "Train_Discriminator_Real_Loss": [],
    "Train_Discriminator_Fake_Loss": [],
    "Train_Generator_Adversarial_Loss": [],
    "Train_Generator_Pixelwise_Loss": [],
    "Train_Generator_Perceptual_Loss": [],
    "Val_Discriminator_Real_Loss": [],
    "Val_Discriminator_Fake_Loss": [],
    "Val_Generator_Adversarial_Loss": [],
    "Val_Generator_Pixelwise_Loss": [],
    "Val_Generator_Perceptual_Loss": [],
    "PSNR": [],
    "SSIM": [],
    "F-measure": [],
    "DRD": [],
    "FPS": []  

}


def convert_grayscale_to_rgb(grayscale_image):
    if grayscale_image.ndim == 2:
        color_image = np.repeat(grayscale_image[:, :, np.newaxis], 3, axis=2)
        return color_image
    return grayscale_image

def visualize_images(images, titles, rows=1, cols=3):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.show()

def train_gan(generator, discriminator, ep_start=1, epochs=32, batch_size=64, callbacks=[]):


    train_deg_images = load_images_from_folder('Patch2013/A/train/')
    train_clean_images = load_images_from_folder('Patch2013/B/train-gt/')

    val_deg_images = load_images_from_folder('Patch2013/A/valid/')
    val_clean_images = load_images_from_folder('Patch2013/B/valid-gt/')

    if len(train_deg_images) != len(train_clean_images):
        raise ValueError("The number of degraded and clean training images do not match.")
    if len(val_deg_images) != len(val_clean_images):
        raise ValueError("The number of degraded and clean validation images do not match.")

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    gan = get_gan_network(discriminator, generator, input_size=(256, 256, 1))

    if os.path.exists('continueTraining/generator_epoch_40.weights.h5'):
        generator.load_weights('continueTraining/generator_epoch_40.weights.h5')
        print("Loaded generator weights from epoch 40")
    if os.path.exists('continueTraining/discriminator_epoch_40.weights.h5'):
        discriminator.load_weights('continueTraining/discriminator_epoch_40.weights.h5')
        print("Loaded discriminator weights from epoch 40")

    d_loss_real_arr, d_loss_fake_arr = [], []
    generator_adversarial_loss_arr, generator_pixelwise_loss_arr, generator_perceptual_loss_arr = [], [], []
    val_d_loss_real_arr, val_d_loss_fake_arr = [], []
    val_generator_adversarial_loss_arr, val_generator_pixelwise_loss_arr, val_generator_perceptual_loss_arr = [], [], []

    for e in range(ep_start, epochs + 1):
        print(f'\n Epoch: {e}')

        total_d_loss_real, total_d_loss_fake = 0, 0
        total_generator_adversarial_loss, total_generator_pixelwise_loss, total_generator_perceptual_loss = 0, 0, 0
        val_total_d_loss_real, val_total_d_loss_fake = 0, 0
        val_total_generator_adversarial_loss, val_total_generator_pixelwise_loss, val_total_generator_perceptual_loss = 0, 0, 0

        for im in tqdm(range(len(train_deg_images)), desc=f"Epoch {e}", leave=False, disable=True):

            b_wat_batch = train_deg_images[im].reshape((1, 256, 256, 1))
            b_gt_batch = train_clean_images[im].reshape((1, 256, 256, 1))
            b_wat_batch = next(datagen.flow(b_wat_batch, batch_size=1, shuffle=False))
        
            # Ensure normalized generated images
            generated_images = generator.predict(b_wat_batch)
        
            valid = np.ones((b_gt_batch.shape[0], 16, 16, 1))
            fake = np.zeros((b_gt_batch.shape[0], 16, 16, 1))
        
            discriminator.trainable = True
            if im % 5 == 0:  
                
                d_loss_real = discriminator.train_on_batch([b_gt_batch, b_wat_batch], valid)
                d_loss_fake = discriminator.train_on_batch([generated_images, b_wat_batch], fake)
                total_d_loss_real += d_loss_real[0]
                total_d_loss_fake += d_loss_fake[0]
        
            discriminator.trainable = False
            g_loss = gan.train_on_batch([b_wat_batch], [valid, b_gt_batch, b_gt_batch])
            total_generator_adversarial_loss += g_loss[1]
            total_generator_pixelwise_loss += g_loss[2]
            total_generator_perceptual_loss += g_loss[3]


        d_loss_real_arr.append(total_d_loss_real / (len(train_deg_images) / 2))
        d_loss_fake_arr.append(total_d_loss_fake / (len(train_deg_images) / 2))
        generator_adversarial_loss_arr.append(total_generator_adversarial_loss / len(train_deg_images))
        generator_pixelwise_loss_arr.append(total_generator_pixelwise_loss / len(train_deg_images))
        generator_perceptual_loss_arr.append(total_generator_perceptual_loss / len(train_deg_images))

        print(f"Training Losses - Discriminator Real: {d_loss_real_arr[-1]:.4f}, Fake: {d_loss_fake_arr[-1]:.4f}, "
              f"Adversarial: {generator_adversarial_loss_arr[-1]:.4f}, Pixel-wise: {generator_pixelwise_loss_arr[-1]:.4f}, Perceptual: {generator_perceptual_loss_arr[-1]:.4f}")

        for im in range(len(val_deg_images)):
            b_wat_batch = val_deg_images[im].reshape((1, 256, 256, 1))
            b_gt_batch = val_clean_images[im].reshape((1, 256, 256, 1))

            if len(b_wat_batch) != 1 or len(b_gt_batch) != 1:
                continue

            generated_images = generator.predict(b_wat_batch)
            valid = np.ones((1, 16, 16, 1))
            fake = np.zeros((1, 16, 16, 1))

            d_loss_real = discriminator.train_on_batch([b_gt_batch, b_wat_batch], valid)
            d_loss_fake = discriminator.train_on_batch([generated_images, b_wat_batch], fake)
            val_total_d_loss_real += d_loss_real[0]
            val_total_d_loss_fake += d_loss_fake[0]

            g_loss = gan.train_on_batch([b_wat_batch], [valid, b_gt_batch, b_gt_batch])
            val_total_generator_adversarial_loss += g_loss[1]
            val_total_generator_pixelwise_loss += g_loss[2]
            val_total_generator_perceptual_loss += g_loss[3]

        val_d_loss_real_arr.append(val_total_d_loss_real / len(val_deg_images))
        val_d_loss_fake_arr.append(val_total_d_loss_fake / len(val_deg_images))
        val_generator_adversarial_loss_arr.append(val_total_generator_adversarial_loss / len(val_deg_images))
        val_generator_pixelwise_loss_arr.append(val_total_generator_pixelwise_loss / len(val_deg_images))
        val_generator_perceptual_loss_arr.append(val_total_generator_perceptual_loss / len(val_deg_images))

        print(f"Validation Losses - Discriminator Real: {val_d_loss_real_arr[-1]:.4f}, Fake: {val_d_loss_fake_arr[-1]:.4f}, "
              f"Adversarial: {val_generator_adversarial_loss_arr[-1]:.4f}, Pixel-wise: {val_generator_pixelwise_loss_arr[-1]:.4f}, Perceptual: {val_generator_perceptual_loss_arr[-1]:.4f}")

        psnr_value, ssim_value, f_measure_value, drd_value, fps_value  = evaluate(generator, discriminator, e, val_deg_images, val_clean_images)



        results["Epoch"].append(e)
        results["Train_Discriminator_Real_Loss"].append(d_loss_real_arr[-1])
        results["Train_Discriminator_Fake_Loss"].append(d_loss_fake_arr[-1])
        results["Train_Generator_Adversarial_Loss"].append(generator_adversarial_loss_arr[-1])
        results["Train_Generator_Pixelwise_Loss"].append(generator_pixelwise_loss_arr[-1])
        results["Train_Generator_Perceptual_Loss"].append(generator_perceptual_loss_arr[-1])
        results["Val_Discriminator_Real_Loss"].append(val_d_loss_real_arr[-1])
        results["Val_Discriminator_Fake_Loss"].append(val_d_loss_fake_arr[-1])
        results["Val_Generator_Adversarial_Loss"].append(val_generator_adversarial_loss_arr[-1])
        results["Val_Generator_Pixelwise_Loss"].append(val_generator_pixelwise_loss_arr[-1])
        results["Val_Generator_Perceptual_Loss"].append(val_generator_perceptual_loss_arr[-1])
        results["PSNR"].append(psnr_value)
        results["SSIM"].append(ssim_value)
        results["F-measure"].append(f_measure_value)
        results["DRD"].append(drd_value)
        results["FPS"].append(fps_value)  


        generator.save_weights(f'weightsDIBCO2013/generator_epoch_{e}.weights.h5')
        discriminator.save_weights(f'weightsDIBCO2013/discriminator_epoch_{e}.weights.h5')
        
        visualize_images([b_wat_batch[0], generated_images[0], b_gt_batch[0]], 
                             ['Degraded', 'Generated', 'Ground Truth'])
    
    df = pd.DataFrame(results)
    df.to_excel('training_results.xlsx', index=False)

    plot_losses(epochs, d_loss_real_arr, d_loss_fake_arr, generator_adversarial_loss_arr, generator_pixelwise_loss_arr,
                generator_perceptual_loss_arr, val_d_loss_real_arr, val_d_loss_fake_arr, val_generator_adversarial_loss_arr,
                val_generator_pixelwise_loss_arr, val_generator_perceptual_loss_arr)

def evaluate(generator, discriminator, epoch, val_deg_images, val_clean_images):
    psnr_total = 0
    ssim_total = 0
    f_measure_total = 0
    drd_total = 0
    num_images = len(val_deg_images)

    for i in range(num_images):
        deg_image = val_deg_images[i].reshape(1, 256, 256, 1)
        clean_image = val_clean_images[i].reshape(1, 256, 256, 1)
        generated_image = generator.predict(deg_image)
        
        deg_image_rgb = convert_grayscale_to_rgb(deg_image.squeeze())
        clean_image_rgb = convert_grayscale_to_rgb(clean_image.squeeze())
        generated_image_rgb = convert_grayscale_to_rgb(generated_image.squeeze())
        
        psnr_total += psnr(clean_image_rgb, generated_image_rgb)
        ssim_total += tf.reduce_mean(tf.image.ssim(tf.cast(clean_image_rgb, tf.float32), tf.cast(generated_image_rgb, tf.float32), max_val=1.0)).numpy()
        f_measure_total += f_measure(clean_image_rgb, generated_image_rgb)
        drd_total += drd(clean_image_rgb, generated_image_rgb)
    
    fps = calculate_fps(generator, val_deg_images)
    
    return psnr_total / num_images, ssim_total / num_images, f_measure_total / num_images, drd_total / num_images, fps

def plot_losses(epochs, d_loss_real_arr, d_loss_fake_arr, generator_adversarial_loss_arr, generator_pixelwise_loss_arr,
                generator_perceptual_loss_arr, val_d_loss_real_arr, val_d_loss_fake_arr, val_generator_adversarial_loss_arr,
                val_generator_pixelwise_loss_arr, val_generator_perceptual_loss_arr):

    epoch_range = range(1, len(d_loss_real_arr) + 1)

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 2, 1)
    plt.plot(epoch_range, d_loss_real_arr, label='Train Discriminator Real Loss')
    plt.plot(epoch_range, val_d_loss_real_arr, label='Val Discriminator Real Loss')
    plt.legend()
    plt.title('Discriminator Real Loss')

    plt.subplot(3, 2, 2)
    plt.plot(epoch_range, d_loss_fake_arr, label='Train Discriminator Fake Loss')
    plt.plot(epoch_range, val_d_loss_fake_arr, label='Val Discriminator Fake Loss')
    plt.legend()
    plt.title('Discriminator Fake Loss')

    plt.subplot(3, 2, 3)
    plt.plot(epoch_range, generator_adversarial_loss_arr, label='Train Generator Adversarial Loss')
    plt.plot(epoch_range, val_generator_adversarial_loss_arr, label='Val Generator Adversarial Loss')
    plt.legend()
    plt.title('Generator Adversarial Loss')

    plt.subplot(3, 2, 4)
    plt.plot(epoch_range, generator_pixelwise_loss_arr, label='Train Generator Pixelwise Loss')
    plt.plot(epoch_range, val_generator_pixelwise_loss_arr, label='Val Generator Pixelwise Loss')
    plt.legend()
    plt.title('Generator Pixelwise Loss')

    plt.subplot(3, 2, 5)
    plt.plot(epoch_range, generator_perceptual_loss_arr, label='Train Generator Perceptual Loss')
    plt.plot(epoch_range, val_generator_perceptual_loss_arr, label='Val Generator Perceptual Loss')
    plt.legend()
    plt.title('Generator Perceptual Loss')

    plt.savefig('training_loss_plots.png')
    plt.close()


def predict_and_visualize(generator, test_images_folder, output_folder):
    test_deg_images = load_images_from_folder(test_images_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, deg_image in enumerate(test_deg_images):
        h = ((deg_image.shape[0] // 256) + 1) * 256
        w = ((deg_image.shape[1] // 256) + 1) * 256

        test_padding = np.zeros((h, w, 1)) + 1
        test_padding[:deg_image.shape[0], :deg_image.shape[1], :] = deg_image

        test_image_p = split2(test_padding.reshape(1, h, w, 1), 1, h, w)
        predicted_list = []
        for l in range(test_image_p.shape[0]):
            predicted_image = generator.predict(test_image_p[l].reshape(1, 256, 256, 1))
            predicted_image = predicted_image.squeeze(0)
            predicted_list.append(predicted_image)

        predicted_image = np.array(predicted_list)
        predicted_image = merge_image2(predicted_image, h, w)
        predicted_image = predicted_image[:deg_image.shape[0], :deg_image.shape[1]]
        predicted_image = predicted_image.reshape(predicted_image.shape[0], predicted_image.shape[1])
        predicted_image = (predicted_image[:, :] * 255).astype(np.uint8)

        imageio.imwrite(os.path.join(output_folder, f'predicted_{i}.bmp'), predicted_image)


# Initialize models
generator = generator_model(biggest_layer=1024)
discriminator = discriminator_model()

# Compile models
discriminator.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=[BinaryAccuracy()])
generator.compile(optimizer=Adam(learning_rate=1e-5), loss=['mse', 'binary_crossentropy', perceptual_loss_fn], metrics=[psnr_metric, ssim_metric])

# Train the GAN
#train_gan(generator, discriminator, ep_start=2, epochs=3, batch_size=64)
#train_gan(generator, discriminator, ep_start=2, epochs=3, batch_size=64, callbacks=[lr_scheduler])
train_gan(generator, discriminator, ep_start=129, epochs=256, batch_size=128, callbacks=[lr_scheduler, reduce_lr])

# Predict and visualize results
predict_and_visualize(generator, 'Patch2013/A/test/', 'ResultsDIBCO2013/test_results/')
