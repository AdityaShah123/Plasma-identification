# Plasma-identification

## Project Overview

This project aims to tackle the challenge of visualizing reactions within a nuclear reactor, where full imaging is not possible. To address this, we developed a model that reconstructs full images from masked images. The primary goal is to convert masked images into unmasked, full images using deep learning techniques, specifically Convolutional Neural Networks (CNN) and the U-Net architecture.

---

## Problem Statement

In nuclear reactors, visualizing the reactions and processes happening inside is a significant challenge. We aimed to create a simplified model that can reconstruct images of the internal processes using a masked-to-unmasked image conversion method. Our solution leverages CNNs and U-Net to accurately generate full images from partially observed ones.

---

## Dataset Generation

To train our model, we needed a dataset that didn't exist. Therefore, we generated a custom dataset using image masking techniques with OpenCV. The steps involved:
1. **Creating Full Images (ytrain)**: Using Gaussian functions to simulate full images.
2. **Generating Masked Images (xtrain)**: Applying masking matrices (1’s and 0’s) to the full images, simulating the partial observations.
3. **Training Dataset**: We created 1200 pairs of masked and unmasked images to train our model.

---

## Methodology

### Convolutional Neural Network (CNN)
CNNs are well-suited for analyzing visual data, especially for tasks like image recognition and segmentation. The network learns to identify important features in images by applying convolutional filters, pooling layers, and fully connected layers.

Our model uses a CNN to:
- Identify the unique features in masked images.
- Reconstruct the unmasked version of the image.

### U-Net Framework
We employed the U-Net architecture, known for its efficiency with limited labeled data, to handle the image-to-image translation task. The U-Net consists of two main components:
1. **Encoder**: Reduces the spatial dimensions of the input image while increasing feature channels. This helps the model identify and abstract important features.
2. **Decoder**: Upscales the feature maps back to the original image size, reconstructing the unmasked image with fine details.

---

## Results

We trained the model using the custom dataset and evaluated its performance using the following metrics:
- **Structural Similarity Index (SSIM)**: To measure the similarity between the reconstructed and the original unmasked image.
- **Root Mean Squared Error (RMSE)**: To quantify the difference between the two images.
- **Residual Analysis**: To further assess the performance of the model.

---

## Learnings

Throughout this project, we gained experience and knowledge in:
- **Deep Learning**: Understanding its application in solving complex visual problems.
- **Convolutional Neural Networks (CNNs)**: How CNNs work and how to implement them for image-based tasks.
- **U-Net Architecture**: Learning the workings of U-Net and its application in medical and other scientific fields.
- **Evaluation Metrics**: Using SSIM, RMSE, and residual analysis to evaluate model performance.

