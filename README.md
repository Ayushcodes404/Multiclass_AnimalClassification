# Multi-Class Animal Species Classifier

This project presents a deep learning-based image classification system designed to identify 90 different animal species. It utilizes transfer learning with MobileNetV2 and features a lightweight, real-time web interface built using Flask. The model aims to support applications in biodiversity monitoring, wildlife conservation, and educational tools.

## Project Overview

Manual classification of animal images is time-consuming and error-prone, particularly when working with large datasets or images from camera traps in the wild. This project introduces an automated, scalable solution that classifies images into animal categories quickly and accurately using deep learning.

## Key Learnings

- Designing and training Convolutional Neural Networks (CNNs)
- Applying transfer learning using MobileNetV2
- Handling multi-class image datasets
- Evaluating and improving model performance
- Developing a Flask-based web application for real-time use

## Tools and Technologies

- Programming Language: Python
- Libraries: TensorFlow, Keras, NumPy, Matplotlib, Seaborn
- Framework: Flask
- Model: MobileNetV2 (Pre-trained)
- Development Environment: Google Colab, Jupyter Notebook
- Dataset Integration: kagglehub

## Dataset

This project uses a dataset containing images of 90 different animal species. The dataset was accessed from Kaggle and used for both training and validation.

Dataset Download Link: //www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/data 

## Model Details

- Architecture: MobileNetV2 with fine-tuning
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Output Layer: Softmax for multi-class classification
- Test Accuracy: Approximately 90 percent
- Validation Accuracy: Approximately 83 percent

Trained Model Download: https://drive.google.com/file/d/1kifVag9NhtgvMtvd1Jssi_x6CDTDwq0Z/view?usp=sharing


## Methodology

1. Data Collection  
   Dataset retrieved via kagglehub.

2. Data Preprocessing  
   Images resized and normalized using ImageDataGenerator with a validation split.

3. Model Training  
   Fine-tuned MobileNetV2 for multi-class classification.

4. Evaluation  
   Model performance tested on unseen data, using accuracy and classification metrics.

5. Deployment  
   Flask-based web interface developed for users to upload images and receive predictions.

## Real-World Applications

- Wildlife conservation and biodiversity tracking
- Educational tools for species recognition
- Research support for ecologists and field biologists
- Automated processing of camera trap images

## Results

- Achieved strong classification accuracy with minimal overfitting
- Most misclassifications occurred due to poor input image quality
- The model generalizes well to unseen data
- Ready for real-time use cases with a fast inference pipeline

## Future Work

- Improve dataset quality and diversity
- Apply advanced augmentation techniques
- Integrate attention mechanisms or ensemble models
- Explore deployment to edge devices for on-field applications


## Acknowledgments

- Dataset by Sourav Banerjee (Kaggle)


## Note :
Find my week1 submission as animal_classification-checkpoint.ipynb
my week2 submission as week2.ipynb
and final submission in the internship_final directory

## Author

Ayush K Tammannavar  
AICTE Student ID: STU679af5085b1861738208520
