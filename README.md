# Deep-Learning-Project


---

# 🖼️ Object Recognition with ResNet50

Welcome to the **Object Recognition** project! In this project, we explore the fascinating world of deep learning and computer vision by building a powerful image classification model using the **ResNet50** architecture. By leveraging this state-of-the-art convolutional neural network, our model achieves high accuracy in recognizing and classifying objects from images.

## 🚀 Project Overview

The goal of this project is to create a robust object recognition system capable of identifying objects from various categories in a dataset of over **5000 images**. Through the power of transfer learning and fine-tuning, we successfully boost our model's accuracy from **47% to an impressive 93%**.

### Key Features:
- **Pretrained ResNet50**: Leveraged transfer learning with the pretrained ResNet50 model to accelerate training.
- **High Accuracy**: Achieved **93% accuracy** after hyperparameter tuning and optimization.
- **5000+ Images Dataset**: Utilized a large dataset from **Kaggle** for training and validation.
- **Image Augmentation**: Applied data augmentation techniques like flipping, rotation, and scaling to enhance model generalization.

## 🛠️ Tools & Technologies

- **Python**: Core language for development
- **TensorFlow** and **Keras**: Deep learning frameworks for building and training the ResNet50 model
- **NumPy**, **pandas**: Data manipulation and analysis
- **OpenCV**, **Matplotlib**, **Seaborn**: For image processing and visualization

## 📊 Data Overview

The dataset consists of over **5000 labeled images** belonging to different object categories. Key steps include:
- **Preprocessing**: Resizing images and normalizing pixel values
- **Augmentation**: Augmenting images to prevent overfitting
- **Splitting**: Dividing the dataset into training and validation sets

## 🏗️ Project Workflow

1. **Data Preprocessing**: Cleaned and prepared the image data for training.
2. **Transfer Learning with ResNet50**: Loaded the pretrained ResNet50 model, excluding the top layers, and added custom layers to adapt it for our specific dataset.
3. **Model Training**: Trained the model using the **Adam optimizer** and **categorical cross-entropy loss**, incorporating early stopping to avoid overfitting.
4. **Evaluation**: Evaluated the model’s performance on the validation set using accuracy, precision, recall, and confusion matrix.
5. **Optimization**: Fine-tuned hyperparameters such as learning rate and batch size to achieve **93% accuracy**.



## 🏅 Results

After extensive training and fine-tuning, our ResNet50-based model achieved:
- **93% accuracy** on the validation set
- High precision and recall, indicating reliable object recognition across categories

## 📊 Visualizations

We’ve included several visualizations to help you understand the model’s performance:
- **Training & Validation Accuracy**: Visualizes how the model improves over time
- **Confusion Matrix**: Displays the classification performance across different object categories
- **Sample Predictions**: Shows some sample images along with the predicted and actual labels

