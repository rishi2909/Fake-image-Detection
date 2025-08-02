# 🧠 Fake Image Detection using InceptionV3

This project implements a deep learning-based solution to detect **fake or AI-generated images** using **Transfer Learning with InceptionV3**. It aims to classify images as either **real** or **fake**, with applications in digital forensics, media integrity, and security.

---

## 🎯 Objective

To build a reliable binary classification model that can identify manipulated or artificially generated images with high accuracy using pre-trained convolutional neural networks.

---

## 📂 Dataset

- Images are divided into three folders: `Train`, `Validation`, and `Test`
- Each folder contains two classes: `real` and `fake`
- Image size: Resized to **299x299** (required for InceptionV3)

---

## 🧠 Model Architecture

- Base Model: **InceptionV3** (pre-trained on ImageNet, `include_top=False`)
- Added:
  - `GlobalAveragePooling2D`
  - `Dropout` layers to reduce overfitting
  - `Dense` layer with 128 units (ReLU)
  - Final `Dense` layer with 1 unit (Sigmoid) for binary classification
- Two-Phase Training:
  1. **Transfer Learning** (InceptionV3 frozen)
  2. **Fine-Tuning** (Top 50 layers unfrozen)

---

## 🧪 Results

| Dataset     | Accuracy | Loss   |
|-------------|----------|--------|
| **Train**   | 95.99%   | 0.1099 |
| **Validation** | 88.08%   | 0.2910 |
| **Test**    | 81.17%   | 0.4931 |

---

## 📦 Libraries Used

- TensorFlow / Keras
- InceptionV3 (`keras.applications`)
- NumPy
- Matplotlib
- OpenCV (for image preprocessing)
- ImageDataGenerator (for augmentation)

---

## 📈 Training Strategy

- Used `ImageDataGenerator` with horizontal flips and zoom for data augmentation
- Optimizer: **Adam**
- Loss Function: **Binary Crossentropy**
- EarlyStopping to avoid overfitting during fine-tuning
- Combined history of both training phases for better metric visualization

---

## 🔐 Practical Applications

- Fake image detection in news/media
- Deepfake filtering on social platforms
- Security and surveillance (IoT cameras)
- Digital forensics and law enforcement tools

---

## 🚀 How to Run

1. Upload the dataset in the expected directory structure (`/Train`, `/Validation`, `/Test`)
2. Run the `Fake_Image_Detection_Model.ipynb` notebook on Kaggle, Colab, or local Jupyter
3. You’ll get classification metrics and training plots

---

## 📌 Future Improvements

- Deploy using Flask or FastAPI as a web service
- Integrate Grad-CAM for model explainability
- Use additional fake image datasets (e.g., DeepFakeDetection, FFHQ)

---


