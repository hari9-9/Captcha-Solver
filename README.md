# CAPTCHA Solver: A Scalable and Efficient Approach

## Overview
This project focuses on building a scalable CAPTCHA solving system capable of identifying text-based CAPTCHAs with varying character lengths (1 to 6) and diverse fonts. Using a modular approach, we implemented and trained seven deep learning models: one for predicting CAPTCHA length and six specialized for CAPTCHA classification based on length.

The project emphasizes scalability by leveraging TensorFlow Lite (TFLite) for efficient deployment on resource-constrained devices like Raspberry Pi, ensuring fast and accurate CAPTCHA classification.

---

## Features
- **Scalable Design:** Utilizes a modular seven-model architecture to handle variable CAPTCHA lengths efficiently.
- **Font Robustness:** Models trained on datasets containing diverse fonts to improve generalization.
- **Efficient Deployment:** Conversion of TensorFlow models to TFLite format ensures compatibility with edge devices like Raspberry Pi.
- **High Performance:** Employs preprocessing and optimized training configurations to achieve high accuracy and efficient resource utilization.

---

## Architecture
The system is based on a **divide-and-conquer** approach, where:
1. **Length Prediction Model:** Predicts the length of the CAPTCHA.
2. **Classification Models:** Six specialized models classify CAPTCHAs for lengths 1 through 6.

### Key Components
- **Data Preprocessing:** Explored various preprocessing techniques, with the best configuration enhancing image clarity and model performance.
- **Training Models:** Each model trained on datasets specific to its target length for optimized performance.
- **Font Handling:** Included images of all font variations in training datasets to ensure robustness.

### Training Configuration
- **Epochs:** 20
- **Batch Size:** 64
- **Early Stopping:** Enabled to prevent overfitting.

### Dataset
- Generated 64,000 images for training each model (90/10 train-test split).
- Data preprocessing included thresholding and noise removal to enhance input quality.

---

## TensorFlow Lite Conversion
To ensure scalability and compatibility with resource-constrained environments:
1. **Model Conversion:** TensorFlow models were converted to TFLite format.
2. **Quantization (Optional):** Experimented with quantized models for faster inference, though with reduced accuracy.

The TFLite models were deployed on a Raspberry Pi, achieving efficient classification within acceptable time limits.

---

## Scalability
- **Local vs. Edge Computing:** Training was performed on local machines with GPU acceleration, while inference was executed on Raspberry Pi using TFLite models.
- **Performance Metrics:** The system demonstrated high computational efficiency on edge devices:
  - **Classification Time:** ~900 seconds for 4,000 images.
  - **Resource Utilization:** Optimal use of CPU cores and memory during inference.

---

## Results
- **Score:** 2249/4000
- **Deployment Metrics:**
  - Classification time on Raspberry Pi: ~900 seconds.
  - Efficient model loading and prediction pipeline reduced resource usage and processing time.

---

## Future Work
- **Optimized Multithreading:** To enhance classification speed on Raspberry Pi.
- **Advanced Preprocessing:** Explore image segmentation and additional preprocessing techniques.
- **Quantization-aware Training:** Improve TFLite model performance without significant accuracy loss.



---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/hari9-9/Captcha-Solver.git
   cd Captcha-Solver
   pip install -r requirements.txt
   python convert_to_tflite.py


