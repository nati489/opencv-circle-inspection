# OpenCV Circle Inspection System

This project is a small computer vision pipeline built using **Python and OpenCV** to simulate a basic **visual inspection task**, similar to what is used in industrial automation.

The system analyzes images containing circular objects and classifies them as **PASS** or **FAIL** based on how closely the detected shape resembles a circle.

---

## What the Project Does

- Converts images to binary masks and extracts contours
- Automatically selects the most likely “part” in the image
- Measures **circularity** to evaluate shape quality
- Classifies each image as **PASS** or **FAIL**
- Evaluates performance using accuracy, precision, and recall

This project focuses on **classical computer vision techniques** rather than machine learning.

---

## Why This Project

Many real-world inspection systems rely on:
- shape analysis
- contour filtering
- rule-based decision logic

This project demonstrates how those ideas work in practice using OpenCV.

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Basic image processing and geometry
- Evaluation metrics (accuracy, precision, recall)

---

## How to Run

### 1. Install dependencies
pip install opencv-python numpy
###2. Run using the code below
- python evaluate.py
