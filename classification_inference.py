import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set this to the number of cores you want to use

import cv2
import numpy as np
import joblib
from utils.feature_extraction import extract_features

if __name__ == "__main__":
    input_image = "dataset/light_commerce/cropped_1.jpg"  # Image to classify

    # Load the trained model
    model = joblib.load('output/knn_model.pkl')

    # Read and preprocess the image
    img = cv2.imread(input_image)
    img = cv2.resize(img, (100, 50))
    features = extract_features(img).reshape(1, -1)

    # Predict the class
    prediction = model.predict(features)
    labels = {0: 'private', 1: 'light_private', 2: 'commerce', 3: 'light_commerce', 4: 'designed'}
    predicted_label = labels[prediction[0]]

    print(f'The predicted class for the input image is: {predicted_label}')