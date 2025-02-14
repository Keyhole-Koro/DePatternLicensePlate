import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set this to the number of cores you want to use

import cv2
import numpy as np
import glob
import logging
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from utils.feature_extraction import extract_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prepare data (images and labels)
# X: Features, y: Labels (e.g., 0=private, 1=light_private, 2=commerce, ...)
X = []
y = []

# Assuming there are class-specific folders in the 'dataset' folder
labels = {'private': 0, 'light_private': 1, 'commerce': 2, 'light_commerce': 3, 'designed': 4}

logging.info("Starting to process images and extract features.")

for label, idx in labels.items():
    logging.info(f"Processing label: {label}")
    for img_path in glob.glob(f'dataset/{label}/*.*'):
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 50))
            features = extract_features(img)
            X.append(features)
            y.append(idx)

X = np.array(X)
y = np.array(y)

logging.info("Finished processing images and extracting features.")
logging.info(f"Total images processed: {len(X)}")

# Training and evaluation
logging.info("Starting training and evaluation.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
logging.info(f'Accuracy: {accuracy:.2f}')

# Save the trained model
model_path = 'output/knn_model.pkl'
joblib.dump(model, model_path)
logging.info(f'Trained model saved to {model_path}')