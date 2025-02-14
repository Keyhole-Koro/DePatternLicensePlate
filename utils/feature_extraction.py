import cv2
import numpy as np

# Calculate HSV histogram
def get_hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

# Calculate color moments
def get_color_moments(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean = np.mean(hsv, axis=(0, 1))
    std = np.std(hsv, axis=(0, 1))
    skew = np.mean((hsv - mean) ** 3, axis=(0, 1))
    return np.concatenate([mean, std, skew])

# Feature extraction
def extract_features(img):
    hist = get_hsv_histogram(img)
    moments = get_color_moments(img)
    return np.concatenate([hist, moments])