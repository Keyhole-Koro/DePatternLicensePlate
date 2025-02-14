import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_dominant_colors(image_path, k=3):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    reshaped = image.reshape((-1, 3)).astype(np.float32)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(reshaped)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Calculate the proportion of each color cluster
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = h * w
    color_ratios = {i: count / total_pixels for i, count in zip(unique, counts)}

    # Sort clusters by size
    sorted_clusters = sorted(color_ratios.items(), key=lambda x: x[1], reverse=True)
    primary_cluster_idx = sorted_clusters[0][0]
    secondary_cluster_idx = sorted_clusters[1][0]

    primary_color = centers[primary_cluster_idx]
    secondary_color = centers[secondary_cluster_idx]

    return primary_color, secondary_color

def is_color_pair(primary, secondary, color_pairs):
    for pair in color_pairs:
        if np.allclose(primary, pair[0], atol=50) and np.allclose(secondary, pair[1], atol=50):
            return True
    return False

if __name__ == "__main__":
    input_image = "image/image4.jpg"  # Image to process

    # Define color pairs (in BGR format)
    color_pairs = [
        (np.array([0, 255, 255]), np.array([0, 0, 0])),  # Yellow, Black
        (np.array([0, 0, 0]), np.array([0, 255, 255])),  # Black, Yellow
        (np.array([0, 100, 0]), np.array([255, 255, 255])),  # Dark Green, White
        (np.array([255, 255, 255]), np.array([0, 255, 0]))  # White, Green
    ]

    primary_color, secondary_color = get_dominant_colors(input_image, k=3)
    print("Primary Color:", primary_color)
    print("Secondary Color:", secondary_color)

    if is_color_pair(primary_color, secondary_color, color_pairs):
        print("The primary and secondary colors match one of the specified pairs.")
    else:
        print("The primary and secondary colors do not match any of the specified pairs.")