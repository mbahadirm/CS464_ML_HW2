import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# 1. Load and flatten all images into a data matrix X (4000 x 3600)
def load_images_as_matrix(image_dir, image_size=(60, 60)):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    num_images = len(image_files)
    X = np.zeros((num_images, image_size[0] * image_size[1]))

    for i, file_name in tqdm(enumerate(image_files), total=num_images):
        image_path = os.path.join(image_dir, file_name)
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        X[i, :] = np.array(img).flatten()
    
    return X

# 2. Apply PCA using SVD
def pca_with_svd(X, num_components=10):
    # Mean center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Perform SVD
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

    # Compute PVE (Proportion of Variance Explained)
    singular_values_squared = S**2
    total_variance = np.sum(singular_values_squared)
    PVE = singular_values_squared[:num_components] / total_variance

    return PVE, VT[:num_components], X_mean

# 3. Run it
image_dir = "HW2/q1_dataset/cat_dog_images"  # replace with your local path if needed
X = load_images_as_matrix(image_dir)
PVE, top_components, X_mean = pca_with_svd(X)

# 4. Print PVE values
for i, pve in enumerate(PVE, 1):
    print(f"Principal Component {i}: {pve:.4f} ({pve * 100:.2f}% variance explained)")

import matplotlib.pyplot as plt

def visualize_principal_components(components, image_shape=(60, 60)):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("First 10 Principal Components (Eigenfaces)", fontsize=16)

    for i in range(10):
        ax = axes[i // 5, i % 5]
        pc_image = components[i].reshape(image_shape)
        ax.imshow(pc_image, cmap='gray')
        ax.set_title(f'PC {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to display the images
visualize_principal_components(top_components)

