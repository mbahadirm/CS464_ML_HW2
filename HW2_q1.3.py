import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# ==== 1. Load the dataset and apply PCA via SVD ====
def load_images_as_matrix(image_dir, image_size=(60, 60)):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    X = np.zeros((len(image_files), image_size[0] * image_size[1]))
    for i, fname in enumerate(image_files):
        img = Image.open(os.path.join(image_dir, fname)).convert('L')
        X[i, :] = np.array(img).flatten()
    return X

def apply_pca_svd(X):
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    return X_mean, VT  # VT = W, rows are principal components

# ==== 2. Load and project source & target images ====
def load_flat_image(path):
    return np.array(Image.open(path).convert('L')).flatten()

def project_to_pca_space(x, X_mean, W):
    return (x - X_mean) @ W.T

def reconstruct_from_pca_space(f, W, X_mean):
    return f @ W + X_mean

# ==== 3. Morphing ====
def morph_faces(source_path, target_path, W, X_mean, t_values):
    x_src = load_flat_image(source_path)
    x_tgt = load_flat_image(target_path)

    f_src = project_to_pca_space(x_src, X_mean, W)
    f_tgt = project_to_pca_space(x_tgt, X_mean, W)

    morphed_images = []
    for t in t_values:
        f_morphed = (1 - t) * f_src + t * f_tgt
        x_morphed = reconstruct_from_pca_space(f_morphed, W, X_mean)
        morphed_images.append(x_morphed.reshape(60, 60))

    return morphed_images

# ==== 4. Run Everything ====
if __name__ == "__main__":
    # Update these paths according to your local setup:
    dataset_dir = r"C:\Users\mbmut\OneDrive\Masaüstü\HW2\q1_dataset\cat_dog_images"
    source_path = r"C:\Users\mbmut\OneDrive\Masaüstü\HW2\q1_dataset\face_morphing\source.png"
    target_path = r"C:\Users\mbmut\OneDrive\Masaüstü\HW2\q1_dataset\face_morphing\target.png"

    X = load_images_as_matrix(dataset_dir)
    X_mean, W = apply_pca_svd(X)

    t_values = np.linspace(0, 1, 11)
    morphed_imgs = morph_faces(source_path, target_path, W, X_mean, t_values)

    # ==== 5. Display results ====
    fig, axes = plt.subplots(1, len(t_values), figsize=(20, 2))
    for i, ax in enumerate(axes):
        ax.imshow(morphed_imgs[i], cmap='gray')
        ax.set_title(f"t={t_values[i]:.1f}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
