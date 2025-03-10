import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def predict_and_show(model, test_image_path, img_size=(256, 256)):

    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, img_size) / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))

    pred_mask = model.predict(img_resized)[0, ..., 0]

    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    pred_mask_resized = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask_resized, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray')
    plt.imshow(pred_mask_resized, cmap='jet', alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.savefig('predicted_mask.png')
    plt.show()

def predict_and_save_masks(model, test_folder, save_folder, img_size=(256, 256)):
    os.makedirs(save_folder, exist_ok=True)

    test_images = [f for f in os.listdir(test_folder) if f.endswith('.png')]

    for img_name in tqdm(test_images, desc="Processing images"):
        img_path = os.path.join(test_folder, img_name)
        save_path = os.path.join(save_folder, img_name.replace('.png', '_Annotation.png'))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_resized = cv2.resize(img, (img_size[1], img_size[0])) / 255.0
        img_resized = np.expand_dims(img_resized, axis=(0, -1))

        pred_mask = model.predict(img_resized)[0, ..., 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        pred_mask_resized = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))

        cv2.imwrite(save_path, (pred_mask_resized * 255))

    print(f"Saved all masks to {save_folder}")

    def export_stat(img_dir, csv_dir) -> pd.DataFrame:
        
        df = pd.read_csv(csv_dir)
        results = []

        for _, row in df.iterrows():
            filename = row["filename"]
            img_path = os.path.join(img_dir, filename)
            annotation_path = img_path.replace(".png", "_Annotation.png")

            annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

            _, binary = cv2.threshold(annotation, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ellipse = cv2.fitEllipse(contours[0])
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse

            pixel_spacing = df[df['filename'] == filename]['pixel size(mm)'].values
            pixel_spacing = pixel_spacing[0]

            center_x_mm = center_x * pixel_spacing
            center_y_mm = center_y * pixel_spacing
            semi_axes_b_mm = (major_axis / 2) * pixel_spacing
            semi_axes_a_mm = (minor_axis / 2) * pixel_spacing
            if angle > 90:
                angle_rad = np.deg2rad(angle - 90)
            else:
                angle_rad = np.deg2rad(90 - angle)

            results.append({
                "filename": filename,
                "center_x_mm": center_x_mm,
                "center_y_mm": center_y_mm,
                "semi_axes_a_mm": semi_axes_a_mm,
                "semi_axes_b_mm": semi_axes_b_mm,
                "angle_rad": angle_rad
            })

        df_results = pd.DataFrame(results)
        return df_results