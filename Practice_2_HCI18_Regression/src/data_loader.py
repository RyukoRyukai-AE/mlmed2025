import glob, os, cv2
import numpy as np

class HC18Dataset:
    def __init__(self, img_dir, mask_dir, img_size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = tuple(img_size)
        self.image_paths, self.mask_paths = self._load_image_mask_paths()

    def _load_image_mask_paths(self):
        image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*HC.png")))
        mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, "*.png")))
        return image_paths, mask_paths

    def load_data(self):
        images, masks = [], []
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (self.img_size[1], self.img_size[0])) / 255.0
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0])) / 255.0

            images.append(img[..., np.newaxis])
            masks.append(mask[..., np.newaxis])

        return np.array(images), np.array(masks)
