import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from load_dataset import ImagesDataset


def augment_image(img_np: np.ndarray, index: int) -> tuple[torch.Tensor, str]:
    # Ensure input is uint8 for PIL-like transforms if needed, or float for Tensor
    # If img_np is float (0-1), these transforms work fine on Tensors.

    transformations = [
        # 1. Blur (Noise robustness)
        transforms.GaussianBlur(kernel_size=5),

        # 2. Rotation (Orientation robustness)
        transforms.RandomRotation(degrees=180),

        # 3. Flips (Corrected to use probability p=0.5 instead of 180)
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),

        # 4. THE CRITICAL FIX: RandomResizedCrop
        # scale=(0.3, 1.0): Learn from parts as small as 30% of the object (e.g., handles)
        transforms.RandomResizedCrop(100, scale=(0.3, 1.0)),

        # 5. Color Jitter (Lighting robustness)
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ]

    v = index % 7
    if v == 0:
        image_torch = torch.from_numpy(img_np)
        return image_torch, "Original"

    elif v <= 5:
        # Apply single transformation
        transform = transformations[v - 1]
        # Ensure we work with Tensor
        img_t = torch.from_numpy(img_np)
        image_torch = transform(img_t)
        return image_torch, transform.__class__.__name__

    else:
        # Apply composite (Mix of 3)
        # We allow crop to be selected here too!
        subset = np.random.choice(transformations, 3, replace=False)
        compose = transforms.Compose(list(subset))
        img_t = torch.from_numpy(img_np)
        image_torch = compose(img_t)
        return image_torch, "Compose"


class TransformedImagesDataset(Dataset):
    def __init__(self, data_set: Dataset):
        self.data_set = data_set

    def __getitem__(self, index: int):
        img_np, class_id, class_name, img_path = self.data_set[index//7]
        image_torch, transform_name = augment_image(img_np, index)
        return image_torch, transform_name, index, class_id, class_name, img_path

    def __len__(self):
        return len(self.data_set)*7


if __name__ == "__main__":
    dataset = ImagesDataset("validated_images", 100, 100, int)
    transformed_ds = TransformedImagesDataset(dataset)
    fig, axes = plt.subplots(2, 4)
    for i in range(0,8):
        trans_img, trans_name, index, classid, classname, img_path = transformed_ds[i]
        _i = i // 4
        _j = i % 4
        axes[_i, _j].imshow(transforms.ToPILImage()(trans_img))
        axes[_i, _j].set_title(f'{trans_name}\n{classname}')

    fig.tight_layout()
    plt.show()
