import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from load_dataset import ImagesDataset

def augment_image(img_np: np.ndarray, index: int) -> (torch.Tensor, str):
    transformations = [
        transforms.GaussianBlur(5),
        transforms.RandomRotation(180),
        transforms.RandomVerticalFlip(180),
        transforms.RandomHorizontalFlip(180),
        transforms.RandomResizedCrop(100),
        transforms.ColorJitter(brightness=0.445, contrast=0.445, saturation=0.445, hue=0.445),

    ]

    v =index % 7
    if v ==0:
        image_torch = torch.from_numpy(img_np)
        return image_torch,"Original"
    elif v <=5:
        transform=transformations[v - 1]
        image_torch=transform(torch.from_numpy(img_np))
        return image_torch,transform.__class__.__name__
    else:
        transform1,transform2,transform3 = np.random.choice(transformations, 3, replace=False)
        compose = transforms.Compose([transform1, transform2, transform3])
        image_torch = compose(torch.from_numpy(img_np))
        return image_torch,"Compose"


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
