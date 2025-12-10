from glob import glob
from os import path
from typing import Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from gray_scale_conversion import to_grayscale
from prepare_images import prepare_image


class ImagesDataset(Dataset):

    def __init__(self, image_dir=None, width=100, height=100, dtype=None,
                 external_image: Optional[Image.Image] = None):

        self.external = external_image
        self.dtype = dtype
        self.width = width
        self.height = height

        if external_image is not None:
            self.image_filepaths = []
            self.filenames_classnames = None
            self.classnames_to_ids = None
            return

        # Normal training mode
        self.image_filepaths = sorted(path.abspath(f) for f in glob(path.join(image_dir, "*.jpg")))
        class_filepath = [path.abspath(f) for f in glob(path.join(image_dir, "*.csv"))][0]
        self.filenames_classnames, self.classnames_to_ids = ImagesDataset.load_classnames(class_filepath)

    @staticmethod
    def load_classnames(class_filepath: str):
        filenames_classnames = np.genfromtxt(class_filepath, delimiter=';', skip_header=1, dtype=str)
        classnames = np.unique(filenames_classnames[:, 1])
        classnames.sort()

        classnames_to_ids = {name: i for i, name in enumerate(classnames)}
        return filenames_classnames, classnames_to_ids

    def __getitem__(self, index):

        # Inference mode
        if self.external is not None:
            img = np.array(self.external, dtype=self.dtype)
            img = to_grayscale(img)
            resized, _ = prepare_image(img, self.width, self.height, 0, 0, 32)
            return resized, 0, "unknown", "external_input"

        # Training mode
        with Image.open(self.image_filepaths[index]) as im:
            img = np.array(im, dtype=self.dtype)

        img = to_grayscale(img)
        resized, _ = prepare_image(img, self.width, self.height, 0, 0, 32)

        classname = self.filenames_classnames[index][1]
        classid = self.classnames_to_ids[classname]

        return resized, classid, classname, self.image_filepaths[index]

    def __len__(self):
        if self.external is not None:
            return 1
        return len(self.image_filepaths)



if __name__ == "__main__":
    dataset = ImagesDataset("validation_images", 100, 100, int)
    for resized_image, classid, classname, _ in dataset:
        print(f'image shape: {resized_image.shape}, dtype: {resized_image.dtype}, '
              f'classid: {classid}, classname: {classname}\n')
