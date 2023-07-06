import torch
import os
from PIL import Image
import numpy as np
import albumentations as A
from torchvision.datasets import VisionDataset


class YOLODataset(VisionDataset):
    def __init__(self, root, transform=None, train=None, valid=None, grid=7, b=2, c=1):
        super().__init__(root, transform=transform)
        # Load the list of image paths and annotation paths
        self.root = root
        self.train = train
        self.valid = valid
        self.annotations = self._load_annotations()
        self.images = self._load_images()
        self.transform = transform
        self.grid = grid
        self.b = b
        self.c = c

    def _load_images(self):

        # Implement the logic to load the list of image paths
        # Return a list of image paths
        imagepath = []
        for annot in self.annotations:
            filename = f'{annot.split(".")[0]}.jpg'
            imagepath.append(filename)

        return imagepath

    def _load_annotations(self):
        # Implement the logic to load the list of annotation paths
        # Return a list of annotation paths
        if self.train:
            fullpath = os.path.join(self.root, 'data', 'labels', 'train')
        elif self.valid:
            fullpath = os.path.join(self.root, 'data', 'labels', 'valid')
        return [labels for labels in os.listdir(fullpath)]

    def __getitem__(self, index):

        # Implement the logic to parse the annotation file and convert it to bounding box format
        # Process the annotation and create a tensor representation (e.g., bounding box coordinates, class labels)
        if self.train:
            image_path = os.path.join(self.root, 'data', 'images', 'train', self.images[index])
        elif self.valid:
            image_path = os.path.join(self.root, 'data', 'images', 'valid', self.images[index])
        annotation_path = self.annotations[index]
        with Image.open(image_path) as f:
            #             image = np.array(f)
            image = np.array(f.convert("RGB"))

        bbox = self._process_annotation(annotation_path)
        #         print(bbox)

        sample = {}
        sample['image'] = image
        sample['bboxes'] = bbox

        if self.transform:
            sample = self.transform(image=image, bboxes=bbox)

        image = sample['image']
        bbox = sample['bboxes']

        image = torch.from_numpy(sample['image']) / 255

        full_label = torch.zeros(self.grid, self.grid, (self.b * 5) + self.c)
        tmp = []

        for box in bbox:

            cls = 1.0
            box = np.array(box)[:4]

            cell_col, cell_row = int(box[0] * self.grid), int(box[1] * self.grid)

            x_cell = box[0] * self.grid - cell_col
            y_cell = box[1] * self.grid - cell_row

            width_cell, height_cell = box[2] * self.grid, box[3] * self.grid

            if full_label[cell_col, cell_row, 1] == 0:
                # Set that there exists an object
                full_label[cell_col, cell_row, 1] = 1

                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                full_label[cell_col, cell_row, 2:6] = box_coordinates

                # Set one hot encoding for class_label
                full_label[cell_col, cell_row, 0] = cls

        return image, full_label

    def __len__(self):
        return len(self.annotations)

    def _process_annotation(self, annotation_path):

        # Implement the logic to process the annotation file and convert it to bounding box format
        # Return the processed annotation as a tensor representation (e.g., bounding box coordinates, class labels)
        if self.train:
            annotation_path = os.path.join(self.root, 'data', 'labels', 'train', annotation_path)
        elif self.valid:
            annotation_path = os.path.join(self.root, 'data', 'labels', 'valid', annotation_path)
        y = np.loadtxt(annotation_path, delimiter=' ')
        if y.ndim > 1:
            y = np.roll(y, -1, axis=1)
            cls = y[:, 0]
            bbox = y[:, 1:]

        else:
            y = np.expand_dims(y, axis=0)
            y = np.roll(y, -1, axis=1)

        return y
