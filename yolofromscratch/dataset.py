import torch
import os
from PIL import Image
import numpy as np
import albumentations as A
from torchvision.datasets import VisionDataset


from albumentations.pytorch import ToTensorV2


class YOLODataset(VisionDataset):
    def __init__(self, root, transform=None, grid=7, b=2, c=1):
        super().__init__(root, transform=transform)
        # Load the list of image paths and annotation paths
        self.root = root
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
        fullpath = os.path.join(self.root, 'data', 'labels', 'train')
        return [labels for labels in os.listdir(fullpath)]

    def __getitem__(self, index):

        # Implement the logic to parse the annotation file and convert it to bounding box format
        # Process the annotation and create a tensor representation (e.g., bounding box coordinates, class labels)
        image_path = os.path.join(self.root, 'data', 'images', 'train', self.images[index])
        annotation_path = self.annotations[index]
        with Image.open(image_path) as f:
            #             image = np.array(f)
            image = np.array(f.convert("RGB"))

        bbox = self._process_annotation(annotation_path)

        sample = {}
        sample['image'] = image
        sample['bboxes'] = bbox
        #         sample['class_labels'] = cls

        # bbox = torch.from_numpy(bbox).to(torch.float32)
        # cls = torch.from_numpy(cls).to(torch.float32)

        if self.transform:
            sample = self.transform(image=image, bboxes=bbox,
                                    #                                     class_labels=sample['class_labels'] ,
                                    )

        image = sample['image']
        bbox = sample['bboxes']

        #         print(bbox)

        image = torch.from_numpy(sample['image']) / 255

        full_label = torch.zeros(self.grid, self.grid, (self.b * 5) + self.c)
        tmp = []

        for box in bbox:

            cls = np.array(box)[-1]
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

        annotation_path = os.path.join(self.root, 'data', 'labels', 'train', annotation_path)
        y = np.loadtxt(annotation_path, delimiter=' ')
        if y.ndim > 1:
            y = np.roll(y, -1, axis=1)
            cls = y[:, 0]
            bbox = y[:, 1:]

        else:
            y = np.expand_dims(y, axis=0)
            y = np.roll(y, -1, axis=1)

        #             cls = np.expand_dims(y[0], axis=0)

        #             bbox = np.expand_dims(y[1:], axis=0)

        return y


def main():
    ROOT = os.path.join(os.getcwd(), 'Project')
    transform = A.Compose([
        A.Resize(448, 448),
        A.ShiftScaleRotate(rotate_limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.2),
        ToTensorV2(),
    ],
        bbox_params=A.BboxParams(format='yolo',
                                 min_visibility=0.3,
                                 label_fields=[],
                                 ),
    )

    dataset = YOLODataset(root=ROOT,
                          transform=transform,
                          train=True,
                          )

    print(dataset[0][0].dtype)



if __name__ == '__main__':
    main()