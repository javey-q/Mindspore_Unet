import os

import cv2
from enum import Enum

import numpy as np

from src.transform import TransformTrain, TransformEval, TransformPred


class Mode(Enum):
    train = 0
    valid = 1
    predict = 2


class RSDataset:
    def __init__(
            self, root: str,
            mode: Mode,
            multiscale: bool,
            scale: float = 0.5,
            base_size=640,
            crop_size=(512, 512),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
    ):
        self._index = 0
        self.root = root
        self.mode = mode
        self.multiscale = multiscale
        self.scale = scale
        self.base_size = base_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

        self.list_path = None
        if mode == Mode.train:
            self.list_path = f'{root}/train/train_segmentation.txt'
        elif mode == Mode.valid:
            self.list_path = f'{root}/valid/valid_segmentation.txt'
        elif mode == Mode.predict:
            self.list_path = f'{root}/test_list.txt'
        else:
            raise ValueError('Mode error')

        with open(self.list_path, mode='r') as file:
            img_list = [line.strip() for line in file]

        if mode == Mode.train:
            self.img_list = [
                (f'{root}/train/images/{filename}', f'{root}/train/masks/{filename}')
                for filename in img_list
            ]
        elif mode == Mode.valid:
            self.img_list = [
                (f'{root}/valid/images/{filename}', f'{root}/valid/masks/{filename}')
                for filename in img_list
            ]
        elif mode == Mode.predict:
            self.img_list = [
                (f'{root}/images/{filename}', filename)
                for filename in img_list
            ]

        self.transform = self.get_transform()

        self._number = len(self.img_list)

    def get_transform(self):
        if self.mode == Mode.train:
            return TransformTrain(
                base_size=self.base_size, crop_size=self.crop_size,
                multi_scale=self.multiscale, scale=self.scale, ignore_label=0,
                mean=self.mean, std=self.std
            )
        elif self.mode == Mode.valid:
            return TransformEval(self.crop_size[0], self.mean, self.std)
        elif self.mode == Mode.predict:
            return TransformPred(self.crop_size[0], self.mean, self.std)
        else:
            return None

    def input_transform(self, image: np.ndarray):
        image = cv2.resize(image, self.crop_size)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image.astype(np.float32)

    def label_transform(self, label):
        label = cv2.resize(label, self.crop_size, interpolation=cv2.INTER_NEAREST)
        label = label / 255.0
        return label.astype(np.float32)

    def generate(self, image, mask=None):
        image = self.input_transform(image)
        image = image.transpose([2, 0, 1])
        if mask is not None:
            mask = self.label_transform(mask)
            return image, mask
        return image

    def __getitem__(self, item):
        # print(item)
        if item < self._number:
            if self.mode != Mode.predict:
                image_path, label_path = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                # image, label = self.generate(image, label)
                image, label = self.transform(image, label)
                label = np.expand_dims(label, axis=0)
                return image.copy(), label.copy()
            else:
                image_path, image_name = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, c = image.shape
                # image = self.generate(image)
                image, resize_shape = self.transform(image)
                return image.copy(), resize_shape, (np.int32(h), np.int32(w)), np.int32(image_name.split('.')[0])
        else:
            raise StopIteration

    def __len__(self):
        return self._number


if __name__ == '__main__':
    pass
    # dataset_train_buffer = RSDataset(root='../datas', mode=Mode.train,
    #                                  multiscale=True, scale=0.5,
    #                                  base_size=640, crop_size=(512, 512))
    # img, mask = dataset_train_buffer[0]
    # print(img.shape, mask.shape)

    # dataset_train_buffer = RSDataset(root='../datas/train', mode=Mode.predict, base_size=640)
    #
    # _img, original_shape, _image_name = dataset_train_buffer[0]
    # print(original_shape, _image_name)
    # cv2.imshow('img', _img.transpose([1, 2, 0]))
    # cv2.waitKey()
