import os

import cv2
from enum import Enum

import numpy as np
import pandas as pd
from src.transform import TransformTrain, TransformEval, TransformPred, TransformPred_Cut


class Mode(Enum):
    train = 0
    valid = 1
    predict = 2
    predict_cut = 3


class RainDataset:
    def __init__(
            self, root: str,
            mode: Mode,
            multiscale: bool,
            scale: float = 0.5,
            base_size=(1080, 1920),
            crop_size=(640, 640),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
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

        list_path =  f'{root}/train_val_list.csv'
        list_csv = pd.read_csv(list_path)
        if mode == Mode.train:
            name_list = list(list_csv.loc[list_csv.phase=='train']['image_name'].values)
            self.img_list = [
                (f'{root}/leftImg8bit/{filename}', f'{root}/gtFine/{filename[:-4]}_gtFine_labelTrainIds.png')
                for filename in name_list
            ]
        elif mode == Mode.valid:
            name_list = list(list_csv.loc[list_csv.phase == 'val']['image_name'].values)
            self.img_list = [
                (f'{root}/leftImg8bit/{filename}', f'{root}/gtFine/{filename[:-4]}_gtFine_labelTrainIds.png')
                for filename in name_list
            ]
        elif mode == Mode.predict:
            pass
        elif mode == Mode.predict_cut:
            pass
        else:
            raise ValueError('Mode error')

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
            return TransformEval(self.crop_size, self.mean, self.std)
        elif self.mode == Mode.predict:
            return TransformPred(self.crop_size, self.mean, self.std)
        elif self.mode == Mode.predict_cut:
            return TransformPred_Cut(self.crop_size, self.mean, self.std)
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
            if self.mode == Mode.train:
                image_path, label_path = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                image, label = self.transform(image, label)
                label = np.expand_dims(label, axis=0)
                return image.copy(), label.copy()
            elif self.mode == Mode.valid:
                image_path, label_path, image_name = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                image, label = self.transform(image, label)
                label = np.expand_dims(label, axis=0)
                return image.copy(), label.copy(), image_name
            elif self.mode == Mode.predict:
                image_path, image_name = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, c = image.shape
                # image = self.generate(image)
                image, resize_shape = self.transform(image)
                return image.copy(), resize_shape, (h, w), int(image_name.split('.')[0])
            elif self.mode == Mode.predict_cut:
                image_path, image_name = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, c = image.shape
                image = self.transform(image)
                return image.copy(), (h, w), image_name
        else:
            raise StopIteration

    def __len__(self):
        return self._number


if __name__ == '__main__':
    # pass
    dataset_train_buffer = RainDataset(root=r'D:\Dataset\Mindspore_Rain\segmentation_realRain', mode=Mode.train,
                                     multiscale=False,
                                     base_size=(720, 1280), crop_size=(640, 1024))
    img, mask = dataset_train_buffer[0]
    print(img.shape, mask.shape)

    img = img.transpose([1, 2, 0])
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', img)
    cv2.imshow('mask', mask.transpose([1, 2, 0]))
    cv2.waitKey()

    # dataset_train_buffer = RSDataset(root='../datas/train', mode=Mode.predict, base_size=640)
    # _img, original_shape, _image_name = dataset_train_buffer[0]
    # print(original_shape, _image_name)