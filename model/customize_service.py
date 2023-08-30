import threading
import os

import logging
import numpy as np
import cv2

from mindspore import context, Tensor
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model_service.model_service import SingleNodeService
from PIL import Image

from src.net import FCN8s, pre_process

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class class_service(SingleNodeService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

        logger.info("self.model_name:%s self.model_path: %s", self.model_name,
                    self.model_path)
        # self.cfg = data_config
        self.batch_size = 0
        self.image_label_name = []
        logger.info("11")
        self.net = FCN8s(n_class=21)
        logger.info("12")
        # 非阻塞方式加载模型，防止阻塞超时
        self.load_model()
        # thread.start()

    def load_model(self):
        logger.info("load network ... \n")
        # num_class = 21
        # 初始化模型结构
        # self.net = FCN8s(n_class=21)
        logger.info("11")
        self.net.set_train(False)
        # ckpt_file = './FCN8s.ckpt'
        logger.info(self.model_path)
        logger.info("12")
        ckpt_file = self.model_path + '/FCN8s.ckpt'
        logger.info(ckpt_file)
        logger.info("13")

        param_dict = load_checkpoint(ckpt_file)
        logger.info("14")
        load_param_into_net(self.net, param_dict)
        logger.info("load network successfully ! \n")

    # 每传输一个文件过来，就会调用一次_preprocess->_inference->_postprocess
    def _preprocess(self, input_data):
        preprocessed_result = {}
        images = []
        logger.info("Get image dict!")
        for k, v in input_data.items():
            for file_name, file_content in v.items():
                img_ = np.array(Image.open(file_content), dtype=np.uint8)
                images.append(img_)
                file_name1 = file_name.split('.')[0] + '.png'
                self.image_label_name.append(file_name1)

        self.batch_size = len(images)
        crop_size = 512
        logger.info("batch_size: %s", self.batch_size)

        batch_img = np.zeros((self.batch_size, 3, crop_size, crop_size), dtype=np.float32)

        # 原图的大小
        ori_hw = []
        # 存放缩放图像大小的值
        resize_hw = []
        for l in range(self.batch_size):
            ori_h, ori_w = images[l].shape[0], images[l].shape[1]
            ori_hw.append([ori_h, ori_w])
            img_ = images[l]
            img_, resize_h, resize_w = pre_process(img_, crop_size)
            batch_img[l] = img_
            resize_hw.append([resize_h, resize_w])

        preprocessed_result['images'] = batch_img
        preprocessed_result['ori_hw'] = ori_hw
        preprocessed_result['resize_hw'] = resize_hw

        return preprocessed_result

    def _inference(self, preprocessed_result):
        result_lst = []
        batch_img = np.ascontiguousarray(preprocessed_result['images'])
        # print('Input shape: ', batch_img.shape)
        logger.info("Input shape: %s", batch_img.shape)
        # exit()
        net_out = self.net(Tensor(batch_img, mstype.float32))
        net_out = net_out.asnumpy()
        print(net_out.shape)
        for bs in range(self.batch_size):
            probs_ = net_out[bs][:, :preprocessed_result['resize_hw'][bs][0],
                     :preprocessed_result['resize_hw'][bs][1]].transpose((1, 2, 0))
            ori_h, ori_w = preprocessed_result['ori_hw'][bs][0], preprocessed_result['ori_hw'][bs][1]

            # 改回原图大小，# list: 4, [ndarray:(375，500，21), ndarray:(358，500，21),……]
            probs_ = cv2.resize(probs_.astype(np.float32), (ori_w, ori_h))
            probs_ = probs_.argmax(axis=2)  # list: 4, [ndarray:(375，500), ndarray:(358，500),……]
            probs_ = probs_.tolist()
            result_lst.append(probs_)  # 输出为所有测试图片的结果

        return result_lst

    def _postprocess(self, result_lst):
        result = {}
        result["image_list"] = result_lst
        result["name_list"] = self.image_label_name

        return result