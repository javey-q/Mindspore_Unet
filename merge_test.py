import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm

def create_zeros_png(image_w,image_h):
    '''Description:
        0. 先创造一个空白图像，将滑窗预测结果逐步填充至空白图像中；
        1. 填充右下边界，使原图大小可以杯滑动窗口整除；
        2. 膨胀预测：预测时，对每个(1024,1024)窗口，每次只保留中心(512,512)区域预测结果，每次滑窗步长为512，使预测结果不交叠；
    '''
    target_w = target_h = target_size
    new_w = (image_w // stride) * stride if (image_w % stride == 0) else (image_w // stride + 1) * stride
    new_h = (image_h // stride) * stride if (image_h % stride == 0) else (image_h // stride + 1) * stride

    zeros = (border+new_h,border+new_w)  #填充空白边界
    zeros = np.zeros(zeros,np.uint8)
    return zeros

images_root = '../datas/testA_new/images/'
slice_dir = "../dataset_1920/test/pred/"
dir_pred = './pred'
stride = 1280
border = 640
target_size = 1920
images_sets = [i for i in os.listdir(images_root) if i.endswith(".tif")]

stride_2 = border//2

for i in tqdm(range(len(images_sets))):
    original_img = cv2.imread(images_root + images_sets[i], cv2.IMREAD_GRAYSCALE)
    original_height, original_width = original_img.shape
    predict_png = create_zeros_png(original_width, original_height)

    csv_file = pd.read_csv("../dataset_1920/test/pos_test.csv")
    slices_pos = csv_file[csv_file.image_name==images_sets[i]].reset_index(drop=True)
    for idx in range(len(slices_pos)):
        slice_name = str(slices_pos.iloc[idx, 1])
        pos_list = slices_pos.iloc[idx, 2:].values.astype("int")
        predict = cv2.imread(slice_dir+slice_name, cv2.IMREAD_GRAYSCALE)

        [topleft_x, topleft_y, buttomright_x, buttomright_y] = pos_list

        if (buttomright_x - topleft_x) == target_size and (buttomright_y - topleft_y) == target_size:
            # 每次预测只保留图像中心(512,512)区域预测结果
            predict_png[topleft_y + stride_2:buttomright_y - stride_2, topleft_x + stride_2:buttomright_x - stride_2] \
                                                                                         = predict[
                                                                                            stride_2:stride_2+stride,
                                                                                            stride_2:stride_2+stride]
        else:
            raise ValueError(
                "target_size!=1920， Got {},{}".format(buttomright_x - topleft_x, buttomright_y - topleft_y))
    h, w = predict_png.shape
    predict_png = predict_png[stride_2:h - stride_2, stride_2:w - stride_2]  # 去除整体外边界
    predict_png = predict_png[:original_height, :original_width]  # 去除补全512整数倍时的右下边界
    mask_name = f'{images_sets[i].split(".")[0]}.png'
    cv2.imwrite(f'{dir_pred}/{mask_name}', predict_png)
        