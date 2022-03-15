import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model.unet4 import UNET4
from model.unet3 import UNET3
from model.unet5 import UNET5
from model.unet4out import UNET4OUT

import cv2
import time
import warnings
warnings.filterwarnings("ignore")


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    a=Image.fromarray(predict_np*255).convert('L')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = a.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    img = np.array(imo)

    img_raw = cv2.imread(image_name, cv2.IMREAD_COLOR)

    # hsv色彩空间变换
    img_hsv = img_raw[:, :, [2, 1, 0]]
    hsv = cv2.cvtColor(img_hsv, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    # 设置亮度的阈值
    h_thed1 = 70  # Green

    ret, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    # 边界提取，contours包含边界值的坐标
    image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    num = 0
    step1_res = []

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w < 10 or h < 10:
            continue
        num += 1
        #print(num)
        # cv2.rectangle(img_raw, (x,y), (x+w,y+h), (153,153,0), 5)
        newimage = img_raw[y - 10:y + h + 10, x - 10:x + w + 10]  # 先用y确定高，再用x确定宽
        #在局部检测圆
        gray1 = cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY)  # 灰度处理
        if gray1 is None: continue
        gaussian = cv2.GaussianBlur(gray1, (3, 3), 0)  # 高斯模糊3*3
        circles1 = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=20, minRadius=5, maxRadius=100)  # 霍夫圆变换
        if circles1 is None or circles1.all()==0:
            continue
        circles = circles1[0, :, :]  # 降维 2*3
        circles = np.uint16(np.around(circles))  # 取整数

        light = circles[0]
        light = light + [x - 10, y - 10, 0]
        # img = img.copy()
        cv2.circle(img_raw, (light[0], light[1]), light[2], (255, 0, 255), 3)  # 标出圆
        #cv2.circle(img_raw, (light[0], light[1]), 2, (255, 0, 255), 2)  # 标出圆心
        step1_res.append(light)

        # 计算均值
        t = int(light[2] / 1.414)
        xmin = light[0] - t
        ymin = light[1] - t
        xmax = light[0] + t
        ymax = light[1] + t
        #if ymax > 544: ymax = 544
        h_total = 0
        h_list = []

        for x_num in range(xmin, xmax):
            for y_num in range(ymin, ymax):

                h_normal = hue[y_num, x_num]
                h_total += h_normal
                h_list.append(h_normal)
        h_mean = h_total / (4 * t * t)
        #print("h_mean is %f" %h_mean)
        color = str(int(h_mean))

        # 判断色调
        '''
        if h_mean <= 40:
            cv2.putText(img_raw, 'Red', (xmin + 16, ymax + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0,0),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(img_raw, 'Red', (xmin + 16, ymax + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                        cv2.LINE_AA)
        '''

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    #np.savetxt('Calib/' + imidx + '.txt', step1_res, fmt='%d')
    #imo.save(d_dir+imidx+'.png')
    cv2.imwrite("Calib_Pic/"+ imidx +".jpg", img_raw)
    print('num of light is', len(step1_res))


def main():
    # --------- 1. get image path and name ---------
    model_name='unet4'#u2netp

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='unet4'):
        print("...load unet4--- MB")
        net = UNET4(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        start = time.clock()
        d0,d1,d2,d3,d4 = net(inputs_test)                 #,d5,d6,d7

        # normalization
        pred = d0[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)

        save_output(img_name_list[i_test],pred,prediction_dir)
        end = time.clock()
        running_time = (end - start) * 1000
        print('Running time: %f ms Seconds' % running_time)
        print('-------------------------------------------')
        del d0,d1,d2,d3,d4    #,d5,d6,d7



    #total_time += running_time

    # print(total_time)


if __name__ == "__main__":
    main()
