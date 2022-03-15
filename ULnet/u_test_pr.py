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

from sklearn.metrics import precision_recall_curve
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import U2NETP

import time

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir,g_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    #imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    a = Image.fromarray(predict_np * 255).convert('L')
    #a = Image.fromarray(predict_np*255).convert('RGB')
    #a.show()
    imo = a.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)
    #print(pb_np.shape)
    y_score = np.reshape(pb_np, newshape=[-1])
    y_score = y_score/255
    y_score = np.around(y_score, decimals=1)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    imo.save(d_dir+imidx+'.png')

    ##真实值
    img = Image.open(g_dir+imidx+'.png')  # 打开图像并转化为数字矩阵
    img = np.array(img)
    y_true = np.reshape(img, newshape=[-1])

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    #print("precision = ", precision)
    #print("recall = ", recall)
    #print("thresholds = ", thresholds)
    return precision, recall






def main():

    # --------- 1. get image path and name ---------
    model_name='unet4'  #unet4'



    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    groundtruth_dir = os.path.join(os.getcwd(), 'test_data', 'Groundtruth'+ os.sep)

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
    total_time = 0
    all_precision=np.zeros(shape=[12, ])
    all_recall=np.zeros(shape=[12, ])
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        #print(inputs_test.size())
        start = time.clock()
        d0,d1,d2,d3,d4= net(inputs_test)                # ,d4 #,d4,d5,d5,d6,d7
        end = time.clock()
        running_time = (end - start)*1000
        total_time += running_time
        print('Running time: %f ms Seconds' % running_time)

        # normalization
        pred = d0[:,0,:,:]
        pred = normPRED(pred)
        #print(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        pre, rec = save_output(img_name_list[i_test], pred, prediction_dir, groundtruth_dir)
        all_precision += pre
        all_recall += rec
        del d0,d1,d2,d3,d4       #d4 ,d5,d5,d6
    print('Total time: %f ms Seconds' % total_time)
    all_precision = all_precision/20
    all_recall = all_recall/20
    #print('all_precision = ', ",".join(str(i) for i in all_precision))
    #print('all_recall = ', ",".join(str(i) for i in all_recall))

if __name__ == "__main__":
    main()

