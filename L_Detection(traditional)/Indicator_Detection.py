import cv2
import numpy as np



class Indicator_Detection:
    def __init__(self):
        pass
    def detect(self, img, number_of_light):

        #e1 = cv2.getTickCount()  #计算运行时间的代码

        # 获取图像高度和宽度
        height = img.shape[0]
        width = img.shape[1]
        print("image height= %d"%height)
        print("image width= %d"%width)

        # 用霍夫变换检测指示灯
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                #灰度处理
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)                #高斯模糊3*3
        circles1 = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=1, maxRadius=80)   #霍夫圆变换
        #print(np.shape(circles1))                #三维数组 1*n*3
        circles = circles1[0, :, :]              #降维 n*3
        circles = np.uint16(np.around(circles))  #取整数
        number_of_circles = len(circles)         #检测到的圆圈个数

        # 没有检测到足够的指示灯
        if number_of_circles < number_of_light:
            number_of_light = number_of_circles
            print("只检测到{}个指示灯".format(number_of_light))
        # 准备输出指示灯检测结果的编号和状态
        print("灯的数量： %d"%number_of_light)

        linght_NO = 0  # 指示灯编号
        state = np.zeros((12, 1))                #指示灯状态 0为灭 1为亮
        gray_value = 0.4                         #设置灰度的阈值

        #定位检测到的指示灯
        for i in circles[:number_of_light]:
            cv2.rectangle(img, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 3)     #标出方形
            #计算均值
            t = int(i[2]/1.414)
            if t==0:
                break
            xmin = i[0]-t
            ymin = i[1]-t
            xmax = i[0]+t
            ymax = i[1]+t
            gray_total = 0
            #print(ymin)
            #print(ymax)
            ## 求取指示灯内一定区域的平均亮度值
            for x_num in range(xmin, xmax):
                for y_num in range(ymin, ymax):
                    gray_normal = gray[y_num, x_num]/ 255
                    #gray_normal = 0
                    gray_total += gray_normal
                    #print(gray_total)
            gray_mean = gray_total/(4*t*t)      #某个指示灯的平均亮度值
            #print("gray_mean is %f"%gray_mean)

            gray_res = gray_mean - gray_value   #与灰度阈值进行比较
            ##   判断状态，并将结果打在屏幕上
            linght_NO += 1
            if gray_res > 0:
                #word = 'Light {} : Location is ({},{}), State: ON'.format(word_y, i[0], i[1])
                state[linght_NO-1] = 1     # 亮灯为1
                cv2.putText(img, str((i[0], i[1])), (i[0]-20, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(img, 'ON', (i[0], ymax+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                #cv2.putText(img, str(linght_NO), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (202, 255, 112 ), 1, cv2.LINE_AA)
                #print('{}号指示灯的圆心位置为（{},{}）的指示灯状态为：亮灯'.format(linght_NO, i[0], i[1]))

            else:
                #word = 'Light {} : Location is ({},{}), State: OFF'.format(word_y, i[0], i[1])
                state[linght_NO-1] = 0     # 灭灯为0
                cv2.putText(img, str((i[0], i[1])), (i[0]-20, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(img, 'OFF', (i[0], ymax+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                #cv2.putText(img, str(linght_NO), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (202, 255, 112 ), 1, cv2.LINE_AA)
                #print('{}号指示灯的圆心位置为（{},{}）的指示灯状态为：灭灯'.format(linght_NO, i[0], i[1]))

        #e2 = cv2.getTickCount()      #计算运行时间的代码
        #time = (e2 - e1) / cv2.getTickFrequency()
        #print(time)

        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite('initialize.jpg', img)      # 存储为图像
        #cv2.imwrite('detection/' + str(c) +'.jpg', img)
        c = circles[0:12, ]
        lights = np.append(c, state, axis=1)

        return lights