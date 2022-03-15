import cv2
import numpy as np

number_of_light = 12                  ##  识别的指示灯个数
# cap = cv2.VideoCapture(0)           ##  读取摄像头
cap = cv2.VideoCapture('test.mp4')    #读取视频
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("width: %d" % width)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("height: %d" % height)
fps = cap.get(cv2.CAP_PROP_FPS)                       # 实时帧数
print("fps: %d" % fps)
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)       # 总帧数
print("total_frame: %d" % total_frame)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')   #写入视频文件保存
out = cv2.VideoWriter("./vedio_result.mp4", fourcc, 15, (1920, 896))


## 初始化后获得的每行灯的纵坐标y的平均值
point1 = 357       #light1_y
point2 = 646       #light2_y
center_line=501    #light_y


state = np.zeros((30, 12))       #12个指示灯，30帧内的亮暗状态
state_single = np.zeros((1, 12))
timeDet = 30      # 视频帧计数间隔频率


c =0
while(cap.isOpened()):
    ret, img = cap.read()       # 读取视频每一帧
    if ret == False:
        print("未读取到图像")
        break  # 读到文件末尾

    r = cv2.pyrUp(img)          # 2倍上采样

    timeF = 1                   # 视频帧计数间隔频率
    c += 1
    j = 0

    if (c % timeF == 0):        # 每隔timeF帧进行操作

        # 用霍夫变换检测指示灯
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)  # 灰度处理
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯模糊3*3
        circles1 = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=10,
                                    maxRadius=80)  # 霍夫圆变换
        # print(np.shape(circles1))               #三维数组 1*n*3
        circles = circles1[0, :, :]              # 降维 n*3
        circles = np.uint16(np.around(circles))  # 取整数

        num_of_circles = len(circles)
        #print("检测到的圆个数为：%d"%num_of_circles)

        circles_cut = []
        circles_1 = []
        circles_2 = []

        for i in circles[:num_of_circles]:
            #print(i[0], i[1])
            if 480 < i[0] < 1400 and 380 < i[1] < center_line:
                circles_1.append(i)
                circles_cut.append(i)
            if 480 < i[0] < 1400 and 600 < i[1] < 750:
                circles_2.append(i)
                circles_cut.append(i)
            if len(circles_cut) == 12:
                continue
        print("挑选完以后剩下的圆： %d"%len(circles_cut))

        # 将12个灯分为两层，每层排序
        light1 = sorted(circles_1, key=(lambda x: [x[0], x[1]]))
        light2 = sorted(circles_2, key=(lambda x: [x[0], x[1]]))

        light1 = np.array(light1)
        light2 = np.array(light2)


        number_of_circles = len(circles_cut)          #
        if number_of_circles < 12:
            #number_of_light = number_of_circles
            print("只检测到{}个指示灯".format(number_of_circles))
            continue
        light = np.concatenate((light1, light2), axis=0)
        #print(light)

        linght_NO = 0  # 指示灯编号
        print("灯的数量： %d"%number_of_light)
        gray_value = 0.22                             #

        #定位检测到的指示灯
        for i in light[:12]:
            # cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 3)                   # 标出圆
            # cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)                                       # 标出圆心
            cv2.rectangle(r, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 3)  # 标出方形
            # 计算灰度均值
            t = int(i[2] / 1.414)
            if t==0:
                break
            xmin = i[0] - t
            ymin = i[1] - t
            xmax = i[0] + t
            ymax = i[1] + t
            gray_total = 0
            # print(ymin)
            # print(ymax)
            for x_num in range(xmin, xmax-5):
                for y_num in range(ymin, ymax-5):
                    #print("gray_normal is %d" % y_num)
                    gray_normal = gray[y_num-8, x_num-8] / 255
                    gray_total += gray_normal
            gray_mean = gray_total / (4 * t * t)
            # print("gray_mean is %f"%gray_mean)

            # 与灰度的阈值进行比较，判断指示灯的亮暗
            gray_res = gray_mean - gray_value
            # 判断状态，并将结果打在屏幕上
            linght_NO += 1
            if gray_res > 0:
                state_single[:, linght_NO - 1] = 1  # 亮灯记为1
                cv2.putText(r, str(linght_NO), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

            else:
                state_single[:, linght_NO - 1] = 0  # 灭灯记为0
                cv2.putText(r, str(linght_NO), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

            #指示灯状态更新
            for i in range(29):
                state[i] = state[i+1]
            state[29] = state_single

            print(state)

            #根据30帧内的明暗情况，对每个指示灯的状态进行判断
            for i in range(12):
                sum_of_state = sum(state[:, i])
                value = sum_of_state/30
                #print(value)
                Light = i + 1
                if value == 1:
                    cv2.putText(r, 'ON', (light[i][0], light[i][1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                    print('{}号指示灯的圆心位置为（{},{}）的指示灯状态为：亮灯'.format(i+1, light[i][0], light[i][1]))
                    # 报警设置
                    if Light == (2 or 3 or 4 or 5 or 6 or 8 or 10 or 11 or 12):
                        print("--------------警告%d号指示灯灭了---------------" % linght_NO)
                elif value == 0:
                    cv2.putText(r, 'OFF', (light[i][0], light[i][1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                    print('{}号指示灯的圆心位置为（{},{}）的指示灯状态为：灭灯'.format(i+1, light[i][0], light[i][1]))
                    # 报警设置
                    if Light == (1 or 7 or 9):
                        print("--------------警告%d号指示灯灭了---------------"%linght_NO)
                else:
                    cv2.putText(r, 'Blink', (light[i][0], light[i][1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                    print('{}号指示灯的圆心位置为（{},{}）的指示灯状态为：闪烁'.format(i+1, light[i][0], light[i][1]))

        out.write(r)
        cv2.imwrite('videok/' + str(c) + '.jpg', r)  # 存储为图像
        cv2.imshow('video_result', r)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()