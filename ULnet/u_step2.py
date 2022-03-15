import cv2
import numpy as np

lights = np.loadtxt('Calib/3z3.txt')
lights = np.uint16(np.around(lights))  # 取整数
#print(lights)
number_of_light = len(lights)
vedio_name = '3z'
cap = cv2.VideoCapture('vedio/'+ vedio_name +'.mp4')    #读取视频
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("width: %d" % width)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("height: %d" % height)
fps = cap.get(cv2.CAP_PROP_FPS)                       # 实时帧数
print("fps: %d" % fps)
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)       # 总帧数
print("total_frame: %d" % total_frame)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')   #写入视频文件保存
out = cv2.VideoWriter("output_vedio/result_" + vedio_name + ".mp4", fourcc, 30, (960, 544))

c = 1
while(cap.isOpened()):
    ret, img = cap.read()       # 读取视频每一帧
    if ret == False:
        print("未读取到图像")
        break  # 读到文件末尾

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    # 设置亮度的阈值
    v_thed = 0.5

    # 定位检测到的指示灯
    for i in lights[:number_of_light]:
        img = img.copy()
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 3)                                             #标出圆
        cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)                                             #标出圆心

        # 计算均值
        t = int(i[2] / 1.414)
        xmin = i[0] - t
        ymin = i[1] - t
        xmax = i[0] + t
        ymax = i[1] + t
        v_total = 0

        for x_num in range(xmin, xmax):
            for y_num in range(ymin, ymax):
                v_normal = v[y_num, x_num] / 255
                v_total += v_normal

        v_mean = v_total / (4 * t * t)
        #print("v_mean is %f"%v_mean)

        # 判断亮度
        if v_mean > v_thed:
            cv2.putText(img, 'ON', (i[0], ymax + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, 'OFF', (i[0], ymax + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    out.write(img)
    #cv2.imwrite('vediok/' + str(c) + '.jpg', img)  # 存储为图像
    #cv2.imshow('video_result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    c += 1

cap.release()                                #释放cap
cv2.destroyAllWindows()                      #销毁所有窗口

