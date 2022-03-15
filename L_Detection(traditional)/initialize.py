from Indicator_Detection import*

img = cv2.imread('10.jpg')   #读取图像
r = cv2.pyrUp(img)           #2倍上采样后的图像
num_of_light = 12            ## 识别的指示灯个数

#调用指示灯检测的函数
dec = Indicator_Detection()
initial = dec.detect(r, num_of_light)

#  对指示灯进行排序和定位
circles_cut = []
circles_1 = []
circles_2 = []

#  选择一定区域内的圆圈12个
for i in initial[:12]:
    if i[1] < 500:
        circles_1.append(i)
    if i[1] > 500:
        circles_2.append(i)           #

# 将12个灯分为两层，每层排序
light1 = sorted(circles_1, key=(lambda x: [x[0], x[1]]))
light2 = sorted(circles_2, key=(lambda x: [x[0], x[1]]))
light1_y = int((light1[0][1]+light1[1][1]+light1[2][1]+light1[3][1]+light1[4][1]+light1[5][1])/6)
light2_y = int((light2[0][1]+light2[1][1]+light2[2][1]+light2[3][1]+light2[4][1]+light2[5][1])/6)
light_y = int((light1_y+light2_y)/2)


print("light1_y= %d"%light1_y)
print("light2_y= %d"%light2_y)
print("light_y= %d"%light_y)

cv2.imshow('img', r)
cv2.waitKey(0)
cv2.destroyAllWindows()



