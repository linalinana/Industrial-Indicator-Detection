# Industrial-Indicator-Detection
Detection and status recognition of industrial indicator lights

Abstract  
This project proposes a method of indicator calibration and state recognition to reduce the burden of the operator and improve the accuracy of the state recognition.   
First, a lightweight salient object detection model is designed to detect the rough area of the indicators for automatically calibrating.   
Then, Hough transform is used to search for the pixel coordinates of the indicators locally.   
Finally, an online algorithm based on HSV color space for the recognition of color and the on-off state of the indicators is proposed.   
This project  can reduce the calibration burden of the operator and realize the real-time state identification of the indicators during video surveillance.  

Environment  
Python = 3.6  
Pytorch = 1.6  
OpenCV = 3.4  

Reference  
Qin X, Zhang Z, Huang C, et al. U2-Net: Going deeper with nested U-Structure for salient object detection[J]. Pattern Recognition, 2020,106:107404.
