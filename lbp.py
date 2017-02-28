# OpenCV bindings
import cv2
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
# Utility package -- use pip install cvutils to install
import cvutils
# To read class from file
import csv
import numpy as np


cap = cv2.VideoCapture(1)
train_images = cvutils.imlist("C:/Users/Petran/PycharmProjects/untitled3/Train/")
train_dic = {}
with open('C:\Users\Petran\PycharmProjects\untitled3\class_train.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        train_dic[row[0]] = int(row[1])

X_test = []
X_name = []
for train_image in train_images:
     # Read the image
    im = cv2.imread(train_image)
     # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
     # Number of points to be considered as neighbourers
    no_points = 8 * radius
     # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
     # Calculate the histogram
    x = itemfreq(lbp.ravel())
     # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
     # Append image path in X_name
    X_name.append(train_image)
     # Append histogram to X_name
    X_test.append(hist)


 # Create the identity filter
kernel = np.zeros((9, 9), np.float32)
kernel[4, 4] = 2.0  # Identity, times two!

 # Create a box filter:
boxFilter = np.ones((9, 9), np.float32) / 81.0

 # Subtract the two:
kernel = kernel - boxFilter
#color map -that u want to crop out
black_min = np.array([0, 0, 0], np.uint8)
black_max = np.array([255, 255, 100], np.uint8)
#string name gia img
a = "tstImgLbp"
while(1):

    ret ,frame = cap.read()
    if ret == True:
        #gray the frame ,blur it,threshold, find contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        GaussianBlur = cv2.GaussianBlur(gray, (25,25), 0)
        thresh = cv2.adaptiveThreshold(GaussianBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 1)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #in contours find center, radius
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 7000 or area > 40000:
                continue
            (x, y), rad = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            xi, yi = center
            rad = int(rad)
            #find the position of contoured item
            img2 = frame[y - (rad-10):y + (rad-10), x - (rad-10):x + (rad-10)]
            #manipulations to crop out unusefull puixels
            upstate_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            # get mask of pixels that are in black range
            try:
                mask_inverse = cv2.inRange(upstate_hsv, black_min, black_max)
            except Exception,exc2:
                print"error while processing item exc2:", exc2
                continue
            # inverse mask to get parts that are not blue
            mask = cv2.bitwise_not(mask_inverse)
            # convert single channel mask back into 3 channels
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            # perform bitwise and on mask to obtain cut-out image that is not blue
            masked_upstate = cv2.bitwise_and(img2, mask_rgb)
            # replace the cut-out parts with white
            masked_replace_black = cv2.addWeighted(masked_upstate, 1, cv2.cvtColor(mask_inverse, cv2.COLOR_GRAY2RGB), 0,0)
            #use sharp filter to sharpen the img
            sharpened = cv2.filter2D(masked_replace_black, -1, kernel)
            #write the image with the apropriate name
            cv2.imwrite(str(a)+".jpg", sharpened)
            im = cv2.imread(str(a)+'.jpg')
            # Convert to grayscale as LBP works on grayscale image
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            cv2.imshow("coin",im_gray)
            radius = 3
              # Number of points to be considered as neighbourers
            no_points = 8 * radius
              # Uniform LBP is used
            lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
            # Calculate the histogram
            x = itemfreq(lbp.ravel())
            # Normalize the histogram
            hist = x[:, 1] / sum(x[:, 1])
            #put the results on this array
            results = []
            # For each image in the training dataset
            # Calculate the chi-squared distance and then sort the values
            for index, x in enumerate(X_test):
                 try:
                    score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.cv.CV_COMP_CHISQR)
                    #print score
                 except Exception, exc1:
                    print"error while processing item exc1:", exc1
                    continue
                 results.append((X_name[index], round(score, 4)))
                 results = sorted(results, key=lambda score: score[1])
            font = cv2.FONT_HERSHEY_SIMPLEX
            try:
                for m,n in train_dic.items():
                    if(m in results[0]):
                        if(n==2 or n==11):
                            cv2.putText(gray, (str(n)+"euro"), (xi - rad, yi - rad), font, 1, (0, 0, 0), 2)
                        else:
                            cv2.putText(gray, (str(n) + "cent"), (xi - rad, yi - rad), font, 1, (0, 0, 0), 2)
            except Exception, exc3:
                print"error while processing item exc3:", exc3
                continue
            cv2.imshow('coin recognition window', gray)
    else:
        print("fail to capture any frame")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()