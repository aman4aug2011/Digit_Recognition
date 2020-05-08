import numpy as np
import cv2,time
from sklearn import svm
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from PIL import ImageGrab
import glob,csv,os
from sklearn import svm
import pandas as pd

#run it for label="0...9"
label="0"
img_list=glob.glob("F:\ML\Dataset/"+label+"/*.png")
for i in img_list:
    img=cv2.imread(i,0)
    gray=cv2.GaussianBlur(img,(15,15),0)
    roi=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA) #region of interest
    X=[]
    X.append(label)
    row,col=roi.shape
    for x in range(row):
        for y in range(col):
            k=roi[x,y]
            if(k>100):
                k=1
            else: 
                k=0
            X.append(k)
            
    with open("F:\ML\Dataset\data.csv",'a') as f: #append in binary
        writer=csv.writer(f)
        writer.writerow(X)
        

df=pd.read_csv("F:\ML\Dataset\data.csv")
df=df.sample(frac=1).reset_index(drop=True)
Xt=df.drop(["1"],axis=1)
Yt=df["1"]

#x_train,y_train=Xt[:10],Yt[:10]
clf = svm.SVC()
clf.fit(Xt,Yt)




for i in range(20):
    time.sleep(5)    
    live_img=ImageGrab.grab(bbox=(80,80,200,200))
    live_img
    live_img.save("F:\ML"+str(i)+".png")
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) -> 0 in imread
    img=cv2.imread("F:\ML"+str(i)+".png",0)
    img
    gray=cv2.GaussianBlur(img,(15,15),0)
    #ret,th_img=cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    re=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)
    #re.shape
    #cv2.imshow("live_img",re) 
    #cv2.waitKey(0)
    X=[]
    a=-1
    row,col=re.shape
    for x in range(row):
        for y in range(col):
            k=re[x,y]
            if(k>100): 
                k=1
            else: 
                k=0
            X.append(k)
    pred=clf.predict([X])      
    print(pred)
   # print(len(X))
    #arr=np.array(X)
    #len(X)
    #arr.shape
    #[X]
   # pred=clf.predict(x_test)
    
#    print("prediction:"+pred[0])