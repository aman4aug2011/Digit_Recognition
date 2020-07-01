import numpy as np
import cv2,os
#from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

path="F:/ML/digit_recog_own/Dataset"
folders=os.listdir(path)
noOfClasses=len(folders)
imgs=[]
id=[]

for x in range(0,noOfClasses):
    folder=os.listdir(path+"/"+str(x))
    for y in folder:
        img = cv2.imread(path+"/"+str(x)+y)
        img = cv2.resize(img,(28,28))
        imgs.append(img)
        id.append(x)
        
imgs=np.array(imgs)
id=np.array(id)

def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img/255    #feature scaling(normalization(255 pixels))
    return img

imgs=np.array(list(map(preprocessing,imgs)))
id=to_categorical(id,noOfClasses)

def model():
    