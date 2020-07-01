import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import cv2,time
from PIL import ImageGrab, Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#from keras import backend as K
# the data, split between train and test sets





(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)

num_classes = 10

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

data_gen= ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
data_gen.fit(x_train)
y_train = keras.utils.to_categorical(y_train, num_classes)
x_train = x_train.astype('float32')
x_train/=255

batch_size = 128
epochs = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=3,activation='relu',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=3,activation='relu'))

#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

model.fit(x_train, y_train,epochs=epochs)

for i in range(100):
    #time.sleep(5)    
    
    live_img=ImageGrab.grab(bbox=(80,80,200,200))
    live_img.save("F:\ML"+str(i)+".png")
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) -> 0 in imread
    img=image.load_img("F:\ML"+str(i)+".png",target_size=(28,28))
    img = img.convert('L')
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    #convert rgb to grayscale
    img = np.array(img)
    #gray=cv2.GaussianBlur(img,(15,15),0)
    #ret,th_img=cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    #img=cv2.resize(img,(28,28))
    #img = np.reshape(img,(1,28,28,1))
    img = img/255
    #predicting the class
    classIndex=model.predict_classes(img)
    res = model.predict(img)
    print(classIndex,np.amax(res))