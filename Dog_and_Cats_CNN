#o conjunto de imagens de cachorro e gato que peguei veem deste site https://www.kaggle.com/competitions/dogs-vs-cats/data?select=sampleSubmission.csv
#esse é um código de rede neural convolucional de gato e cachorro
import cv2 as cv, os, glob, random, shutil, tensorflow as tf, numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from google.colab.patches import cv2_imshow
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
cat_img_files=glob.glob('/content/cat/cat.*')
print(len(cat_img_files))
print(cat_img_files)
random_cats=[cv.imread(i) for i in cat_img_files]
cv2_imshow(random_cats[0])
dog_img_files=glob.glob('/content/dog/dog.*')
print(len(dog_img_files))
print(dog_img_files)
random_dogs=[cv.imread(i) for i in dog_img_files]
cv2_imshow(random_dogs[0])
os.mkdir('train_folder')
os.mkdir('train_folder/dog')
os.mkdir('train_folder/cat')
os.mkdir('val_folder')
os.mkdir('val_folder/dog')
os.mkdir('val_folder/cat')
os.mkdir('test_folder')
os.mkdir('test_folder/dog')
os.mkdir('test_folder/cat')
percent_val = 0.10
percent_test = 0.20
new_train_dog='/content/train_folder/dog'
new_train_cat='/content/train_folder/cat'
new_val_dog='/content/val_folder/dog'
new_val_cat='/content/val_folder/cat'
new_test_dog='/content/test_folder/dog'
new_test_cat='/content/test_folder/cat'

def moveImagesToCorrectFolder():
    dog_train_files = glob.glob('/content/dog/dog.*')
    cat_train_files = glob.glob('/content/cat/cat.*')
    for f in dog_train_files:
        rand_val = random.random()
        filename = f.split("/")[-1]
        if rand_val <= percent_val:
            shutil.move(f, new_val_dog + "/" + filename)
        elif rand_val > percent_val and rand_val <= percent_val + percent_test:
            shutil.move(f, new_test_dog + "/" + filename)
        else:
            shutil.move(f, new_train_dog + "/" + filename)
    for f in cat_train_files:
        rand_val = random.random()
        filename = f.split("/")[-1]
        if rand_val <= percent_val:
            shutil.move(f, new_val_cat + "/" + filename)
        elif rand_val > percent_val and rand_val <= percent_val + percent_test:
            shutil.move(f, new_test_cat + "/" + filename)
        else:
            shutil.move(f, new_train_cat + "/" + filename)
moveImagesToCorrectFolder()
print(tf.config.list_physical_devices('GPU'))
train_folder='/content/train_folder'
val_folder='/content/val_folder'
test_folder='/content/test_folder'
train_dataset = image_dataset_from_directory(train_folder,image_size=(180, 180),batch_size=32)
validation_dataset = image_dataset_from_directory(val_folder,image_size=(180, 180),batch_size=32)
test_dataset = image_dataset_from_directory(test_folder,image_size=(180, 180),batch_size=32)
model=keras.Sequential()
model.add(Rescaling(1./255))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
callbacks=[ModelCheckpoint(filepath="model3.keras",save_best_only=True,monitor="val_loss")]
history=model.fit(train_dataset,epochs=30,validation_data=validation_dataset,callbacks=callbacks)

img_ada='ada.jpg'
ada=cv.imread(img_ada)
cv2_imshow(ada)
img_gato='gato.jpeg'
gato=cv.imread(img_gato)
cv2_imshow(gato)
ada_img=image.load_img('ada.jpg', target_size=(180, 180))
x=image.img_to_array(ada_img)
x=np.expand_dims(x, axis=0)
pred=(model.predict(x) > 0.5).astype('int32')[0][0]

if pred == 1:
    print("Cachorro")
else:
    print("Gato")
    
print(model.predict(x))
gato_img=image.load_img('gato.jpeg', target_size=(180, 180))
x1=image.img_to_array(gato_img)
x1=np.expand_dims(x1, axis=0)
pred1=(model.predict(x1) > 0.5).astype('int32')[0][0]

if pred1==1:
  print('Cachorro')
else:
  print('Gato')

print(model.predict(x1))
