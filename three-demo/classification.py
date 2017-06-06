'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras import applications
from keras.engine import Input
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras import backend
from keras.callbacks import ModelCheckpoint , TensorBoard

weights_path = 'D:/binary_classification_pictures/weights/weights2.hdf5'
top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'D:/3classes2017-06-03/image3classes2017-06-03/test3classtrain'
validation_data_dir = 'D:/3classes2017-06-03/image3classes2017-06-03/test3classvalidation'
nb_train_samples = 1289
nb_validation_samples = 300
epochs = 20
batch_size = 16

# build the VGG16 network
input_tensor = Input(shape=(224, 224,3))
model = applications.VGG16(input_tensor=input_tensor,weights='imagenet',include_top=False,classes=3)
last = model.output
x = Flatten()(last)
x =Dense(4096,activation='relu')(x)
x = Dropout(0.5)(x)
#x =Dense(4096,activation='relu')(x)
#x = Dropout(0.5)(x)
preds = Dense(3, activation='softmax')(x)
new_model = Model(model.input, preds)

#new_model.load_weights(weights_path)
print('Model loaded.')

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
#top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
#model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
#for layer in new_model.layers[:25]:
#    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
new_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.00001),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

checkpointer =ModelCheckpoint(filepath="D:/binary_classification_pictures/weights/weights6.hdf5", verbose=1, save_best_only=True)
tensorboard = TensorBoard()
# fine-tune the model
new_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[checkpointer,tensorboard])

#eval
'''
eval_datagen = ImageDataGenerator(rescale=1. / 255)
eval_generator = eval_datagen.flow_from_directory(
    'E:/BUI/X-POS50NEG54',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')
result = new_model.evaluate_generator(eval_generator,10)
print(result)'''
