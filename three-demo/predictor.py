from keras import applications
from keras.engine import Input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from PIL import Image
import os
import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator

weights_path = 'D:/ml_result2017-06-05/weights6.hdf5'
top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'D:/ml_result2017-06-05/images/DeskTopPC'
validation_data_dir = 'D:/3classes2017-06-03/image3classes2017-06-03/test3classvalidation'
nb_train_samples = 1289
nb_validation_samples = 300
epochs = 20
batch_size = 16

# build the VGG16 network
input_tensor = Input(shape=(224, 224, 3))
model = applications.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False, classes=3)
last = model.output
x = Flatten()(last)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
# x =Dense(4096,activation='relu')(x)
# x = Dropout(0.5)(x)
preds = Dense(3, activation='softmax')(x)
new_model = Model(model.input, preds)
new_model.load_weights(weights_path)
print('Model loaded.')


def img_to_array(path):
    img = Image.open(path)
    new_img = img.resize((img_width, img_height))
    arr = np.asarray(new_img)
    # img.close()
    img.close()
    return arr


def generate_data(directory_path):
    X_data_list = []
    y_data_list = []
    X_data = np.array(X_data_list)
    y_data = np.array(y_data_list)
    classes_folder = os.listdir(directory_path)
    for i in range(len(classes_folder)):
        category = i
        item_list = os.listdir(os.path.join(directory_path, classes_folder[i]))
        for item_folder in item_list:
            image_list = os.listdir(os.path.join(directory_path, classes_folder[i], item_folder))
            images_predictions = []
            images_predictions = np.asarray(images_predictions)
            for image in image_list:
                image_array = img_to_array(os.path.join(directory_path, classes_folder[i], item_folder, image))
                image_array = image_array / 255
                try:
                    prediction_array = image_array.reshape(1, img_width, img_height, 3)
                except ValueError:
                    print(os.path.join(directory_path, classes_folder[i], item_folder, image))
                result = new_model.predict(prediction_array)
                images_predictions = np.append(images_predictions, result, axis=0)
                # images_predictions.append(result[0])
            X_data_list = np.append(X_data_list, images_predictions, axis=0)
            # X_data_list.append(images_predictions)
            y_data_list.append(category)
            # y_data_list.append(category)
    y_data = np.asanyarray(y_data_list)
    return X_data, y_data


test_datagen = ImageDataGenerator(rescale=1. / 255)

data_dir = 'D:/ml_result2017-06-05/rnn_set/test_img'


def generate_data(data_dir, max_length):
    classes_dir_list = os.listdir(data_dir)
    X_data = []
    y_data = []
    i = 0
    '''max_length = 0
    for class_folder in os.listdir(data_dir):
        for item in os.listdir(os.path.join(data_dir,class_folder)):
            if max_length < len(os.listdir(os.path.join(data_dir, class_folder, item))):
                max_length = len(os.listdir(os.path.join(data_dir, class_folder, item)))'''
    for class_dir in classes_dir_list:
        train_generator = test_datagen.flow_from_directory(
            os.path.join(data_dir, class_dir),
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
        steps = math.ceil(train_generator.samples / batch_size)

        result = new_model.predict_generator(train_generator, steps)
        file_names = train_generator.filenames
        score_dict = dict(zip(file_names, result))
        item_list = os.listdir(os.path.join(data_dir, class_dir))
        for item in item_list:
            item_score = []
            for image in os.listdir(os.path.join(data_dir, class_dir, item)):
                item_score.append(score_dict[item + '\\' + image])
                if len(item_score) == max_length:
                    break
            item_score = np.asanyarray(item_score)
            zero_array = np.zeros((max_length - len(item_score), 3))
            if not len(item_score) == 0:
                item_score = np.concatenate((item_score, zero_array))
            elif len(item_score) == 0:
                item_score = zero_array
            X_data.append(item_score)
            y_data.append(i)
        i = i + 1
    return np.asanyarray(X_data), np.asanyarray(y_data)


X_data, y_data = generate_data(data_dir,7)
print(X_data, y_data)
