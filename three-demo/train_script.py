from model.vgg16 import VGG16
from keras.engine import Input
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, Embedding, LSTM
from keras.layers.merge import Concatenate, concatenate
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from generator.products_data_generator import ProductsDataGenerator

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'D:/images/10classes_2017-06-23/10classes_2017-07-12/train'
validation_data_dir = 'D:/images/10classes_2017-06-23/10classes_2017-07-12/validation'
nb_train_samples = 12131
nb_validation_samples = 4050
epochs = 10
batch_size = 16
classes = 10
max_images = 3
max_features = 1000
maxlen = 15

# build the VGG16 network
input_tensor = Input(shape=(max_images, img_width, img_height))
model = VGG16(input_tensor=input_tensor, weights=None, include_top=False, classes=classes,
              input_shape=(max_images, img_width, img_height))
last = model.output
x = Flatten()(last)
'''x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(classes, activation='softmax')(x)
new_model = Model(model.input, preds)'''

input_2 = Input(shape=(maxlen,))

text_model = Embedding(max_features, 64, input_length=maxlen)(input_2)
text_model = LSTM(64)(text_model)

m = concatenate([x, text_model], axis=1)

x = Dense(4096, activation='relu')(m)
x = Dropout(0.5)(x)
# merged = Dense(4096, activation='relu')(merged)
# merged = Dropout(0.5) (merged)
merged = Dense(classes, activation='softmax')(x)
final_model = Model(inputs=[input_tensor, input_2], outputs=merged)

print('Model loaded.')

final_model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-4),
                    metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ProductsDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ProductsDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    max_images=max_images,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    max_images=max_images,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
tensorboard = TensorBoard()
# fine-tune the model
final_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[checkpointer])
# print(new_model.summary())
