from keras.preprocessing.image import ImageDataGenerator
import math

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = test_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

steps = math.ceil(train_generator.samples / batch_size)

result = new_model.predict_on_generator(train_generator,steps)

file_names = train_generator.filenames
score_dict = dict(zip(file_names,result))