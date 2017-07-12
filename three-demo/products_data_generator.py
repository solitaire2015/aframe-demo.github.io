import threading
import json
from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, Iterator, K, array_to_img
from keras.preprocessing import text, sequence


class ProductsIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format='channels_first', max_images=8,
                 max_text_features=1000,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (max_images,)
            else:
                self.image_shape = (max_images,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.max_images = max_images

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        def _get_tokenizer(json, max_text_features):
            tokenizer = text.Tokenizer(num_words=max_text_features, lower=True, split=" ")
            title_list = []
            for key, value in json.items():
                title_list.append(value)
            tokenizer.fit_on_texts(title_list)
            return tokenizer

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for dirs in os.listdir(subpath):
                # for item in dirs:
                # is_valid = False
                if os.path.isdir(os.path.join(subpath, dirs)):
                    self.samples += 1
                    # for extension in white_list_formats:
                    #   if fname.lower().endswith('.' + extension):
                    #       is_valid = True
                    #       break
                    # if is_valid:
                    # self.samples += 1
        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        self.json = None
        with open('json/item_title.json') as data_file:
            self.json = json.load(data_file)
        self.tokenizer = _get_tokenizer(self.json, max_text_features)
        # second, build an index of the images in the different class subfolders
        self.items = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for dirs in os.listdir(subpath):
                # for item in os.path.join(subpath,dirs):
                if os.path.isdir(os.path.join(subpath, dirs)):
                    self.classes[i] = self.class_indices[subdir]
                    i += 1
                    absolute_path = os.path.join(subpath, dirs)
                    self.items.append(os.path.relpath(absolute_path, directory))
                    # for extension in white_list_formats:
                    #    if fname.lower().endswith('.' + extension):
                    #        is_valid = True
                    #        break
                    # if is_valid:
                    #    self.classes[i] = self.class_indices[subdir]
                    #    i += 1
                    #    # add filename relative to directory
                    #    absolute_path = os.path.join(root, fname)
                    #    self.filenames.append(os.path.relpath(absolute_path, directory))
        super(ProductsIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _generate_sequence(self, text, tokenizer, maxlen):
        seque = tokenizer.texts_to_sequences([text])
        seque = np.asanyarray(seque)
        seque = sequence.pad_sequences(seque, maxlen=maxlen, padding='post', truncating='post')
        return seque

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = []  # np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_x_text = []
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            item = self.items[j]
            img_ndarray_list = []
            item_text = self.json[item.split('\\')[-1]]
            item_sequence = self._generate_sequence(item_text, self.tokenizer,
                                                    15)  # todo add a parameter to change maxlen
            batch_x_text.append(item_sequence[0])
            for img in os.listdir(os.path.join(self.directory, item)):
                try:
                    image = Image.open(os.path.join(self.directory, item, img))
                    image = image.resize(self.target_size)
                    image = image.convert('L')
                    greyscale_map = list(image.getdata())
                    greyscale_map = np.array(greyscale_map)
                    greyscale_map = greyscale_map.reshape(self.target_size[1], self.target_size[0])
                    img_ndarray_list.append(greyscale_map)
                except IOError:
                    print(os.path.join(self.directory, item, img))
                    continue
                if len(img_ndarray_list) == self.max_images:
                    break
            length = len(img_ndarray_list)
            if length < self.max_images:
                for i in range(self.max_images - length):
                    empty_matrix = np.zeros(self.target_size)
                    img_ndarray_list.append(empty_matrix)
            x = np.asarray(img_ndarray_list)
            # x = x.reshape((self.target_size[0],self.target_size[1],self.max_images))

            # img = load_img(os.path.join(self.directory, fname),
            #               grayscale=grayscale,
            #               target_size=self.target_size)
            # x = img_to_array(img, data_format=self.data_format)
            # x = self.image_data_generator.random_transform(x)
            # x = self.image_data_generator.standardize(x)
            batch_x.append(x)
        batch_x = np.asanyarray(batch_x)
        batch_x_text = np.asanyarray(batch_x_text)
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(int(1e4)),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return [batch_x,  batch_x_text], batch_y


class ProductsDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            max_images=8,
                            max_text_features=1000,
                            save_format='jpeg',
                            follow_links=False):
        return ProductsIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            max_images=max_images,
            max_text_features=max_text_features,
            follow_links=follow_links)
