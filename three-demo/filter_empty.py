import os


def filter_empty(dir):
    classes_list = os.listdir(dir)
    for class_dir in classes_list:
        for item in os.listdir(os.path.join(dir, class_dir)):
            if len(os.listdir(os.path.join(dir, class_dir, item))) == 0:
                os.rmdir(os.path.join(dir, class_dir, item))

filter_empty('D:/ml_result2017-06-05/images')
