import json
import os
import shutil

data = None

with open('json/item_title.json') as json_file:
    data = json.load(json_file)

folder_path = 'D:/images/10classes_2017-06-23/10classes_2017-07-12/validation'

for category in os.listdir(folder_path):
    for item in os.listdir(os.path.join(folder_path, category)):
        if not item in data:
            shutil.rmtree(os.path.join(folder_path, category, item))
