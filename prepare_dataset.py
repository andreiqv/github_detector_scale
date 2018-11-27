import os
import json
from tqdm import tqdm

directory_path = '/home/andrei/Data/Datasets/ScalesDetector/detector-261018/'

for file in tqdm(os.listdir(directory_path)):
    if file.endswith('.jpg'):
        image_file = directory_path + file
        json_file = directory_path + file + ".json"
        text_file = directory_path + file.replace(".jpg", ".txt")
        with open(text_file, "w") as text_file:
            if not os.path.isfile(json_file):
                text_file.write("0 0 0 0\n")
            else:
                with open(json_file, "r") as js:
                    json_obj = json.load(js)

                if len(json_obj["samples"]):
                    sample = json_obj["samples"][0]

                    x_min = sample["rect"][0]
                    x_max = sample["rect"][1]
                    y_min = sample["rect"][2]
                    y_max = sample["rect"][3]

                    w = (x_max - x_min)
                    h = (y_max - y_min)
                    x = x_min + w / 2.0
                    y = y_min + h / 2.0

                    text_file.write("{:.15f} {:.15f} {:.15f} {:.15f}\n".format(x, y, w, h))
                else:
                    text_file.write("0 0 0 0\n")
