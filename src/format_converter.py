from __future__ import division

import os
import json


def save_image_set_txt(path, index):
    f = open(path, 'w')
    for ind in index:
        f.write(ind + '\n')
    f.close()


converted_annotations = 'annotations'
train_img_names = []

with open('rectangles_new.json') as data_file:
    data = json.load(data_file)

for i in range(len(data)):
    # if len(data[i]['annotations']) == 0:
    #     continue
    # get INDEX name for annotations (image filename without extension)
    INDEX = data[i]['filename'].split('/')[-1].split('.')[0]

    train_img_names.append(INDEX)

    # add basic annotations info
    f = open(converted_annotations + INDEX + '.xml', 'w')
    line = "<annotation>" + '\n'
    f.write(line)
    line = '\t\t<folder>' + "folder" + '</folder>' + '\n'
    f.write(line)
    line = '\t\t<filename>' + INDEX + '</filename>' + '\n'
    f.write(line)
    line = '\t\t<source>\n\t\t<database>Source</database>\n\t</source>\n'
    f.write(line)
    #     TODO: add proper image sizes
    #     (width, height) = image.shape[0], image.shape[1]
    (width, height) = 1000, 1000
    line = '\t<size>\n\t\t<width>' + str(width) + '</width>\n\t\t<height>' + str(height) + '</height>\n\t'
    line += '\t<depth>3</depth>\n\t</size>'
    f.write(line)
    line = '\n\t<segmented>Unspecified</segmented>'
    f.write(line)

    for annotation in data[i]['annotations']:
        x_min = max(int(annotation['x']), 2)
        y_min = max(int(annotation['y']), 2)

        w = int(annotation['width'])
        h = int(annotation['height'])

        x_max = min(999, x_min + w)
        y_max = min(999, y_min + h)

        line = '\n\t<object>'
        line += '\n\t\t<name>' + annotation['class'] + '</name>\n\t\t<pose>Unspecified</pose>'
        line += '\n\t\t<truncated>Unspecified</truncated>\n\t\t<difficult>0</difficult>'
        line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(x_min) + '</xmin>'
        line += '\n\t\t\t<ymin>' + str(y_min) + '</ymin>'
        line += '\n\t\t\t<xmax>' + str(x_max) + '</xmax>'
        line += '\n\t\t\t<ymax>' + str(y_max) + '</ymax>'
        line += '\n\t\t</bndbox>'
        line += '\n\t</object>\n'

        f.write(line)

    line = "</annotation>" + '\n'
    f.write(line)
    f.close()

# save_image_set_txt('train.txt', train_img_names)
