# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Pascal VOC database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

import cPickle
import os
import numpy as np
from tqdm import tqdm
from ..logger import logger
from imdb import IMDB

from ds_utils import unique_boxes, filter_small_boxes
import pandas as pd
from PIL import Image


class Nexar2(IMDB):
    def __init__(self, image_set, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        super(Nexar2, self).__init__('nexar2', image_set, root_path, devkit_path)  # set self.name
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = devkit_path

        self.classes = ['__background__',  # always index 0
                        'car']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('%s num_images %d' % (self.name, self.num_images))

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """

        if self.image_set == 'train':
            image_set_index = os.listdir(os.path.join(self.data_path, 'annotations'))
        else:
            image_set_index = os.listdir(os.path.join(self.data_path, self.image_set))

        image_set_index = map(lambda x: x[:-4], image_set_index)

        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, self.image_set, index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('%s gt roidb loaded from %s' % (self.name, cache_file))
            return roidb

        gt_roidb = [self.load_pascal_annotation(index) for index in tqdm(self.image_set_index)]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        logger.info('%s wrote gt roidb to %s' % (self.name, cache_file))

        return gt_roidb

    def load_pascal_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """

        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        img = Image.open(roi_rec['image'])
        width, height = img.size

        size = (height, width, 3)

        roi_rec['height'] = size[0]
        roi_rec['width'] = size[1]

        if self.image_set == 'train':
            filename = os.path.join(self.data_path, 'annotations', index + '.csv')
            df = pd.read_csv(filename)
            num_objs = df.shape[0]

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

            class_to_index = dict(zip(self.classes, range(self.num_classes)))
            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(df.index):
                # Make pixel indexes 0-based

                name = 'car'
                x1 = max(df.loc[ix, 'xmin'], 1)
                y1 = max(df.loc[ix, 'ymin'], 1)
                x2 = min(df.loc[ix, 'xmax'], width - 1)
                y2 = min(df.loc[ix, 'ymax'], width - 1)

                cls = class_to_index[name]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0

            roi_rec.update({'boxes': boxes,
                            'gt_classes': gt_classes,
                            'gt_overlaps': overlaps,
                            'max_classes': overlaps.argmax(axis=1),
                            'max_overlaps': overlaps.max(axis=1),
                            'flipped': False})
        else:
            boxes = np.zeros((1, 4), dtype=np.uint8)
            roi_rec.update({'boxes': boxes,
                            'flipped': False})
        return roi_rec

    def load_selective_search_roidb(self, gt_roidb):
        """
        turn selective search proposals into selective search roidb
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import scipy.io
        matfile = os.path.join(self.root_path, 'selective_search_data', self.name + '.mat')
        assert os.path.exists(matfile), 'selective search data does not exist: {}'.format(matfile)
        raw_data = scipy.io.loadmat(matfile)['boxes'].ravel()  # original was dict ['images', 'boxes']

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1  # pascal voc dataset starts from 1.
            keep = unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_roidb(self, gt_roidb, append_gt=False):
        """
        get selective search roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :param append_gt: append ground truth
        :return: roidb of selective search
        """
        cache_file = os.path.join(self.cache_path, self.name + '_ss_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('%s ss roidb loaded from %s' % (self.name, cache_file))
            return roidb

        if append_gt:
            logger.info('%s appending ground truth annotations' % self.name)
            ss_roidb = self.load_selective_search_roidb(gt_roidb)
            roidb = IMDB.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self.load_selective_search_roidb(gt_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        logger.info('%s wrote ss roidb to %s' % (self.name, cache_file))

        return roidb

    def evaluate_detections(self, detections, ann_type='bbox', all_masks=None, epoch=0):
        cls_ind = 1

        temp = []

        for im_ind, index in enumerate(self.image_set_index):
            dets = detections[cls_ind][im_ind]
            for k in range(dets.shape[0]):
                confidence = dets[k, -1]
                file_name = str(index) + '.jpg'
                x0 = dets[k, 0]
                y0 = dets[k, 1]
                x1 = dets[k, 2]
                y1 = dets[k, 3]

                temp += [(file_name, x0, y0, x1, y1, 'car', confidence)]

        df = pd.DataFrame(temp, columns=['image_filename', 'x0', 'y0', 'x1', 'y1', 'label', 'confidence'])

        df.to_csv(os.path.join('../data', self.image_set + '.csv'), index=False)
