import os
import pickle
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse

from datasets.imdb import imdb
from model.utils.config import cfg


class kittivoc(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'kittivoc_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = self._devkit_path
        self._classes = ('__background__',  # always index 0
                         'car')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._remove_empty_samples()
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'
        self._year = ''
        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': True,  # using difficult samples
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier
        :param index filename stem e.g. 000000
        :return filepath
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'KITTIVOC')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest, aka, the annotations.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _remove_empty_samples(self):
        """
        Remove images with zero annotation ()
        """
        print('Remove empty annotations: ',)
        for i in range(len(self._image_index) - 1, -1, -1):
            index = self._image_index[i]
            filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
            tree = ET.parse(filename)
            objs = tree.findall('object')
            non_diff_objs = [
                obj for obj in objs if
                int(obj.find('difficult').text) == 0 and obj.find('name').text.lower().strip() != 'dontcare']
            num_objs = len(non_diff_objs)
            if num_objs == 0:
                print(index,)
                self._image_index.pop(i)
        print('Done. ')

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            if len(non_diff_objs) != len(objs):
                print('Removed {} difficult objects'.format(
                    len(objs) - len(non_diff_objs)))
            objs = non_diff_objs
        # only need car, pedestrian, cyclist classes.
        need_objs = [obj for obj in objs if obj.find('name').text.lower().strip()
                     in ['car', 'dontcare']]
        objs = need_objs

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # just the same as gt_classes
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # --------------------------------------------------
        care_inds = np.empty((0), dtype=np.int32)
        dontcare_inds = np.empty((0), dtype=np.int32)
        # --------------------------------------------------

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(float(bbox.find('xmin').text) - 1, 0)
            y1 = max(float(bbox.find('ymin').text) - 1, 0)
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            # --------------------------------------------
            diffc = obj.find('difficult')
            difficult = 0 if diffc is None else int(diffc.text)
            ishards[ix] = difficult
            # --------------------------------------------

            class_name = obj.find('name').text.lower().strip()
            if class_name != 'dontcare':
                care_inds = np.append(care_inds, np.asarray([ix], dtype=np.int32))
            if class_name == 'dontcare':
                dontcare_inds = np.append(dontcare_inds, np.asarray([ix], dtype=np.int32))
                boxes[ix, :] = [x1, y1, x2, y2]
                continue
            cls = self._class_to_ind[class_name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        # deal with dontcare areas
        dontcare_areas = boxes[dontcare_inds, :]
        boxes = boxes[care_inds, :]
        gt_classes = gt_classes[care_inds]
        overlaps = overlaps[care_inds, :]
        seg_areas = seg_areas[care_inds]
        ishards = ishards[care_inds]

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'dontcare_areas': dontcare_areas,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    import pprint

    d = pascal_voc('trainval', '2007')
    pprint.pprint(d)
    res = d.roidb
    from IPython import embed;

    embed()
