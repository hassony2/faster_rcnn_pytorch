# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import PIL
import numpy as np
from torch.utils.data import Dataset

from ..utils.cython_bbox import bbox_overlaps


class HandDetectionDataset(Dataset):
    """Image database."""

    def __init__(self, name, left_from_right=False):
        self.name = name
        self.left_from_right = left_from_right
        if left_from_right:
            self.classes = ['background', 'right_hand', 'left_hand']
        else:
            self.classes = ['background', 'hand']
        self.num_classes = len(self.classes)
        self._obj_proposer = 'selective_search'

        self.datadir = 'data'
        # Matching lists of image paths and annotations
        self.image_names = []
        self.annotations = []

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(self.datadir, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def __len__(self):
        return len(self.image_names)

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def flip_bboxes(self, img, annot):
        """
        Flips annot

        Args:
        img: PIL image
        annot: dict with keys boxes, gt_overlaps, gt_classes, flipped
        """
        width = img.size[0]
        boxes = annot['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        flipped_annot = {
            'boxes': boxes,
            'gt_overlaps': annot['gt_overlaps'],
            'gt_classes': annot['gt_classes'],
            'flipped': True
        }
        return flipped_annot

    def evaluate_recall(self,
                        candidate_boxes=None,
                        thresholds=None,
                        area='all',
                        limit=None):
        """Evaluate detection proposal recall metrics.

        Returns:
        results: dictionary of results with keys
        'ar': average recall
        'recalls': vector recalls at each IoU overlap threshold
        'thresholds': vector of IoU overlap thresholds
        'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {
            'all': 0,
            'small': 1,
            'medium': 2,
            'large': 3,
            '96-128': 4,
            '128-256': 5,
            '256-512': 6,
            '512-inf': 7
        }
        area_ranges = [
            [0**2, 1e5**2],  # all
            [0**2, 32**2],  # small
            [32**2, 96**2],  # medium
            [96**2, 1e5**2],  # large
            [96**2, 128**2],  # 96-128
            [128**2, 256**2],  # 128-256
            [256**2, 512**2],  # 256-512
            [512**2, 1e5**2],  # 512-inf
        ]
        assert area in areas, 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in range(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(
                axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(
                boxes.astype(np.float), gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in range(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert (gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert (_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {
            'ar': ar,
            'recalls': recalls,
            'thresholds': thresholds,
            'gt_overlaps': gt_overlaps
        }

    def enrich_annots(self):
        """Enrich the annotation list by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        img_paths = self.image_names
        sizes = [PIL.Image.open(img_path).size for img_path in img_paths]
        for i in range(len(img_paths)):
            self.annotations[i]['image'] = img_paths[i]
            self.annotations[i]['width'] = sizes[i][0]
            self.annotations[i]['height'] = sizes[i][1]

            gt_overlaps = self.annotations[i]['gt_overlaps']

            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)

            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            self.annotations[i]['max_classes'] = max_classes
            self.annotations[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            if not all(max_classes[nonzero_inds] != 0):
                import pdb
                pdb.set_trace()
            assert all(max_classes[nonzero_inds] != 0)


def concat_datasets(dataset_list):
    """Returns new dataset that is a combination of the listed datasets

    Args:
    dataset_list (list): list of datasets to concatenate
    """
    left_from_rights = [dataset.left_from_right for dataset in dataset_list]
    assert_string = 'all left_from_rights statemeent should be the same\
    but got'.format(left_from_rights)

    assert all(flag == left_from_rights[0]
               for flag in left_from_rights), assert_string
    dataset_name = '_'.join([dataset.name for dataset in dataset_list])
    all_dataset = HandDetectionDataset(
        dataset_name, left_from_right=left_from_rights[0])
    all_dataset.image_names = [
        img_path for dataset in dataset_list
        for img_path in dataset.image_names
    ]
    all_dataset.annotations = [
        annot for dataset in dataset_list for annot in dataset.annotations
    ]
    assert_string = 'Annotations and image paths should be of same lenght after dataset\
        concat but are {} and {}'

    assert_string = assert_string.format(
        len(all_dataset.image_names), len(all_dataset.annotations))
    assert len(all_dataset.image_names) == len(all_dataset.annotations)
    return all_dataset
