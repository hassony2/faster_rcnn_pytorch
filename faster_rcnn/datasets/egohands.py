import copy
import os
import pickle

import numpy as np
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset

from faster_rcnn.datasets.imdb import HandDetectionDataset
from faster_rcnn.datasets.utils.augmentation import data_augmentation


class EgoHandDataset(HandDetectionDataset):
    def __init__(self,
                 split,
                 transform=None,
                 transform_params=None,
                 use_cache=False,
                 left_from_right=False):
        """
        Args:
            split(str): either test or train
        """
        super(EgoHandDataset, self).__init__(
            'egohands_ + split', left_from_right=left_from_right)
        self.split = split
        self.use_cache = use_cache
        self.transform = transform
        self.transform_params = transform_params

        # Set usefull paths for given split
        self.name = 'egohands' + split
        self._data_path = os.path.join(self.datadir, 'egohands')
        self._video_path = os.path.join(self._data_path, '_LABELLED_SAMPLES')
        self._annot_path = os.path.join(self._data_path, 'metadata.mat')
        assert os.path.exists(
            self._data_path), 'Path does not exist: {}'.format(
                self._data_path)
        assert os.path.exists(self._annot_path), 'Path does not exist:\
            {}'.format(self._annot_path)

        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.video_nb = 48
        self.frame_nb = 100  # labelled frames per video

        self.load_dataset()
        self.num_samples = len(self.image_names)

    def __getitem__(self, idx):
        annotations = copy.deepcopy(self.annotations[idx])
        image_path = self.image_names[idx]
        img = Image.open(image_path)
        original_img = np.array(img)

        # Data augmentation for image
        if self.transform_params is not None:
            final_shape = self.transform_params['shape']
            jitter = self.transform_params['jitter']
            hue = self.transform_params['hue']
            saturation = self.transform_params['saturation']
            exposure = self.transform_params['exposure']
            img, annotations['boxes'] = data_augmentation(
                img, annotations['boxes'].copy(), final_shape, jitter, hue,
                saturation, exposure)
        if self.transform is not None:
            img = self.transform(img)
        return img, original_img, annotations

    def get_image_path(self,
                       video_idx,
                       frame_idx,
                       img_file_template='frame_{frame:04d}.jpg'):
        """To be called after self.load_dataset()
        """
        video_folder = os.path.join(self._video_path,
                                    self.video_ids[video_idx])
        frames_annots = self.videos_annots[0, video_idx]
        frame_annots = frames_annots[0, frame_idx]
        frame_num = frame_annots['frame_num'][0, 0]
        frame_path = os.path.join(
            video_folder, img_file_template.format(frame=frame_num))
        assert os.path.exists(frame_path), 'file {} not found'.format(
            frame_path)
        return frame_path

    def load_dataset(self):
        metadata = sio.loadmat(self._annot_path)
        annots_frame = metadata['video']  # numpy.ndarray of shape (1, 48)
        nd_video_ids = annots_frame['video_id']
        self.videos_annots = annots_frame['labelled_frames']
        self.video_ids = [
            str(nd_video_ids[0, i][0]) for i in range(self.video_nb)
        ]

        self.idx_tuples = [(video_idx, video_frame)
                           for video_idx in range(self.video_nb)
                           for video_frame in range(self.frame_nb)]
        self.image_indexes = range(len(self.idx_tuples))

        img_paths = [
            self.get_image_path(video_idx, frame_idx)
            for video_idx, frame_idx in self.idx_tuples
        ]
        self.image_names = img_paths
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up
        future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file) and self.use_cache:
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [
            self._annotation_from_index(index) for index in self.image_indexes
        ]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _annotation_from_index(self, index):
        """
        Load bounding boxes info from .mat metadata file
        """
        video_idx, frame_idx = self.idx_tuples[index]
        bboxes, overlaps, gt_classes = self.get_frame_annots(
            video_idx, frame_idx)

        return {
            'boxes': bboxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False
        }

    def get_frame_annots(self, video_idx, frame_idx):
        """Gets frame bounding boxes and left/right hand labels

            Returns:
                bbox(numpy.ndarray): each row is a [x_min, y_min, x_max, y_max]
                    bounding box
                hand_labels(numpy.ndarray): each value matches a bbox row with
                    1 for right and 0 for left hand
        """
        frames_annots = self.videos_annots[0, video_idx]
        frame_annots = frames_annots[0, frame_idx]

        def get_bb(frame_annots, segm_name):
            """Reads bbox from segmentation where frame_annots
            is the frame data_frame and segm_name is the key of
            the segmentation in the data_frame
            """
            my_left_seg = frame_annots[segm_name]
            if len(my_left_seg):
                x_min, y_min = my_left_seg.min(0)
                x_max, y_max = my_left_seg.max(0)
                bbox = np.array(
                    [int(x_min),
                     int(y_min),
                     int(x_max),
                     int(y_max)])
            else:
                bbox = None
            return bbox

        # Left bboxes
        bbox_myleft = get_bb(frame_annots, 'myleft')
        bbox_yourleft = get_bb(frame_annots, 'yourleft')

        # Right bboxes
        bbox_myright = get_bb(frame_annots, 'myright')
        bbox_yourright = get_bb(frame_annots, 'yourright')

        # Remove empty bboxes and stack them together
        # ! ordering of bboxes matters in labeling !
        bboxes = [bbox_myleft, bbox_yourleft, bbox_myright, bbox_yourright]
        bboxes = [bbox for bbox in bboxes if bbox is not None]
        if len(bboxes) == 0:
            # np_bboxes = None
            # labels = None
            np_bboxes = np.array([0, 0, 0, 0]).reshape(1, 4)
            labels = np.array([1])  # value has to be >0 to not break
            # enrich_annotations
        else:
            np_bboxes = np.stack(bboxes)

            # Retrieve labels as numpy array

            if self.left_from_right:
                left_hand_cls_idx = self._class_to_ind['left_hand']
                right_hand_cls_idx = self._class_to_ind['right_hand']
                left_labels = [
                    left_hand_cls_idx for bbox in [bbox_myleft, bbox_yourleft]
                    if bbox is not None
                ]
                right_labels = [
                    right_hand_cls_idx
                    for bbox in [bbox_myright, bbox_yourright]
                    if bbox is not None
                ]
                labels = np.array(left_labels + right_labels)
            else:
                hand_label = self._class_to_ind['hand']
                labels = hand_label * np.ones(len(bboxes)).astype(int)
            assert len(labels) == len(bboxes), 'label number {}\
                should match bbox count'.format(len(labels), len(bboxes))

        num_objects = len(labels)
        gt_overlaps = np.zeros((num_objects, self.num_classes))
        for ix, label in enumerate(labels):
            gt_overlaps[ix, label] = 1.0
        return np_bboxes, gt_overlaps, labels
