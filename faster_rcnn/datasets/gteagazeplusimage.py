import random
import re
import os

from faster_rcnn.datasets.imdb import HandDetectionDataset


class GTEAGazePlusImage(HandDetectionDataset):
    def __init__(
            self,
            root_folder="data/GTEAGazePlus",
            original_labels=True,
            seqs=['Ahmad', 'Alireza', 'Carlos', 'Rahul', 'Shaghayegh', 'Yin'],
            left_from_right=False):
        """
        Args:
        transform: transformations to apply during training
        base_transform: transformations to apply during testing
        untransform: transform to reapply after transformation
        to visualize original image
        use_video (bool): whether to use video inputs or png inputs
        """

        super().__init__('gtea', left_from_right=left_from_right)

        self.seqs = seqs
        self.seqs = seqs
        self.path = root_folder
        self.label_path = os.path.join(self.path, 'labels_cleaned')
        self.rgb_path = os.path.join(self.path, 'png')
        self.video_path = os.path.join(self.path, 'avi_files')

        self.action_clips = self.get_all_actions()
        self.image_names = [
            self.get_image_path(i) for i in range(len(self.action_clips))
        ]

    def get_image_path(self, index):
        # Load clip
        action, objects, subject, recipe, beg, end = self.action_clips[index]
        sequence_name = subject + '_' + recipe

        frame_idx = random.randint(beg, end)
        frame_name = "{frame:010d}.png".format(frame=frame_idx)
        img_path = os.path.join(self.rgb_path, sequence_name, frame_name)
        return img_path

    def get_all_actions(self):
        """Extracts all possible actions in the format (action,
        objects, subject, recipe, first_frame, last_frame) """
        annot_paths = [
            os.path.join(self.label_path, annot_file)
            for annot_file in os.listdir(self.label_path)
        ]
        actions = []
        # Get classes for each subject
        for subject in self.seqs:
            subject_annot_files = [
                filepath for filepath in annot_paths if subject in filepath
            ]
            for annot_file in subject_annot_files:
                recipe = re.search('.*_(.*).txt', annot_file).group(1)
                action_lines = process_lines(annot_file)
                for action, objects, begin, end in action_lines:
                    if (action, objects):
                        actions.append((action, objects, subject, recipe,
                                        begin, end))
        return actions


def process_lines(annot_path, inclusion_condition=None):
    """
    Returns list of action_object as
    ["action_name", "object1, object2", first_frame, last_frame]
    """
    with open(annot_path) as f:
        lines = f.readlines()
    processed_lines = []
    for line in lines:
        matches = re.search('<(.*)><(.*)> \((.*)-(.*)\)', line)
        if matches:
            action_label, object_label = matches.group(1), matches.group(2)
            begin, end = int(matches.group(3)), int(matches.group(4))
            object_labels = tuple(object_label.split(','))
            if inclusion_condition is None or\
                    inclusion_condition(action_label, object_labels):
                processed_lines.append((action_label, object_labels, begin,
                                        end))
    return processed_lines
