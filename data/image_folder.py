###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import collections

DATA_EXTENSIONS = [
    '.npy',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in DATA_EXTENSIONS)

def make_dataset_with_labels(dir, classnames):
    # Support multiple source domains
    if isinstance(dir, list):
        images = []
        labels = []

        for d in dir:
            _images, _labels = make_dataset_with_labels(d, classnames)
            images += _images
            labels += _labels

        return images, labels

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    labels = []

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname not in classnames:
                continue

            label = classnames.index(dirname)

            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                labels.append(label)

    return images, labels

def make_dataset_classwise(dir, category):
    # Support multiple source domains
    if isinstance(dir, list):
        images = []

        for d in dir:
            images += make_dataset_classwise(d, category)

        return images

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname != category:
                continue
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def make_dataset(dir):
    # Support multiple source domains
    if isinstance(dir, list):
        images = []

        for d in dir:
            images += make_dataset(d)

        return images

    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
