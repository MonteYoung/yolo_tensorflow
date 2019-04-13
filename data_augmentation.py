# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

ia.seed(1)


# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
    return images



if __name__ == '__main__':
    images = load_images_from_folder("/home/yang/PycharmProjects/carND/video_iamge/")
    seq = iaa.Sequential([
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ], random_order=True) # apply augmenters in random order

    images_aug = seq.augment_images(images)
    seq.show_grid(images[0], 4,4)

