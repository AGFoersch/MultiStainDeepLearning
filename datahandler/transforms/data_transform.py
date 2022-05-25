from torchvision import transforms
from base.base_transforms import BaseTransform
from utils import ifnone
import random


class RandomRotation():
    def __init__(self, degrees=None, p=0.5):
        self.degrees = ifnone(degrees, [0,90,180,270])
        self.p = p

    def __call__(self, img):
        degree = random.choice(self.degrees)
        num = random.random()
        return img.rotate(degree) if num < self.p else img


class BasicTransform(BaseTransform):
    def __init__(self,normalization=([0], [1]), size=512):
        super().__init__(normalization, size)


class ColorTransform(BaseTransform):
    def __init__(self, normalization=([0], [1]), hue=0, saturation=0, brightness=0, contrast=0, size=512):
        super().__init__(normalization, size)
        tfms = [transforms.ColorJitter(brightness, contrast, saturation, hue)]
        self.tfms = self.extend_transforms(tfms)


class StandardTransform(BaseTransform):
    def __init__(self, size, normalization=([0], [1]), hue=0, saturation=0, brightness=0, contrast=0):
        super().__init__(normalization, size)
        tfms = [transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRotation()]
        self.tfms = self.extend_transforms(tfms)


class ImageAugmentation(BaseTransform):
    def __init__(self, size=512, normalization=([0], [1]), hue=0, saturation=0, brightness=0, contrast=0):
        super().__init__(normalization, size)
        tfms = [transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRotation(),
                # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.98, 1.02)),
                transforms.ColorJitter(brightness, contrast, saturation, hue)
                ]
        self.tfms = self.extend_transforms(tfms)