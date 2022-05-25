import torch
from torchvision import transforms as tf

class Unnormalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class BaseTransform:
    def __init__(self, normalization:tuple, size:int):
        self.mean, self.std = normalization
        self.size = size
        self.norm = tf.Compose([tf.ToTensor(),
                                tf.Normalize(self.mean, self.std)])
        self.base = tf.Compose([tf.Resize(size),
                                tf.ToTensor(),
                                tf.Normalize(self.mean, self.std)])
        self.tfms = self.norm
        self.transforms = getattr(self.tfms, 'transforms')

        self.unnorm = Unnormalize(self.mean, self.std)

    def __call__(self, img):
        return self.tfms(img)

    def extend_transforms(self, tfms):
        new_tfms = tf.Compose([tf.Resize(self.size), *tfms, *self.norm.transforms])
        return new_tfms