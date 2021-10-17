import torchvision.transforms as T

try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur

    T.GaussianBlur = GaussianBlur

class VisnirSimSiamTransform():
    def __init__(self, image_size, train=True):
        image_size = 224 if image_size is None else image_size  # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.visnir_vis_mean_std = [[0.3614] * 3, [0.1909] * 3]
        self.visnir_nir_mean_std = [[0.6393] * 3, [0.1793] * 3]
        # self.visnir_vis_mean_std = [[0.3614], [0.1909]]
        # self.visnir_nir_mean_std = [[0.6393], [0.1793]]
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor()
            ])

    def __call__(self, x1, x2):
        x1 = T.Normalize(*self.visnir_vis_mean_std)(self.transform(x1))
        x2 = T.Normalize(*self.visnir_nir_mean_std)(self.transform(x2))
        return x1, x2
