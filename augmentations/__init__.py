from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
from .visnir_simsiam_aug import VisnirSimSiamTransform
from .visnir_byol_aug import VisnirByolTransform


def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None):

    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        elif 'visnir_simsiam' in name:
            augmentation = VisnirSimSiamTransform(image_size)
        elif name == 'visnir_byol':
            augmentation = VisnirByolTransform(image_size)
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        if 'visnir_simsiam' in name:
            augmentation = VisnirSimSiamTransform(image_size, train=False)
        elif name == 'visnir_byol':
            augmentation = VisnirByolTransform(image_size, train=False)
        else:
            augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








