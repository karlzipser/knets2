
from utilz2 import *

_fill=(0,0.5,0.5)

ensemble_args=dict(
    t0=dict(
        RandomPerspective=False,
        RandomRotation=False,
        RandomResizedCrop=True,
        RandomResizedCrop_scale=(1,1),
        RandomResizedCrop_ratio=(1,1),
        RandomHorizontalFlip=False,
        RandomVerticalFlip=False,
        RandomZoomOut=False,
        ColorJitter=False,
        proportion_of_data_to_use=1.,
    ),
    t8=dict(
        RandomPerspective=True,
        RandomPerspective_distortion_scale=0.5,
        RandomPerspective_p=0.6,
        RandomPerspective_fill=_fill,

        RandomRotation=True,
        RandomRotation_angle=15,
        RandomRotation_fill=_fill,

        RandomResizedCrop=True,
        RandomResizedCrop_scale=(0.5,1),
        RandomResizedCrop_ratio=(0.75,1.333),

        RandomHorizontalFlip=True,
        RandomHorizontalFlip_p=0.5,
            
        RandomVerticalFlip=False,
        RandomVerticalFlip_p=0.5,

        RandomZoomOut=True,
        RandomZoomOut_fill=_fill,
        RandomZoomOut_side_range=(1.0,2.0),

        ColorJitter=False,
    ),
    t9=dict(
        RandomPerspective=True,
        RandomPerspective_distortion_scale=0.7,
        RandomPerspective_p=0.95,
        RandomPerspective_fill=_fill,

        RandomRotation=True,
        RandomRotation_angle=15,
        RandomRotation_fill=_fill,

        RandomResizedCrop=True,
        RandomResizedCrop_scale=(0.5,1),
        RandomResizedCrop_ratio=(0.75,1.333),

        RandomHorizontalFlip=True,
        RandomHorizontalFlip_p=0.5,
            
        RandomVerticalFlip=False,
        RandomVerticalFlip_p=0.5,

        RandomZoomOut=True,
        RandomZoomOut_fill=_fill,
        RandomZoomOut_side_range=(1.0,4.0),

        ColorJitter=False,
    ),
)

#EOF
