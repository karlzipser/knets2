
from utilz2 import *

"""

DO NOT EDIT THESE FOR SPECIFIC PROJECT OR RUNTIME

"""

all_args=dict(

    display_timer_time = 15,
    save_figure = False,
    save_figure_timer_time=300,
    status_timer_time = minute,
    freq_timer_time = minute,
    load_weights = True,
    save_timer_time = 300,
    save_individual_weight_updates = False,
    weights_name='latest.pth',
    weights_path='weights',
    runfolder='',
    project='',
    base_path = '',
    workers = 4,
    batch_size = 16,
    showbatch=False,
    image_size = 128,
    num_epochs = 5,
    lr =  0., # must be initalized to 0.
    beta1 = 0.5,
    print_nets = False,
    datapath=opjD('data/kgan-data'),
    do_GAN_training=1.,
    do_autoencoder_training=1.,
    do_bounding_box_training=0.,
    device='cuda:0',
    R_seedweights='',
    G_seedweights='',
    D_seedweights='',
    todo='',
    plotcorrcoef=False,
    processlosses=False,
    do_ious=0.,
    numimagestoshow=8,
    figwidth=17,
    figheight=10,

    use_center_mask=False,

    RandomPerspective=False,
    RandomPerspective_distortion_scale=0.5,
    RandomPerspective_p=0.95,
    RandomPerspective_fill=(.5,.5,.5),

    RandomRotation=False,
    RandomRotation_angle=15,
    RandomRotation_fill=(.5,.5,.5),

    Pad=False,
    Pad_fill=(0.5,0.5,0),
    Resize=False,
    CenterCrop=False,
    
    RandomResizedCrop=False,
    RandomResizedCrop_scale=(0.5,1),
    RandomResizedCrop_ratio=(0.75,1.333),

    RandomHorizontalFlip=False,
    RandomHorizontalFlip_p=0.5,
        
    RandomVerticalFlip=False,
    RandomVerticalFlip_p=0.5,

    RandomZoomOut=False,
    RandomZoomOut_fill=(0.5,0.5,0),
    RandomZoomOut_side_range=(1.0,6.0),

    ColorJitter=False,
    ColorJitter_brightness=(.9,1.1),
    ColorJitter_contrast=(0.9,1),
    ColorJitter_saturation=(0.75,1),
    ColorJitter_hue=(-.02,.02),

    usetrainingschedule=False,
    starttime=0.,
    t0=time.time(),

    proportion_of_data_to_use=1.,
    use_try_in_main=False,
    startk='t_longterm',
    savelatents=False,

    sleeptime=0,
    task='train', # or 'test'
    shuffle=True,

    nc=3,
    nz=100,
    ndf=32,

    n_random_rectangle_masks=0,
    swap_image_portions=False,
)
assert not all_args['lr'] # non-zero values should be set in command line

#EOF
