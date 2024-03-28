
from utilz2 import *

"""

edit for specific project

"""

project_args=dict(
    display_timer_time = 29,
    save_figure_timer_time=300,
    save_figure = True,
    save_timer_time = 300,
    load_weights=True,
    do_GAN_training=1.,
    do_autoencoder_training=1.,
    startk='t0',
    use_try_in_main=False,
    use_center_mask=True,
    n_random_rectangle_masks=0,
    swap_image_portions=False,
)
assert 'lr' not in project_args # should be set in commands line

#EOF
