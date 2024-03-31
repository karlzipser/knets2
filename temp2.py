32x32 gan
autoencoder
rectangular vessels
transformers
get losses done correctly, save with timestamps so resampling is easy,
allow for averaging within time windows, hold multiple loss types


if p.project==...:
    p.prepare_show_data=...
    p.setup_input=...
    do_training_list=[
        do_GAN_training,
        do_autoencoder_training,
        do_bounding_box_training,
        do_ious,
        render_graphics,
        save_weights,
    ]
    p.netsdic=...

def training_loop():

    print("Starting Training Loop...")

    p.NE=NetworkEnsemble(
        p.netsdic,
    ) 

    dataloader=get_dataloader(p)


    while True:

            for i, datadic in enumerate(dataloader, 0):

                p.setup_input(datadic)
                
                for train_function in do_training_list:
                    train_function(p)



#EOF
