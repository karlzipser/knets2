
from knets2.imports import *
from knets2.process_args import get_p

p=get_p()

kprint(p.__dict__,'p.__dict__')

##############

from knets2.dataloaders.multi_folder_dataloader import get_dataloader
from utilz2.dev.view_tensor import get_image_of_tensor
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    dataloader=get_dataloader(p)

    for i, datadic in enumerate(dataloader, 0):
        data=datadic['img']
        print(data.size())
        break

    bs=data.size()[0]
    mapping={
        0:0,
        1:0,
        2:0,
        3:1,
        4:1,
        5:1,
        6:2,
        7:2,
        8:2,
    }
    g=get_image_of_tensor(data,mapping)
    sh(g,r=1)

###########

if False:
    def set_weights_path(p):
        fs=sggo(pname(opj(p.runfolder)),'*')
        fs=['<None>']+fs
        _w=select_from_list(fs,title=d2n('Select desired run for project ',qtds(p.project),':'))
        if _w=='<None>':
            p.load_weights=False
            p.weights_path=''
            print('*** Starting with random weights')
        else:
            p.load_weights=True
            p.weights_path=opj(_w,'weights')
    set_weights_path(p)


#EOF
