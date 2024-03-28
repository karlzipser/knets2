from knets2.process_args import *

kprint(p.__dict__,'p.__dict__')


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
