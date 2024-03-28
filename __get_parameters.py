
from knets2.imports import *


def get_p_from_args(project_args,all_args,ensemble_args):
    _project_args_to_all_args(project_args,all_args)
    p=_get_p(all_args)
    if not len(p.project):
        cE("ERROR, project not specified")
        assert False
    p.runfolder=opjh('kprojects',p.project,'runs',time_str())
    _check_parameters(p)
    if len(ensemble_args):
        assert 'lr' not in p.startk
        mergedict(p,ensemble_args[p.startk])
    if False:
        kprint(p.__dict__,title='p based on all_args, rutimeargs and command line args')
    p.command_line='$ python3 '+' '.join(sys.argv)
    print(p.command_line)
    return p


def _project_args_to_all_args(project_args,all_args):
    for k in project_args:
        if k=='todo':
            v='. . . '
        else:
            v=project_args[k]
        if False:
            clp('runtime:','`--r',d2n(k,'=',v))

        if k not in all_args:
            cE(k,'not in all_args')
            assert False
        if type(project_args[k])!=type(all_args[k]):
            cE("type(project_args[",k,"])!=type(all_args[",k,"])")
        all_args[k]=project_args[k]


def _get_p(all_args):
    if ('p' not in locals()) and (not interactive()):
        p=getparser(**all_args)
    else:
        p=kws2class(all_args)
        if '__file__' not in locals():
            cE("Warning, __file__=",__file__)
    return p


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


def _check_parameters(p):

    p.base_path = opj(pname(__file__))

    if not p.do_GAN_training:
        if not p.do_autoencoder_training:
            if not p.do_bounding_box_training:
                cE('Warning, all training off',r=True)

    if p.save_timer_time<60 or p.save_timer_time>600:
        cE('Warning, check out save_timer=',p.save_timer_time,r=False)
        
    if p.save_figure_timer_time<30:
        cE('Warning, check out save_figure_timer_time=',p.save_figure_timer_time,r=False)

    p.datapath=opj(p.datapath,p.task)

    for a in [p.base_path,p.datapath]:
        if not ope(a):
            cE('Error! Path',qtds(a),'does not exit!')
            assert(False)
    p.display_timer= Timer(p.display_timer_time); p.display_timer.trigger()
    p.save_timer=    Timer(p.save_timer_time)
    p.save_figure_timer=    Timer(p.save_figure_timer_time)
    
    p.status_timer=  Timer(p.status_timer_time)
    p.freq_timer=    Timer(p.freq_timer_time)

    if not torch.cuda.is_available():
        p.device="cpu"





#EOF
