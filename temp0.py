from utilz2 import *
import importlib
from knets2.get_parameters import get_p_from_args,set_weights_path

package_name='knets2'

all_args=importlib.import_module(d2p(package_name,'parameters.all')).all_args

def get_project_name_from_command_line():
    _project=''
    for i in range(1,len(sys.argv)-1):
        if sys.argv[i]=='--project':
            _project=sys.argv[i+1]
            break
    if not _project:
        cE('Error',__file__,': get_project_name_from_command_line(): --project not in command line')
        sys.exit()
    return _project

project_args=importlib.import_module(
    d2p(package_name,'parameters.project',get_project_name_from_command_line())).project_args

ensemble_args=importlib.import_module(d2p(package_name,'parameters.ensemble')).ensemble_args

kprint(all_args,'all_args')
kprint(project_args,'project_args')
kprint(ensemble_args,'ensemble_args')

p=get_p_from_args(project_args,all_args,ensemble_args)

set_weights_path(p)

kprint(p.__dict__,'p.__dict__')
print('done')


