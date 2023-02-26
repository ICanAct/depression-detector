import importlib
import yaml
import os
import datetime

class Config(object):
    def __init__(self, type='twitter', file_name='twitter', test_path=None):
        assert type in ("twitter", "reddit"),\
            "Error! 'type' should be one of the following: 'twitter', 'reddit'"
        BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        dir_name = os.path.join(BASE_DIR, type)
        
        self._model_conf_file = os.path.join(dir_name, '{}.yml'.format(file_name))
        if test_path is not None:
            self._model_conf_file = test_path

    def _read_model_conf(self):
        with open(self._model_conf_file, encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def load(self):
        return self._read_model_conf()
    
    
def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_write(print_str, log_file):
    time_now = '[{}]'.format(datetime.now().strftime('%H:%M:%S'))
    print(time_now, *print_str)
    with open(log_file, 'a') as f:
        print(time_now, *print_str, file=f)