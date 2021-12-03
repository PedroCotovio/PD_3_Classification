import os
import inspect

def get_path(path='dataset', opt_print=True):

    """

    :param path: str, path to datasets folder from current path
    :return: path to datasets folder
    """
    
    pth = os.getcwd()
    pth = os.path.join(pth, str(path))
    if opt_print is True:
        print('Path to Data: {}'.format(pth))
    return pth

def rescue_code(function):
    """
    retrieve deleted jupyter functions, that are defined in the kernel
    """
    get_ipython().set_next_input("".join(inspect.getsourcelines(function)[0]))
