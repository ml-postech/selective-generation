import argparse

def to_tree_namespace(params):
    """
    handle only depth-1 tree
    """
    for key, val in vars(params).copy().items():
        if "." in key:
            delattr(params, key)
            group, sub_key = key.split(".", 2)
            if not hasattr(params, group):
                setattr(params, group, argparse.Namespace())
            setattr(getattr(params, group), sub_key, val)
    return params

def print_args(params, param_str="args", n_tap_str=1):
    print("\t"*(n_tap_str-1)+param_str + ":" )
    for key, val in vars(params).items():
        if "Namespace" in str(type(val)):
            print_args(val, param_str=key, n_tap_str=n_tap_str+1)
        else:
            print("\t"*n_tap_str + key + ":", val)
    print()


def propagate_args(args, name):
    for k, v in args.__dict__.items():
        if isinstance(v, argparse.Namespace):
            a = getattr(args, k)
            setattr(a, name, getattr(args, name))
            setattr(args, k, a)
    return args
        
    
