from . import *

lookup = {
    GD.ID: GD
}


def OptimizerFactory(data:dict):
    opt_id = data.pop("ID")
    return lookup[opt_id].load(data)