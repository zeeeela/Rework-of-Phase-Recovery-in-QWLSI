import ml_collections
from ml_collections import config_dict

def d(**kwargs): 
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config():
    config = config_dict.ConfigDict()
    config.simul = d(
        A = 1.0,
        d = 5,
        s = 14,
        )
    
    config.simul.u0 = 1 / config.simul.d
    config.simul.v0 = 1 / config.simul.d
    config.seed = 42
    
    return config