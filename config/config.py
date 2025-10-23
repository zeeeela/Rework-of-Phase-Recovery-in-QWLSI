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

    config.model = d(
        learning_rate = 1e-3,
        num_epochs = 10,
        batch_size = 16,
        filters_down = [1, 32, 64, 128, 256, 512],
        filters_up = [512, 256, 128, 64, 32, 1],
        kernel_size_conv = 3,
        padding_conv = 1,
        pool_size = 2,
        kernel_size_conv_transpose = 2,
        stride_transpose = 2,
    )
    
    config.data = d(
        train_data_path = '/content/drive/MyDrive/dataset/simulated',
        val_data_path = '',
        test_data_path = '',
    )


    return config