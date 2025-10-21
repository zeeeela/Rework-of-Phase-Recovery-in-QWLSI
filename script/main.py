import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset.qwlsi_simulation import forward_model, visualize
from config.config import get_config, d

phase_img = ''

config = get_config()

img = forward_model(phase_img, config)


