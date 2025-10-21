from config.config import get_config
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from PIL import Image

def forward_model(phase_path, config):
    '''phase object --> Interferogram (QWLSI) image'''
    img_pil = Image.open(phase_path).convert('F')   # convert to float grayscale
    H, W_ = img_pil.height, img_pil.width
    W = np.array(img_pil)                            # numeric array for shift, arithmetic
    x = np.arange(W_)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)


    A = config.simul.A
    d = config.simul.d
    s = config.simul.s
    u0 = config.simul.u0
    v0 = config.simul.v0

    E1 = A * np.exp(1j * (shift(W, (-s/2, 0)) + 2 * np.pi * u0 * X))
    E2 = A * np.exp(1j * (shift(W, (s/2, 0)) - 2 * np.pi * u0 * X))
    E3 = A * np.exp(1j * (shift(W, (0, -s/2)) + 2 * np.pi * v0 * Y))
    E4 = A * np.exp(1j * (shift(W, (0, s/2)) - 2 * np.pi * v0 * Y))
    E_total = E1 + E2 + E3 + E4
    I = np.abs(E_total)**2
    I_norm = (I - I.min()) / (I.max() - I.min())

    return I_norm



def visualize(I_norm):
    plt.imshow(I_norm, cmap='gray')
    plt.title('Simulated Interferogram')
    plt.axis('off')
    plt.show()
