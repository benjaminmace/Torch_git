import torch
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from ALADDIN_WGAN_model import Discriminator, Generator, initialize_weights
import torchvision

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 1
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

checkpoint = torch.load('WGAN_GP.tar')

gen.load_state_dict(checkpoint['gen_state_dict'])

opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])

gen.eval()

fixed_noise = torch.randn(1, Z_DIM, 1, 1).to(device)

real = gen(fixed_noise)

img_grid_real = torchvision.utils.save_image(real, 'real.jpeg')

print(img_grid_real)