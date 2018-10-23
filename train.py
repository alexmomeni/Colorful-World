import tensorflow as tf
from config import Config
from model import GAN

config  = Config()
model = GAN(config)
model.train()