import os 

class Config(object):
    
    def __init__(self, lr_dis=0.0001, lr_gen=0.001, n_epochs=20, batch_size= 33, image_size=128, gpu="0",
                  train_dir='/labs/gevaertlab/data/momena/lfw/', model_dir='model_params/', test_dir='testing_data/', prediction_dir='prediction_data/', predicted_dir='predicted_data/'):

        self.lr_dis = lr_dis 
        self.lr_gen = lr_gen 
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model_dir = model_dir 
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_size = image_size
        self.save_frequency = n_epochs
        self.gpu = gpu
        self.prediction_dir = prediction_dir
        self.predicted_dir = predicted_dir
        self.steps_per_epoch =  len(os.listdir(self.train_dir)) // self.batch_size