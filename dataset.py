import numpy as np
import os
import cv2
from tqdm import tqdm


class Dataset_Color():
    '''
    Data pipeline of colored images to train the model
    '''
    
 
    def __init__(self, config):
        self.config = config  
        self.train_dir = self.config.train_dir
        self.prediction_dir = self.config.prediction_dir
    
    def convert_to_arrays(self, samples, training = True):
        '''
        Convert RGB images to arrays, resize and create the B&W version
        '''
        
        images = []
        images_bw = []
        
        for sample in samples:
            if training == True: 
                file = self.train_dir + sample
            else: 
                file = self.prediction_dir + sample
                
            try:    
                img = cv2.imread(file)
                img = cv2.resize(img,(self.config.image_size,self.config.image_size))
                bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img)
                images_bw.append(bw)
            except:
                pass
          

        X_color = np.array(images)
        X_bw = np.array(images_bw)
        X_color = X_color / 256
        X_bw = X_bw / 256
        
        return X_color, X_bw
  