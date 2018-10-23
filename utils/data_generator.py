import numpy as np
import os   
            
class Data_Generator(object):
    
    
    def __init__(self, config, dataset):
        
        self.config = config
        self.dataset = dataset
        self.list_IDs = os.listdir(self.config.train_dir)
   
    
    def generate(self):
        'Generates batches of samples'
        
        while 1:
            indexes = self.__get_exploration_order()
            imax = int(len(indexes)/self.config.batch_size)
            for i in range(imax):
                list_IDs_temp = [self.list_IDs[k] for k in indexes[i*self.config.batch_size:(i+1)*self.config.batch_size]]
                X_color, X_bw = self.__data_generation(list_IDs_temp)
                
                return X_color, X_bw

    def __get_exploration_order(self):
        'Generates order of exploration'
        indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, list_IDs_temp):
        
        X_color, X_bw = self.dataset.convert_to_arrays(list_IDs_temp)
        
        return X_color, X_bw
    
    
  
    
    
    
