import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
from glob import glob
import cv2
from utils.data_generator import Data_Generator
from dataset import Dataset_Color
from PIL import Image


class Generator():
    
        '''
    The pix2pix generator is a U-net, i.e. a variant of an auto-encoder where layers are symmetrically stacked one to the other. 
    It learns a mapping between gray scale to RGB space, conditionally to a grayscale image.
    
    References: 
    https://arxiv.org/pdf/1611.07004.pdf
    https://arxiv.org/pdf/1505.04597.pdf
    '''
    
    def generator(self,c):
    
        with tf.variable_scope('generator'):
            
            self.initializer = tf.truncated_normal_initializer(stddev=0.02)

            #Encoder
            enc0 = slim.conv2d(c,64,[3,3],padding="SAME",
                biases_initializer=None,activation_fn= tf.nn.leaky_relu,
                weights_initializer=self.initializer)      
            enc0 = tf.space_to_depth(enc0,2)

            enc1 = slim.conv2d(enc0,128,[3,3],padding="SAME",
                activation_fn= tf.nn.leaky_relu,normalizer_fn=slim.batch_norm,
                weights_initializer=self.initializer)
            enc1 = tf.space_to_depth(enc1,2)

            enc2 = slim.conv2d(enc1,256,[3,3],padding="SAME",
                normalizer_fn=slim.batch_norm,activation_fn= tf.nn.leaky_relu,
                weights_initializer= self.initializer)
            enc2 = tf.space_to_depth(enc2,2)

            enc3 = slim.conv2d(enc2,512,[3,3],padding="SAME",
                normalizer_fn=slim.batch_norm,activation_fn= tf.nn.leaky_relu,
                weights_initializer=self.initializer)

            enc3 = tf.space_to_depth(enc3,2)

            #Decoder
            gen0 = slim.conv2d(
                enc3,num_outputs=512,kernel_size=[3,3],
                padding="SAME",normalizer_fn=slim.batch_norm,
                activation_fn=tf.nn.elu, weights_initializer=self.initializer)
            gen0 = tf.depth_to_space(gen0,2)

            gen1 = slim.conv2d(
                tf.concat([gen0,enc2],3),num_outputs=256,kernel_size=[3,3],
                padding="SAME",normalizer_fn=slim.batch_norm,
                activation_fn=tf.nn.elu,weights_initializer=self.initializer)
            gen1 = tf.depth_to_space(gen1,2)

            gen2 = slim.conv2d(
                tf.concat([gen1,enc1],3),num_outputs=128,kernel_size=[3,3],
                padding="SAME",normalizer_fn=slim.batch_norm,
                activation_fn=tf.nn.elu,weights_initializer=self.initializer)
            gen2 = tf.depth_to_space(gen2,2)

            gen3 = slim.conv2d(
                tf.concat([gen2,enc0],3),num_outputs=64,kernel_size=[3,3],
                padding="SAME",normalizer_fn=slim.batch_norm,
                activation_fn=tf.nn.elu, weights_initializer=self.initializer)
            gen3 = tf.depth_to_space(gen3,2)

            g_out = slim.conv2d(
                gen3,num_outputs=3,kernel_size=[1,1],padding="SAME",
                biases_initializer=None,activation_fn=tf.nn.tanh,
                weights_initializer=self.initializer)
            return g_out

class Discriminator():
    
    '''
    The discriminator is a classical ConvNet classifier.
    It learns to discriminate fake-colored image vs. originally colored images
    '''
    
    def discriminator(self, c, reuse = False):
        
        with tf.variable_scope('discriminator'):
            
            self.initializer = tf.truncated_normal_initializer(stddev=0.02)

            conv1 = slim.conv2d(c,32,[3,3],padding="SAME",scope='d0',
                biases_initializer=None,activation_fn= tf.nn.leaky_relu,stride=[2,2],
                reuse=reuse,weights_initializer=self.initializer)

            conv2 = slim.conv2d(conv1,64,[3,3],padding="SAME",scope='d1',
                normalizer_fn=slim.batch_norm,activation_fn= tf.nn.leaky_relu,stride=[2,2],
                reuse=reuse,weights_initializer=self.initializer)
            
            conv3 = slim.conv2d(conv2,128,[3,3],padding="SAME",scope='d2',
                normalizer_fn=slim.batch_norm,activation_fn= tf.nn.leaky_relu,stride=[2,2],
                reuse=reuse,weights_initializer=self.initializer)
            
            conv4 = slim.conv2d(conv3,256,[3,3],padding="SAME",scope='d3',
                normalizer_fn=slim.batch_norm,activation_fn= tf.nn.leaky_relu,stride=[2,2],
                reuse=reuse,weights_initializer=self.initializer)
                
            dis_full = slim.fully_connected(slim.flatten(conv4),1024,activation_fn= tf.nn.leaky_relu,scope='dl',
                reuse=reuse, weights_initializer=self.initializer)

            d_out = slim.fully_connected(dis_full,1,activation_fn=tf.nn.sigmoid,scope='do',
                reuse=reuse, weights_initializer=self.initializer)
            
            return d_out


class GAN():
    
    '''
    The Generative Adversarial Network framework we used to train our color generator
    '''
    
    def __init__(self, config):      
        self.config = config
        self.data_init()
        self.model_init()
        
    def data_init(self):
        
        print("\nData init")

        self.dataset = Dataset_Color(self.config)
        self.data_generator = Data_Generator(self.config, self.dataset)

      #  self.prediction_dataset = Dataset_BW(self.config)
        
    def model_init(self):
        
        print("\Model init")
        self.D = Discriminator()
        self.G = Generator()    
        self.lambda_L1 = 100
        
        self.condition_in = tf.placeholder(shape=[None,self.config.image_size,self.config.image_size,1],dtype=tf.float32)
        self.real_in = tf.placeholder(shape=[None,self.config.image_size,self.config.image_size,3],dtype=tf.float32) 
        
        self.Gx = self.G.generator(self.condition_in) 
        self.Dx = self.D.discriminator(self.real_in) 
        self.Dg = self.D.discriminator(self.Gx,reuse=True) 
        
        self.d_loss = -tf.reduce_mean(tf.log(self.Dx) + tf.log(1.- self.Dg))
        self.g_loss = -tf.reduce_mean(tf.log(self.Dg)) + self.lambda_L1 *tf.reduce_mean(tf.abs(self.Gx - self.real_in)) 
        
        self.trainerD = tf.train.AdamOptimizer(learning_rate= self.config.lr_dis)
        self.trainerG = tf.train.AdamOptimizer(learning_rate= self.config.lr_gen)
        self.d_grads = self.trainerD.compute_gradients(self.d_loss,slim.get_variables(scope='discriminator'))
        self.g_grads = self.trainerG.compute_gradients(self.g_loss, slim.get_variables(scope='generator'))

        self.update_D = self.trainerD.apply_gradients(self.d_grads)
        self.update_G = self.trainerG.apply_gradients(self.g_grads)

    
    def train(self):
        
        '''
        Training loop for the GAN
        '''
        
        print('\n\n\n------------ Starting training ------------')

        self.init = tf.global_variables_initializer()        
        self.session_config = tf.ConfigProto()
        self.session_config.gpu_options.visible_device_list = self.config.gpu
        self.session_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.session_config)
        self.session.run(self.init)
        self.var_list = tf.trainable_variables()
        self.saver       = tf.train.Saver(self.var_list)

        
#        self.saver = tf.train.Saver()



        for i in range(self.config.n_epochs):
            
            print ('\n------ Epoch %i ------' % i)

            for j in range(self.config.steps_per_epoch):
                X_color, X_bw = self.data_generator.generate()
                ys = (np.reshape(X_color,[self.config.batch_size,self.config.image_size,self.config.image_size,3])- 0.5) * 2.0
                xs = (np.reshape(X_bw,[self.config.batch_size,self.config.image_size,self.config.image_size,1])- 0.5) * 2.0
                
                _, self.dLoss = self.session.run([self.update_D,self.d_loss],feed_dict={self.real_in: ys, self.condition_in:xs}) 
                _, self.gLoss = self.session.run([self.update_G,self.g_loss],feed_dict={self.real_in:ys, self.condition_in:xs}) 

                if j % 10 == 0:
                    print ("Step: " + str(j) + " Gen Loss: " + str(self.gLoss) + " Disc Loss: " + str(self.dLoss))
                
        
        if (i+1) % self.config.save_frequency == 0 and i != 0:
            if not os.path.exists(self.config.model_dir):
                os.makedirs(self.config.model_dir)
            self.saver.save(self.session,self.config.model_dir+'/model-'+str(i)+'.cptk')
            print ("Saved Model") 
            
    def predict(self):
        
        '''
        Load model weights and generate colored images from B&W images
        '''
        
        if not os.path.exists(self.config.predicted_dir):
            os.makedirs(self.config.predicted_dir)
        
        self.init = tf.global_variables_initializer()
        self.session_config = tf.ConfigProto()
        self.session_config.gpu_options.visible_device_list = self.config.gpu
        self.session_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.session_config)
        self.session.run(self.init)
        self.var_list = tf.trainable_variables()
        self.saver = tf.train.Saver(self.var_list)
        
        ckpt = tf.train.get_checkpoint_state(self.config.model_dir)
        self.saver.restore(self.session,ckpt.model_checkpoint_path)
       

        generated_frames = []
        samples = os.listdir(self.config.prediction_dir)
        print(samples)
        _, X_bw = self.dataset.convert_to_arrays(samples, training = False) 


        xs = (np.reshape(X_bw,[X_bw.shape[0],X_bw.shape[1],X_bw.shape[2],1]) - 0.5) * 2.0

        self.sample_G = self.session.run(self.Gx,feed_dict={self.condition_in:xs})    
        generated_frames.append(self.sample_G)
        generated_frames = np.vstack(generated_frames)

        for i in range(len(generated_frames)):
            red = generated_frames[i][:,:,2].copy()  
            blue = generated_frames[i][:,:,1].copy()
            green = generated_frames[i][:,:,0].copy()

            generated_frames[i][:,:,0] = red
            generated_frames[i][:,:,1] = blue
            generated_frames[i][:,:,2] = green

            im = Image.fromarray((((generated_frames[i])  /2 + 0.5)* 256).astype('uint8'))
            im.save(self.config.predicted_dir + '%i.jpg' %i)
                
          
        
   