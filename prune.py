#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import gym
import random                # Handling random number generation
import time                  # Handling time calculation
import sys
import cv2
from skimage import transform# Help us to preprocess the frames
from skimage import color
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs
from skimage.measure import block_reduce
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore') 



game= gym.make('PhoenixDeterministic-v4')
possible_actions = np.array(np.identity(game.action_space.n,dtype=int).tolist())


def preprocess_frame(frame):
    #grayscale
    gray=color.rgb2gray(frame) 
    cropped_frame = gray[22:-28,:]
    #downsample
    downscale=block_reduce(cropped_frame,block_size=(2,2),func=np.mean)
    #convert to binary
    downscale[downscale<.1]=0
    downscale[downscale>=.1]=255
    binaryOutput=downscale.astype(np.bool)
    processed_frame=downscale.astype(np.uint8)
    
    return processed_frame



stack_size = 4 # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((80,80), dtype=np.int) for i in range(stack_size)], maxlen=4) 

def stack_frames(state1,state2,state3,state4):
    # Preprocess frame
    frame1 = preprocess_frame(state1)
    frame2 = preprocess_frame(state2)
    frame3 = preprocess_frame(state3)
    frame4 = preprocess_frame(state4)
    #plt.imshow(frame4)
    #plt.show()
    # Clear our stacked_frames
    stacked_frames = deque([np.zeros((80,80), dtype=np.int) for i in range(stack_size)], maxlen=4)
    
    # Because we're in a new episode, copy the same frame 4x
    stacked_frames.append(frame1)
    stacked_frames.append(frame2)
    stacked_frames.append(frame3)
    stacked_frames.append(frame4)
    
    # Stack the frames
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames



### MODEL HYPERPARAMETERS
state_size = [80,80,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
action_size = game.action_space.n              # 3 possible actions: left, right, shoot
learning_rate =  0.00015      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 500000        # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 32             

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.1            # minimum exploration probability 
decay_rate = 0.00004            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 50000          # Number of experiences the Memory can keep

max_tau=20000



training = False
episode_render = False


class DDQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name=name
        
        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, 80,80,4], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 8], name="actions_")
            self.ISWeights_=tf.placeholder(tf.float32,[None,1],name='IS_weights')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            """
            # Input is 80x80x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 16,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                         activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            
            
            """
            Second convnet:
            CNN
            """
            

            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 32,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            
            
            self.flatten = tf.layers.flatten(self.conv2_out)
            
            #Calculate V(s)
            self.value_fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="value_fc")

            self.value= tf.layers.dense(inputs= self.value_fc,
                                    units=1,
                                    activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="value")
            
            #Calculates A(s,a)
            self.advantage_fc=tf.layers.dense(inputs=self.flatten,
                                    units=512,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="advantage_fc")
            

            self.advantage= tf.layers.dense(inputs=self.advantage_fc,
                    units=self.action_size,
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="advantages")

            #agregate
            #Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))

            self.output=self.value+tf.subtract(self.advantage,tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            #Predicted Q value
            self.Q=tf.reduce_sum(tf.multiply(self.output,self.actions_),axis=1)

            self.absolute_errors=tf.abs(self.target_Q-self.Q)



  
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            #self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.loss= tf.losses.huber_loss(self.target_Q,self.Q) 
            
            
            """Change to adam"""
            self.optimizer=tf.train.AdamOptimizer(0.0005).minimize(self.loss)
            
            #self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)




# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork1 = DDQNetwork(state_size, action_size, learning_rate, "DQNetwork")





# Render the environment
game.reset()

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("tensorboard/dqn/2")

## Losses
tf.summary.scalar("Loss", DQNetwork1.loss)

write_op = tf.summary.merge_all()






saver = tf.train.Saver()
import time
import heapq
import math
np.set_printoptions(threshold=sys.maxsize)
pruning_factor=0.2
with tf.Session() as sess:

    game= gym.make('PhoenixDeterministic-v4')

    totalScore = 0


    #saver.restore(sess, "./models/model.ckpt")
    # Load the model
    
    for i in range(50,95):

        saver.restore(sess, "modelsDDQN/models{}/models/model.ckpt".format(i))
        done=False

        conv1=[v for v in tf.trainable_variables()if v.name=="DQNetwork/conv1/kernel:0"][0]
        conv2=[v for v in tf.trainable_variables()if v.name=="DQNetwork/conv2/kernel:0"][0]
        #print(vars)
        
        conv1_val=sess.run(conv1)
        conv2_val=sess.run(conv2)
        
        
        conv1_abs_array=np.absolute(conv1_val)
        conv2_abs_array=np.absolute(conv2_val)

                    
        conv1_masknp=np.ones(np.shape(conv1_abs_array))
        conv2_masknp=np.ones(np.shape(conv2_abs_array))
        
        
        conv1_pruned_elements=math.floor(conv1_abs_array.size*pruning_factor)
        conv2_pruned_elements=math.floor(conv2_abs_array.size*pruning_factor)
        
        for count in range(1,int(conv1_pruned_elements)):
            ind=np.unravel_index(np.argmin(conv1_abs_array,axis=None),conv1_abs_array.shape)
            
            conv1_abs_array[ind]=99
            conv1_masknp[ind]=0


        
	for count in range(1,int(conv2_pruned_elements)):
            ind=np.unravel_index(np.argmin(conv2_abs_array,axis=None),conv2_abs_array.shape)

            conv2_abs_array[ind]=99
            conv2_masknp[ind]=0
        

        conv1_mask=tf.constant(conv1_masknp)
        conv2_mask=tf.constant(conv2_masknp)
       


 	conv1_val=tf.multiply(conv1_mask,conv1_val)
 	conv2_val=tf.multiply(conv2_mask,conv2_val)
        
        
	
        
	conv1_finalnp=sess.run(conv1_val)
	conv2_finalnp=sess.run(conv2_val)

        
	conv1_final=tf.constant(conv1_finalnp,dtype=tf.float32)
	conv2_final=tf.constant(conv2_finalnp,dtype=tf.float32)


        g=tf.get_default_graph()
        
	conv1_tensor=g.get_tensor_by_name('DQNetwork/conv1/kernel:0')
	sess.run(tf.assign(conv1_tensor,conv1_final))

	conv2_tensor=g.get_tensor_by_name('DQNetwork/conv2/kernel:0')
	sess.run(tf.assign(conv2_tensor,conv2_final))

        #print(test)
        
         
        #print("var: {}, value: {}".format(vars.name, vars_val))
        


        state=game.reset()
        reward=0
        state, stacked_frames = stack_frames(state,state,state,state)
        steps=0
        while not done:
            Qs = sess.run(DQNetwork1.output, feed_dict = {DQNetwork1.inputs_: state.reshape((1, 80,80,4))})
            action = np.argmax(Qs)
            game.render()
            #time.sleep(0.10)
            state1,reward1,done1,_=game.step(action)
            state2,reward2,done2,_=game.step(action)
            state3,reward3,done3,_=game.step(action)
            state4,reward4,done4,_=game.step(action)

            reward=reward+reward1+reward2+reward3+reward4
            done= done1 or done2 or done3 or done4

            state,stacked_Frames=stack_frames(state1,state2,state3,state4)
            steps=steps+1
            if steps>1000:
                done=True
        save_path = saver.save(sess, "./models/model.ckpt")
        print(i,"Score: ", reward,steps)
        totalScore += reward
    print("TOTAL_SCORE", totalScore/45.0)
    game.close()
