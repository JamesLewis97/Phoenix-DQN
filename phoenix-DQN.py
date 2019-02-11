#!/usr/bin/python
import gym
import numpy as np
import rebin
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from skimage import transform 
from skimage.color import rgb2gray
from collections import deque


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None,40,40,4], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")            
            
            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")
            
            self.output = tf.layers.dense(inputs = self.fc, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)
            

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)




def preprocess_frame(state):
    gray=rgb2gray(state)
    #cropped_frame=gray[8:-12,4:-12]
    cropped_frame=gray[22:-28,:]
    downscale=block_reduce(cropped_frame,block_size=(4,4),func=np.mean)
    downscale[downscale<.1]=0
    downscale[downscale>=.1]=1
    return downscale

stack_size=4
def stack_frames(stacked_frames,state,is_new_episode):
    frame=preprocess_frame(state)
    
    if is_new_episode:
        stacked_frames=deque([np.zeros((40,40),dtype=np.int) for i in range (stack_size)],maxlen=4)
        for i in range(stack_size):
            stacked_frames.append(frame)
    
    
    else:
        stacked_frames.append(frame)

    #plt.imshow(frame,interpolation='nearest')
    #plt.show()
    #print(stacked_frames)
    return stacked_frames

def test_environment(number_of_episodes):
    

    #leads to  action every 60/4 of a second
    stacked_frames=deque([np.zeros((40,40),dtype=np.int) for i in range (stack_size)],maxlen=4)

    for i_episode in range(number_of_episodes):
        observation=env.reset()
        totalRew=0
        is_new_episode=True
        for t in range(100000):
            #env.render()
            action=env.action_space.sample()
            observation,reward,done,info=env.step(action)
                   
            #test=tf.Session.run(DQN.output,feed_dict={DQN.inputs_:observation}) 
            stacked_frames=stack_frames(stacked_frames,observation,is_new_episode)
            is_new_episode=False
            
            plt.imshow(stacked_frames[0],interpolation='nearest')
            plt.show()
            observation=np.around(observation,2)
            totalRew+=reward
            
            if done:
                break
        print(i_episode,totalRew)
        #averageScores.append(totalRew/500)
    
    env.close();


######################################################
################### Actual Code running ##############
######################################################

env=gym.make('Phoenix-v0')
DQN=DQNetwork([40,40,4],env.action_space.n,0.1)
saver=tf.train.Saver()


test_environment(500)



