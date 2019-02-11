#!/usr/bin/python
import gym
import numpy as np
import rebin
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from skimage import transform 
from skimage.color import rgb2gray
from collections import deque


stack_size=4


def preprocess_frame(state):
    gray=rgb2gray(state)
    #cropped_frame=gray[8:-12,4:-12]
    cropped_frame=gray[22:-28,:]
    downscale=block_reduce(cropped_frame,block_size=(4,4),func=np.mean)
    downscale[downscale<.1]=0
    downscale[downscale>=.1]=1
    return downscale

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
    env=gym.make('Phoenix-v0')

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
           
            stacked_frames=stack_frames(stacked_frames,observation,is_new_episode)
            is_new_episode=False
            
            #plt.imshow(downscale,interpolation='nearest')
            #plt.show()
            observation=np.around(observation,2)
            totalRew+=reward
            
            
            if done:
                if(totalRew>4000): 
                    plt.imshow(stacked_frames.pop(),interpolation='nearest')    
                    
                    plt.show()
                    plt.imshow(stacked_frames.pop(),interpolation='nearest')    
                    plt.show()
                    plt.imshow(stacked_frames.pop(),interpolation='nearest')    
                    plt.show()
                    plt.imshow(stacked_frames.pop(),interpolation='nearest')    
                    plt.show()
                    #print("Episode finished after {} timesteps".format(t+1))
                break
        print(i_episode,totalRew)
        #averageScores.append(totalRew/500)
    
    env.close();


######################################################
################### Actual Code running ##############
######################################################

test_environment(500)

