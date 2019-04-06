#!/usr/bin/env python
# coding: utf-8


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



def update_target_graph():
    
    # Get the parameters of our DQNNetwork
    
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []
    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

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
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 500000        # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 32             

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.1            # minimum exploration probability 
decay_rate = 0.0004            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 100000          # Number of experiences the Memory can keep

training = False
episode_render = False


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name=name
        
        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, 80,80,4], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 8], name="actions_")
            
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
            
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                      
                                  activation = tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")
            
            
            self.output = tf.layers.dense(inputs = self.fc, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = 8, 
                                        activation=None)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            #self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.loss= tf.losses.huber_loss(self.target_Q,self.Q) 
            
            
            """Change to adam"""
            self.optimizer=tf.train.AdamOptimizer(0.001).minimize(self.loss)
            
            #self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)




# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork1 = DQNetwork(state_size, action_size, learning_rate, "DQNetwork")

TargetNetwork = DQNetwork(state_size, action_size, learning_rate, "TargetNetwork")

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]


# Instantiate memory
memory = Memory(max_size = memory_size)

# Render the environment
game.reset()
if training==True:
    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            # First we need a state
            state = game.reset()
            state, stacked_frames = stack_frames(state,state,state,state)
        
        # Random action
        choice=random.randint(1,len(possible_actions))-1
        action= possible_actions[choice]

        next_state1, reward1,done1,_ = game.step(choice)
        next_state2, reward2,done2,_ = game.step(choice)
        next_state3, reward3,done3,_ = game.step(choice)
        next_state4, reward4,done4,_ = game.step(choice)
        reward= reward1+reward2+reward3+reward4;
        next_state,stacked_frames= stack_frames(next_state1,next_state2,next_state3,next_state4)
        done= done1 or done2 or done3 or done4

        if done:
            #Finished the episode

            next_state = game.reset()
            next_state,stacked_frames=stack_frames(next_state,next_state,next_state,next_state)
            # Add experience to memory
            memory.add((state, action, reward, next_state, done))
            
            # Start a new episode
            state=game.reset()
             
            # Stack the frames
            
            state, stacked_frames = stack_frames(state,state,state,state)
        else:
            # Get the next state
            #next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
            # Add experience to memory
            memory.add((state, action, reward, next_state, done))
            
            # Our state is now the next_state
            state = next_state
    print("Memory Ready")


# Setup TensorBoard Writer
writer = tf.summary.FileWriter("tensorboard/dqn/2")

## Losses
tf.summary.scalar("Loss", DQNetwork1.loss)

write_op = tf.summary.merge_all()



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        #exploration
        action = random.choice(possible_actions)
        
    else:
        # exploitation
        # Estimate the Qs values 
        Qs = sess.run(DQNetwork1.output, feed_dict = {DQNetwork1.inputs_: state.reshape((1, 80,80,4))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability


saver = tf.train.Saver()

if training == True:
    print("Training Started")
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
                
        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0

        game.reset()

        update_target=update_target_graph()
        sess.run(update_target)


        for episode in range(total_episodes):
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            state=game.reset()
            
            state, stacked_frames = stack_frames(state,state,state,state)

            while step < max_steps:
                step += 1
                
                decay_step +=1
                
                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
                choice=action.tolist().index(1)


                if(episode_render):
                    game.render()
                
                next_state1,reward1,done1,_=game.step(choice)
                next_state2,reward2,done2,_=game.step(choice)
                next_state3,reward3,done3,_=game.step(choice)
                next_state4,reward4,done4,_=game.step(choice)
                
                reward=reward1+reward2+reward3+reward4
                done=done1 or done2 or done3 or done4
                

                next_state,stacked_frames=stack_frames(next_state1,next_state2,next_state3,next_state4)
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = game.reset()
                    next_state, stacked_frames = stack_frames(next_state,next_state,next_state,next_state)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                              'Reward: {}'.format(total_reward),
                              'Loss: {}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((state, action, reward, next_state, done))

                else:
                    
                    # Stack the frame of the next_state
                    #next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    

                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))
                    
                    # st+1 is now our current state
                    state = next_state


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                #print(np.shape(np.array([each[3] for each in batch],ndmin=3)))
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []
                 # Get Q values for next_state 
                Qs_next_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.inputs_: next_states_mb})
                
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, other = sess.run([DQNetwork1.loss, DQNetwork1.optimizer],
                                    feed_dict={DQNetwork1.inputs_: states_mb,
                                               DQNetwork1.target_Q: targets_mb,
                                               DQNetwork1.actions_: actions_mb})
                
                summary = sess.run(write_op, feed_dict={DQNetwork1.inputs_: states_mb,
                                                   DQNetwork1.target_Q: targets_mb,
                                                   DQNetwork1.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            # Save model every 5 episodes
            if episode % 50 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                update_target=update_target_graph()
                sess.run(update_target)
                
                
                print("Model Saved")


import time
with tf.Session() as sess:
    
    game= gym.make('PhoenixDeterministic-v4')
    
    totalScore = 0
    
   
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    for i in range(10):
        done=False
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
            
        print("Score: ", reward,steps)
        totalScore += reward
    print("TOTAL_SCORE", totalScore/100.0)
    game.close()

