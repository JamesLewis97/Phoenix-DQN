#!/usr/bin/env python
# coding: utf-8

# This is the final version

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



training = True
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

TargetNetwork = DDQNetwork(state_size, action_size, learning_rate, "TargetNetwork")


class SumTree(object):

    data_pointer=0

    def __init__(self,capacity):
        self.capacity=capacity
        #Generate the tree
        self.tree=np.zeros(2*capacity-1)
        #Contains experiances
        self.data=np.zeros(capacity,dtype=object)

    #adds priority
    def add(self,priority,data):
        tree_index = self.data_pointer + self.capacity -1

        #update data
        self.data[self.data_pointer]=data

        #update leaf
        self.update(tree_index,priority)

        #add 1 to data pointer
        self.data_pointer+=1 

        #if we overflow
        if self.data_pointer>= self.capacity:
            self.data_pointer=0

    def update(self,tree_index,priority):
        #Change = new prio - old prio
        change=priority-self.tree[tree_index]
        self.tree[tree_index] =priority

        #propagete
        while tree_index!=0:
                tree_index = (tree_index-1)//2
                self.tree[tree_index]+=change

    def get_leaf(self,v):
        parent_index=0

        while True:
            left_child_index=2*parent_index+1
            right_child_index=left_child_index+1

            if left_child_index>= len(self.tree):
                leaf_index=parent_index
                break
            else: #downward search

                if v <=self.tree[left_child_index]:
                    parent_index=left_child_index

                else:
                    v-=self.tree[left_child_index]
                    parent_index=right_child_index
        data_index= leaf_index - self.capacity +1 

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]

               

        


class Memory():
    PER_e=0.01 #avoids 0 probability

    PER_a =0.6 # Makes a trade between only exp with high prio and random

    PER_b=0.4 #importance- sampling, from initial value to 1

    PER_b_increment_per_sampling =0.001

    absolute_error_upper = 1. #clip avs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self,experiance):
        max_priority=np.max(self.tree.tree[-self.tree.capacity:])

        # if max =0 fix the min

        if max_priority==0:
            max_priority= self.absolute_error_upper
        self.tree.add(max_priority,experiance)


    def sample(self,n):
        memory_b=[]

        b_idx,b_ISWeights = np.empty((n,), dtype=np.int32),np.empty((n,1),dtype=np.float32)

        priority_segment=self.tree.total_priority / n

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        #max weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight= (p_min * n) ** (-self.PER_b)

        for i in range(n):

            a,b = priority_segment * i , priority_segment * (i+1)
            value= np.random.uniform(a,b)

            index,priority,data=self.tree.get_leaf(value)

            #P(j)
            sampling_probabilities= priority / self.tree.total_priority

            b_ISWeights[i,0] = np.power(n*sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i]=index

            experience= [data]

            memory_b.append(experience)

        return b_idx, memory_b,b_ISWeights


    def batch_update(self,tree_idx,abs_errors):
        abs_errors+= self.PER_e
        clipped_errors= np.minimum(abs_errors, self.absolute_error_upper)
        ps=np.power(clipped_errors, self.PER_a)

        for ti,p in zip (tree_idx, ps):
            self.tree.update(ti,p)


    










# Instantiate memory
memory = Memory(memory_size)

# Render the environment
game.reset()

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
        experience=state,action,reward,next_state,done  
        memory.store(experience)
        
        # Start a new episode
        state=game.reset()
         
        # Stack the frames
        state, stacked_frames = stack_frames(state,state,state,state)
        
    else:
        # Get the next state
        #next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        
        # Add experience to memory
        experience=state, action, reward, next_state, done
        memory.store(experience)
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
        
        tau=0

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
                tau+=1    
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

                    experience=state, action, reward, next_state, done
                    memory.store(experience)

                else:
                    
                    # Stack the frame of the next_state
                    #next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    

                    # Add experience to memory
                    experience=state, action, reward, next_state, done
                    memory.store(experience)
                    
                    # st+1 is now our current state
                    state = next_state


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                tree_idx,batch,ISWeights_mb=memory.sample(batch_size)
                #print(np.shape(np.array([each[3] for each in batch],ndmin=3)))
                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch]) 
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []
                 # Get Q values for next_state 
                q_next_state = sess.run(DQNetwork1.output, feed_dict = {DQNetwork1.inputs_: next_states_mb})
                
                q_target_next_state=sess.run(TargetNetwork.output,feed_dict={TargetNetwork.inputs_: next_states_mb})
                
                
                
                
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    
                    #action'

                    action=np.argmax(q_next_state[i])
                    
                    
                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]

                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                _, loss, absolute_errors = sess.run([DQNetwork1.optimizer,DQNetwork1.loss, DQNetwork1.absolute_errors],
                        feed_dict={DQNetwork1.inputs_:states_mb,
                                    DQNetwork1.target_Q:targets_mb,
                                    DQNetwork1.actions_: actions_mb,
                                    DQNetwork1.ISWeights_:ISWeights_mb})
               

                memory.batch_update(tree_idx,absolute_errors)

                summary = sess.run(write_op, feed_dict={DQNetwork1.inputs_: states_mb,
                                                   DQNetwork1.target_Q: targets_mb,
                                                   DQNetwork1.actions_: actions_mb,
                                                   DQNetwork1.ISWeights_:ISWeights_mb})

                writer.add_summary(summary, episode)
                writer.flush()

                if tau > max_tau:
                    update_target=update_target_graph()
                    sess.run(update_target)
                    tau=0
                    print("Model updated")

            # Save model every 5 episodes
            if episode % 50 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                update_target=update_target_graph()
                sess.run(update_target)
                
                
                print("Model Saved")



with tf.Session() as sess:
    game = gym.make('PheonixDeterministic-v4')
    
    
    totalScore = 0
    
   
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(10):
        
        game.new_episode()
        while not game.is_episode_finished():
            frame = game.get_state().screen_buffer
            state = stack_frames(stacked_frames, frame)
            Qs = sess.run(DQNetwork1.output, feed_dict = {DQNetwork1.inputs_: state.reshape((1, 80,80,4))})
            action = np.argmax(Qs)
            action = possible_actions[int(action)]
            game.make_action(action)        
            score = game.get_total_reward()
        print("Score: ", score)
        totalScore += score
    print("TOTAL_SCORE", totalScore/100.0)
    game.close()

