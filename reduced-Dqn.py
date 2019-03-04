#!/usr/bin/python
import gym
import numpy as np
import rebin
import tensorflow as tf
import random
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
            self.inputs_ = tf.placeholder(tf.float32, [None,160,160,4], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
             
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            ELU
            """
            # Input is 40x40x1
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 16,
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
                                 filters = 32,
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
            #self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                     filters = 64,
            #                     kernel_size = [3,3],
            #                     strides = [2,2],
            #                     padding = "VALID",
            #                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                     name = "conv3")
#
#            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv2_out)
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 256,
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






def preprocess_frame(state):
    	gray=rgb2gray(state)
    	#cropped_frame=gray[8:-12,4:-12]
    	cropped_frame=gray[22:-28,:]
	#downscale=block_reduce(cropped_frame,block_size=(2,2),func=np.mean)
    	downscale=cropped_frame
        downscale[downscale<.1]=0
    	downscale[downscale>=.1]=1
    
    	#plt.imshow(downscale, interpolation='nearest')
    	#plt.show()
	return downscale

stack_size=4
def stack_frames(stacked_frames,state,is_new_episode):
    frame=preprocess_frame(state)
    
    if is_new_episode:
        
        stacked_frames=deque([np.zeros((160,160),dtype=np.int) for i in range (stack_size)],maxlen=4)
        for i in range(stack_size):
            stacked_frames.append(frame)
    
    	stacked_state=np.stack(stacked_frames,axis=2)
    else:
        stacked_frames.append(frame)

    	stacked_state=np.stack(stacked_frames,axis=2)
    #plt.imshow(frame,interpolation='nearest')
    #plt.show()
    #print(stacked_frames)
    return stacked_state,stacked_frames

def test_environment(number_of_episodes):
    

    #leads to  action every 60/4 of a second

    stacked_frames=deque([np.zeros((160,160),dtype=np.int) for i in range (stack_size)],maxlen=4)
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




def predict_action(explore_start,explore_stop,decar_rate,decay_step,observation,actions):
	##EPSILON GREEDY STRATEGY
	# Choose action a from state s using epsilon greedy.
	## First we randomize a number
	exp_exp_tradeoff = np.random.rand()

	# Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
	explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
	
	if (explore_probability > exp_exp_tradeoff):
	    # Make a random action (exploration)
	    choice = random.randint(1,len(possible_actions))-1
	    action = possible_actions[choice]
	    
	else:
	    # Get action from Q-network (exploitation)
	    # Estimate the Qs values state
	    Qs = sess.run(DQN.output, feed_dict = {DQN.inputs_:observation.reshape((1,160,160,4))})
	    # Take the biggest Q value (= the best action)
	    choice = np.argmax(Qs)
            action=possible_actions[choice]
		    
		    
        return action, explore_probability

######################################################
# Some vague hyper parameters i took to get started ##
######################################################
### MODEL HYPERPARAMETERS
action_size=8
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50            # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 64                # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 0.5            # exploration probability at start
explore_stop = 0.5            # minimum exploration probability 
decay_rate = 0.000000           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4                 # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


######################################################
################### Actual Code running ##############
######################################################
tf.reset_default_graph()
env=gym.make('Phoenix-v0')
DQN=DQNetwork([160,160,4],action_size,learning_rate)
memory=Memory(max_size=memory_size)
saver=tf.train.Saver()
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
print(possible_actions)
stacked_frames=deque([np.zeros((160,160),dtype=np.int) for i in range (stack_size)],maxlen=4)

####################
##Show agent########
####################
show=False
if show:
    print(True)
    with tf.Session() as sess:
        total_test_rewards=[]

        #loadModel
         
        saver.restore(sess,"./models/model.ckpt")
        for episode in range(100):
            total_rewards=0

            observation=env.reset()
            observation,stacked_frames= stack_frames(stacked_frames,observation,True)

            while True:
                
		Qs=sess.run(DQN.output, feed_dict = {DQN.inputs_: observation.reshape(1,160,160,4)})
                choice=np.argmax(Qs)
                
                next_observation,reward,done,_=env.step(choice)
                env.render()
                total_rewards+=reward
                
                if done:
                    print ("Score", total_rewards)
                    total_test_rewards.append(total_rewards)
                    break
                
                next_observation, stacked_frames = stack_frames(stacked_frames, next_observation, False)
                observation = next_observation



else :
    ####################
    #Tensor board setup#
    ####################

    writer = tf.summary.FileWriter("dqn/1")


    tf.summary.scalar("Loss", DQN.loss)

    write_op = tf.summary.merge_all()





    ####################
    #Instantiate Memory#
    ####################

    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            observation = env.reset()
            
            observation, stacked_frames = stack_frames(stacked_frames, observation, True)
            
        # Get the next_state, the rewards, done by taking a random action
        choice=random.randint(1,len(possible_actions))-1
        action = possible_actions[choice]
        
        next_observation, reward, done, _ = env.step(choice)
        
        #env.render()
        
        # Stack the frames
        next_observation, stacked_frames = stack_frames(stacked_frames, next_observation, False)
        
        
        # If the episode is finished (we're dead 3x)
        if done:
            # We finished the episode
            next_observation = np.zeros(state.shape)
            
            # Add experience to memory
            memory.add((observation, action, reward, next_state, done))
            
            # Start a new episode
            observation = env.reset()
            
            # Stack the frames
            observation, stacked_frames = stack_frames(stacked_frames, state, True)
            
        else:
            # Add experience to memory
            memory.add((observation, action, reward, next_observation, done))
            
            # Our new state is now the next_state
            observation  = next_observation

    ###################
    #Memory Instantiated#
    ####################y


    rewards_list=[]
    with tf.Session() as sess:
       
        #Load a model if it exists
        #saver.restore(sess,"./models/model.ckpt")
        #print("Loaded model")
        sess.run(tf.global_variables_initializer())
       
        stacked_frames=deque([np.zeros((160,160),dtype=np.int) for i in range (stack_size)],maxlen=4)
        #Iinitialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0
        
        for episode in range(5000):
            # Set step to 0
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            observation = env.reset()
            
            # Remember that stack frame function also call our preprocess function.
            observation, stacked_frames = stack_frames(stacked_frames, observation, True)



            while step < max_steps:
                    step += 1
                    
                    #Increase decay_step
                    decay_step +=1
                    
                    # Predict the action to take and take it
                    
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, observation,possible_actions)
                   

                    choice=action.tolist().index(1)
                    
                    #Perform the action and get the next_state, reward, and done information
                    next_observation,reward,done,info=env.step(choice)
                    #next_observation, reward, done, info = env.step(action)
                    
                    if episode_render:
                        env.render()
                    
                    # Add the reward to total reward
                    episode_rewards.append(reward)



                    #if the game is finished
                    if done:
                        # The episode ends so no next state
                        
                        stacked_frames=deque([np.zeros((160,160),dtype=np.int) for i in range (stack_size)],maxlen=4)

                        # Set step = max_steps to end the episode
                        step = max_steps

                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)
                        
                        print('Episode: {}'.format(episode),
                                      'Total reward: {}'.format(total_reward),
                                      'Explore P: {:.4f}'.format(explore_probability))
                                    #'Training Loss {:.4f}'.format(loss))


                        # Store transition <st,at,rt+1,st+1> in memory D
                        #memory.add((state, action, reward, next_state, done))

                    else:
                        # Stack the frame of the next_state
                        next_observation, stacked_frames = stack_frames(stacked_frames, next_observation, False)
                    
                        # Add experience to memory
                        #memory.add((state, action, reward, next_state, done))

                        # st+1 is now our current state
                        observation = next_observation

                    ###################
                    #LEARNING PART#####
                    #Experiance Replay#
                    ###################

                    # Obtain random mini-batch from memory
                    batch = memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch]) 
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    # Get Q values for next_state 
                    Qs_next_state = sess.run(DQN.output, feed_dict = {DQN.inputs_: next_states_mb})


                    #################
                    #Update Q values#
                    #################

                    #Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])
                            
                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)
                            


                    targets_mb=np.array([each for each in target_Qs_batch])

                    #print(actions_mb)
                    #print(targets_mb)
                    loss, _ = sess.run([DQN.loss, DQN.optimizer],
                                            feed_dict={DQN.inputs_: states_mb,
                                                       DQN.target_Q: targets_mb,
                                                       DQN.actions_: actions_mb})

                    # Write TF Summaries
                    summary = sess.run(write_op, feed_dict={DQN.inputs_: states_mb,
                                                           DQN.target_Q: targets_mb,
                                                           DQN.actions_: actions_mb})

                    writer.add_summary(summary,episode)
                    writer.flush()

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")



















































