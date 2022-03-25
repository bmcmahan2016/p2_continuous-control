import numpy as np
import random
import copy
from collections import namedtuple, deque
from os import mkdir
import datetime
from config import CONFIG
import matplotlib.pyplot as plt

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = CONFIG['BUFFER_SIZE']  # replay buffer size
BATCH_SIZE = CONFIG['BATCH_SIZE']        # minibatch size
GAMMA = CONFIG['GAMMA']            # discount factor
TAU = CONFIG['TAU']              # for soft update of target parameters
LR_ACTOR = CONFIG['LR_ACTOR']         # learning rate of the actor 
LR_CRITIC = CONFIG['LR_CRITIC']        # learning rate of the critic
WEIGHT_DECAY = CONFIG['WEIGHT_DECAY']        # L2 weight decay
UPDATE_EVERY = CONFIG['UPDATE_EVERY']
UPDATES_PER_STEP = CONFIG['UPDATES_PER_STEP']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.t_step = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units=CONFIG['fc1_units'], fc2_units=CONFIG['fc2_units']).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units=CONFIG['fc1_units'], fc2_units=CONFIG['fc2_units']).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, fcs1_units=CONFIG['fc1_units'], fc2_units=CONFIG['fc2_units']).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fcs1_units=CONFIG['fc1_units'], fc2_units=CONFIG['fc2_units']).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # initialize target and local networks with same weights
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(local_param.data)

        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(local_param.data)

        # Replay memory
        self.memory = ReplayBuffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # initializes a logginig directory and saves config file to it
        self.log_dir = 'DDPG_training_logs/' + CONFIG['desc'] + '_' + datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + '/'
        mkdir(self.log_dir)
        with open(self.log_dir + "config.txt", 'w') as config_file:
            config_file.write("TRAINING HYPER-PARAMETERS\n\n")
            for hyper_param, value in CONFIG.items():
                config_file.write('%s : %s\n' % (hyper_param, value))
        config_file.close()

    def load_weights(self,  fpath, model_num):
        '''loads weights from existing model'''
        self.actor_local.load_state_dict(torch.load(fpath + '/ddpg_actor_local_' + str(model_num)))
        self.actor_target.load_state_dict(torch.load(fpath + '/ddpg_actor_target_' + str(model_num)))
        self.critic_local.load_state_dict(torch.load(fpath + '/ddpg_critic_local_' + str(model_num)))
        self.critic_target.load_state_dict(torch.load(fpath + '/ddpg_critic_target_' + str(model_num)))

    def save_actor_critic(self, episode_num=0):
        '''saves the actor and critic models'''
        torch.save(self.actor_local.state_dict(), self.log_dir + 'ddpg_actor_local_' + str(episode_num))
        torch.save(self.actor_target.state_dict(), self.log_dir + 'ddpg_actor_target_' + str(episode_num))
        torch.save(self.critic_local.state_dict(), self.log_dir + 'ddpg_critic_local_' + str(episode_num))
        torch.save(self.critic_target.state_dict(), self.log_dir + 'ddpg_critic_target_' + str(episode_num))
    
    def save_training_run(self, scores, episode_num):
        '''
        plots the learning curve for this training run and logs it to file
        '''
        # generate learning curve
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), np.mean(scores, axis=-1))
        plt.ylabel('Score')
        plt.xlabel('Training Episode')
        plt.savefig(self.log_dir + 'learning_curve')

        # save the final model
        self.save_actor_critic(episode_num=episode_num)

        # save the scores array
        with open(self.log_dir + 'scores.npy', 'wb') as f:
            np.save(f, np.array(scores))
        f.close()

        plt.show()
    
    def step(self, state, action, reward, next_state, next_action, next_reward, next_next_state, next_next_action, next_next_reward, next_next_next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # compute the TD-error for these experience tuples
        '''self.actor_target.eval()
        self.actor_local.eval()
        self.critic_local.eval()
        with torch.no_grad():
            # each agent takes a next action, so there should be 20 actions in R4
            actions_next = self.actor_target(torch.from_numpy(next_next_next_state).float().cuda()) # (20, 4)
            # the predicted Q-value using the last state and next action
            # should be 20 values since there are 20 agents
            Q_target_next_next = self.critic_target(torch.from_numpy(next_next_next_state).float().cuda(), actions_next).cpu().numpy()
            # Compute Q targets for current states (y_i) using the bellman operator
            # this should still have 20 values since there are 20 agents
            Q_targets = np.array(reward).reshape(-1,1) + GAMMA * ( np.array(next_reward).reshape(-1,1) + GAMMA * (np.array(next_next_reward).reshape(-1,1) + GAMMA * Q_target_next_next*(1-np.array(done).reshape(-1,1))))
            Q_expected = self.critic_local(torch.from_numpy(state).float().cuda(), torch.from_numpy(action).float().cuda()).cpu().numpy()
            td_errors = np.abs(Q_targets - Q_expected)
        self.actor_target.train()
        self.actor_local.train()
        self.critic_local.train()'''
        td_errors = np.ones((20, 1))
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, next_action, next_reward, next_next_state, next_next_action, next_next_reward, next_next_next_state, done, td_errors)
        self.t_step += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.t_step %UPDATE_EVERY == 0:
            for _ in range(UPDATES_PER_STEP) :
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True, episode_num=0):
        """Returns actions  for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()  
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise_process = np.exp(-episode_num/100.0) * np.random.randn(1, 4)
            action += noise_process
        return np.clip(action, -1, 1)

    def reset(self):
        pass  # was used to reset noise which I removed

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, next_actions, next_rewards, next_next_states, next_next_actions, next_next_rewards, next_next_next_states, dones, probabilities = experiences

        
       # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        # must normalize updates with the following
        # 1/replay_buffer_size * 1/sampling_proabiblility
        importance_sampling = (1./BUFFER_SIZE) * (1./probabilities)
        importance_sampling = torch.tensor(importance_sampling).to(device)
        importance_sampling = importance_sampling / torch.max(importance_sampling)
        importance_sampling = importance_sampling.view(-1,1)

        actions_next = self.actor_target(next_next_next_states)

        # construct the target distribution
        Q_target_next_next = self.critic_target(next_next_next_states, actions_next)  # (batch_size, 2)
        mean_targ = Q_target_next_next[:, 0].view(-1,1)           # (batch_size, 1)
        var_targ = Q_target_next_next[:, 1].view(-1,1)            # (batch_size, 1)
        var_targ = torch.maximum(0.1*torch.ones_like(var_targ), var_targ)
        #print("mean_targ shape:", mean_targ.shape)  # should still be (batch_size, 1)
        #print("var_targ shape:", var_targ.shape)    # should still be (batch_size, 1)
        
        # Compute mean of target distribution
        mean_targ = rewards + gamma * ( next_rewards + gamma * (next_next_rewards + gamma * mean_targ*(1-dones)))
        var_targ = ((gamma*gamma*gamma)**2)*var_targ

        #print("mean_targ shape:", mean_targ.shape)  # should still be (batch_size, 1)
        #print("var_targ shape:", var_targ.shape)    # should still be (batch_size, 1)

        # now compute the critic distribution on the current state
        Q_expected = self.critic_local(states, actions)
        mean = Q_expected[:, 0].view(-1,1)
        var = torch.maximum(0.1*torch.ones_like(Q_expected[:,1].view(-1,1)), Q_expected[:, 1].view(-1,1))

        #dud = Q_targets[2]   # sanity check this line should throw an error
        #print("mean shape:", mean.shape)  # should be (batch_size, 1)
        #print("var shape:", var.shape)    # should be (batch_size, 1)

        # now we draw many random samples from the target distribution for each experience tuple
        n_samples = 1000
        z_targs = (var_targ)*torch.randn(BATCH_SIZE, n_samples).to(device) + mean_targ 
        #print("z_targs shape:", z_targs.shape)    # should be (batch_size, 10)
        
        # now we compute the KL-divergence between the target and critic distributions
        exp = -(z_targs - mean)**2 / (2*var**2)                            # (batch_size, n_samples)
        #print("var:", var)
        #print("min exp:", torch.min(exp))
        #print("max exp:", torch.max(exp))
        p = torch.exp(exp) / torch.sqrt(2*torch.pi*var**2)   # (batch_size, n_samples)
        #print("min sampled probs:", torch.min(p))
        #print("max sampled probs:", torch.max(p))
        log_p = torch.sum(torch.log(p), axis=-1) / n_samples            # (batch_size, 1)
        #print("min log(p)", torch.min(log_p))
        #print("max log(p)", torch.max(log_p))
        
        
        
        critic_loss = -log_p                         # (batch_size, 1)  -- add back importance sampling
        #print("shape of pre-scalar critic loss:", critic_loss.shape)
        critic_loss = torch.sum(critic_loss) / BATCH_SIZE               # scalar
        #print("critic loss:", critic_loss)
        #print("\n\n")


        #Q_targets = rewards + gamma * ( next_rewards + gamma * (next_next_rewards + gamma * Q_target_next_next*(1-dones)))
        #Q_targets = importance_sampling * Q_targets
        # Compute critic loss
        #Q_expected = importance_sampling * self.critic_local(states, actions)
        #print("shape of Q_expected:", Q_expected.shape)  # should be (1024,)
        #print("shape of Q_targets:", Q_targets.shape)    # should be (1024,)
        #critic_loss = F.mse_loss(Q_expected, Q_targets)
        #print("shape of critic loss:", critic_loss.shape)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()


        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred)[:,0].mean()  # expected value over all replay tuples
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # --------------------- update replay priorities  --------------------- #
        #print("Q_targets", Q_targets.shape)
        #print("Q_expected", Q_expected.shape)
        #td_errors = torch.abs(mean.detach() - mean_targ.detach())  # (batch_size, 1)
        #self.memory.update_priorities(td_errors)

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.state_size=state_size
        # buffer size must be a multiple of the number of agents
        self.buffer_size=int(buffer_size - buffer_size %20)    # how many samples to keep in buffer
        self.buffer_ix=0                # points to the end of the buffer
        self.buffer_len=0              # number of elements in buffer
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.batch_ixs = None           # holds the indices of the last batch

        # initialize the replay buffer
        self.create_buffers()

    def create_buffers(self):
        '''creates replay buffer on GPU'''
        self.states = torch.zeros((self.buffer_size, self.state_size)).to(device)
        # next 3 actions
        self.actions = torch.zeros((self.buffer_size, self.action_size)).to(device)
        self.next_actions = torch.zeros((self.buffer_size, self.action_size)).to(device)
        self.next_next_actions = torch.zeros((self.buffer_size, self.action_size)).to(device)
        # next 3 rewards
        self.rewards = torch.zeros((self.buffer_size,1)).to(device)
        self.next_rewards = torch.zeros((self.buffer_size,1)).to(device)
        self.next_next_rewards = torch.zeros((self.buffer_size,1)).to(device)
        # 3 state look ahead
        self.next_states = torch.zeros((self.buffer_size, self.state_size)).to(device)
        self.next_next_states = torch.zeros((self.buffer_size, self.state_size)).to(device)
        self.next_next_next_states = torch.zeros((self.buffer_size, self.state_size)).to(device)
        # final dones after 3 state look ahead
        self.done = torch.zeros((self.buffer_size,1)).to(device)
        self.priorities = torch.zeros((self.buffer_size,1)).to(device)  # replay priority

    
    def add(self, state, action, reward, next_state, next_action, next_reward, next_next_state, next_next_action, next_next_reward, next_next_next_state, done, priorities):
        """Add a new experience to memory."""
        # pointer to end of buffer
        ix = self.buffer_ix
        # add experiences to buffers
        self.states[ix:ix+20] = torch.from_numpy(state)
        self.actions[ix:ix+20] = torch.from_numpy(action)
        self.next_actions[ix:ix+20] = torch.from_numpy(next_action)
        self.next_next_actions[ix:ix+20] = torch.from_numpy(next_next_action)
        self.rewards[ix:ix+20] = torch.tensor(reward).view(20,1)
        self.next_rewards[ix:ix+20] = torch.tensor(next_reward).view(20,1)
        self.next_next_rewards[ix:ix+20] = torch.tensor(next_next_reward).view(20,1)
        self.next_states[ix:ix+20] = torch.from_numpy(next_state)
        self.next_next_states[ix:ix+20] = torch.from_numpy(next_next_state)
        self.next_next_next_states[ix:ix+20] = torch.from_numpy(next_next_next_state)
        self.done[ix:ix+20] = torch.tensor(done).view(20,1)
        self.priorities[ix:ix+20] =  torch.from_numpy(priorities)
        # update the number of elements in the buffer
        self.buffer_len += 20
        self.buffer_len = np.minimum(self.buffer_len, self.buffer_size)
        # increment buffer pointer
        self.buffer_ix += 20
        self.buffer_ix = self.buffer_ix %self.buffer_size

    def update_priorities(self, priorities):
        '''
        updates the priorites of replayed tuples
        '''
        self.priorities[self.batch_ixs] = priorities
        
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # convert priorities to probabilities
        probabilities = self.priorities[:self.buffer_len, 0].cpu().numpy()
        probabilities += 0.05*np.random.rand(self.buffer_len)
        probabilities = probabilities**0.0 / np.sum(probabilities**0.0)
        self.batch_ixs = np.random.choice(self.buffer_len, size=self.batch_size, p=probabilities)

        states = self.states[self.batch_ixs]
        actions = self.actions[self.batch_ixs]
        next_actions = self.next_actions[self.batch_ixs]
        next_next_actions = self.next_next_actions[self.batch_ixs]
        rewards = self.rewards[self.batch_ixs]
        next_rewards = self.next_rewards[self.batch_ixs]
        next_next_rewards = self.next_next_rewards[self.batch_ixs]
        next_states = self.next_states[self.batch_ixs]
        next_next_states = self.next_states[self.batch_ixs]
        next_next_next_states = self.next_states[self.batch_ixs]
        dones = self.done[self.batch_ixs]

        return (states, actions, rewards, next_states, next_actions, next_rewards, next_next_states, next_next_actions, next_next_rewards, next_next_next_states, dones, probabilities[self.batch_ixs])

    def __len__(self):
        """Return the current size of internal memory."""
        return self.buffer_len