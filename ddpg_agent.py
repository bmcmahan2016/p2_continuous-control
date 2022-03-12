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

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

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


    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.t_step %UPDATE_EVERY == 0:
            for _ in range(UPDATES_PER_STEP) :
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()  
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

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
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

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

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

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

        # initialize the replay buffer
        self.create_buffers()

    def create_buffers(self):
        '''creates replay buffer on GPU'''
        self.states = torch.zeros((self.buffer_size, self.state_size)).to(device)
        self.actions = torch.zeros((self.buffer_size, self.action_size)).to(device)
        self.rewards = torch.zeros((self.buffer_size,1)).to(device)
        self.next_states = torch.zeros((self.buffer_size, self.state_size)).to(device)
        self.done = torch.zeros((self.buffer_size,1)).to(device)

    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # pointer to end of buffer
        ix = self.buffer_ix
        # add experiences to buffers
        self.states[ix:ix+20] = torch.from_numpy(state)
        self.actions[ix:ix+20] = torch.from_numpy(action)
        self.rewards[ix:ix+20] = torch.tensor(reward).view(20,1)
        self.next_states[ix:ix+20] = torch.from_numpy(next_state)
        self.done[ix:ix+20] = torch.tensor(done).view(20,1)
        # update the number of elements in the buffer
        self.buffer_len += 20
        self.buffer_len = np.minimum(self.buffer_len, self.buffer_size)
        # increment buffer pointer
        self.buffer_ix += 20
        self.buffer_ix = self.buffer_ix %self.buffer_size
        
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        batch_ixs = np.random.choice(self.buffer_len, size=self.batch_size)

        states = self.states[batch_ixs]#torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = self.actions[batch_ixs]#torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = self.rewards[batch_ixs]#torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = self.next_states[batch_ixs]#torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = self.done[batch_ixs]#torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.buffer_len