# -*- coding: utf-8 -*-

import math
import random
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from net import DQN
import data
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

if __name__ == '__main__':
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if gpu is to be used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))


    class ReplayMemory(object):

        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def push(self, *args):
            """Saves a transition."""
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    BATCH_SIZE = 64
    GAMMA = 0.95
    TARGET_UPDATE = 30

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)


    episode_durations = []


    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())


    def optimize_model():
        if len(memory) < BATCH_SIZE:
            time.sleep(1)
            return 0
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return 1
        
    data.start()
    if len(sys.argv) > 2:
        policy_net.load_state_dict(torch.load("models/" + sys.argv[2] + ".pt"))
        target_net.load_state_dict(policy_net.state_dict())
        data.updateNet(policy_net.state_dict())
    
    num_episodes = int(sys.argv[1])
    i_episode = 0
    data_count = 0
    net_count = 0
    while i_episode < num_episodes:
        for data_point in data.getData():
            data_count += 1
            action_dict = {'l':0, 's':1, 'r':2}
            state = data_point[0].to("cpu")
            action = torch.tensor([[action_dict[data_point[1]]]], device="cpu")
            if data_point[2] is None:
                next_state = None
            else:
                next_state = data_point[2].to("cpu")
            reward = torch.tensor([float(data_point[3])], device="cpu")
            memory.push(state, action, next_state, reward)
            
        #print("Optimizing, {0}".format(len(memory)))
        i_episode += optimize_model()
        if i_episode % TARGET_UPDATE == TARGET_UPDATE - 1:
            print("UPDATE_NET, Data Count: ", data_count, "Episode Count: ", i_episode)
            target_net.load_state_dict(policy_net.state_dict())
            data.updateNet(policy_net.state_dict())
            
        if i_episode % 1000 == 999:
            torch.save(policy_net.state_dict(), "models/" + str(net_count) + ".pt")
            net_count += 1

    print('Complete')
    data.stop()
    plt.ioff()
    plt.show()
