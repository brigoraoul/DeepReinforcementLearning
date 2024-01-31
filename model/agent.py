import torch
import torch.optim as optim             # optim module contains optimizers for stochastic gradient descent (like adam)
import torch.nn.functional as F         # The 'functional' module contains loss-functions for neural networks.

import logging
import math
import random

from network import NeuralNet2Layer, NeuralNet1Layer
from replayMemory import ReplayMemory
from fileManager import FileManager
from transition import Transition
from direction import Direction
from staticParameters import StaticParameters


# Class 'Agent':
#   This class contains the actual q-learning implementation, making use of a neural net from 'network.py' and the
#   replay memory from 'replayMemory.py'. The main two responsibilities of this class are the selection of actions and
#   the q-learning optimization of the network. I have taken inspiration for the general structure of the agent
#   implementation from the referenced pytorch reinforcement learning tutorial.
#   Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class Agent:

    def __init__(self):

        # check if a gpu is available and if so use it to process pytorch tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instance variables which are loaded from class StaticParameters are explained in class StaticParameters

        self.model = NeuralNet1Layer(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=StaticParameters.LEARNING_RATE)

        self.memory = ReplayMemory(StaticParameters.REPLAY_MEMORY_CAPACITY)

        self.eps_start = StaticParameters.EPSILON_START
        self.eps_end = StaticParameters.EPSILON_END
        self.eps_decay = StaticParameters.EPSILON_DECAY

        self.gamma = StaticParameters.GAMMA
        self.batch_size = StaticParameters.BATCH_SIZE

        self.filename = StaticParameters.MODEL_FILENAME

        self.last_transition = Transition(torch.Tensor(StaticParameters.INPUT, device=self.device).float().unsqueeze(0),
                                          torch.Tensor(StaticParameters.INPUT, device=self.device).float().unsqueeze(0),
                                          torch.tensor([0], device=self.device).unsqueeze(0),
                                          torch.tensor([0], device=self.device))

        # number of completed steps, used for epsilon decay
        self.steps_done = 0

    # Method 'select_action':
    #   Purpose: The output of the neural net contains a value for every possible action. So, somehow the DQN still
    #       needs to determine what action to take next. Essentially, this is done by using the softmax-function
    #       (torch.nn.functional.softmax) as an activation function for the neural net output layer. Softmax returns a
    #       probability distribution for all actions. Based on this distribution and the probability of choosing a
    #       random action, 'select_action' returns the next action to perform for the agent.
    #   Parameters:
    #       'nn_input': input for the neural net in form of a tensor (torch.Tensor)
    #   Return: The next action to perform for the agent in form of a tensor (torch.Tensor)
    def select_action(self, nn_input):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done = self.steps_done + 1

        if random.random() > eps_threshold:
            nn_output = self.model.forward(nn_input)
            return nn_output.max(1)[1].view(1, 1)  # take recommendation of the model

        return torch.tensor([[random.randrange(3)]], device=self.device)  # choose randomly

    # Method 'optimize_model':
    #   Purpose: After a reasonable amount of experience in form of transitions stored in memory, the agent starts
    #       using this experience to develop a policy for the model. More specifically, it starts optimizing its
    #       neural net for a specific loss function, in this case the 'Huber Loss' (Reference:
    #       https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html). The actual optimization is done
    #       by updating the neural net's weights with backpropagation.
    def optimize_model(self):
        if not self.memory.has_batch_size(self.batch_size):
            return

        transitions = self.memory.sample(self.batch_size)
        states, new_states, actions, rewards = transitions

        output = torch.gather(self.model.forward(states), 1, actions)
        new_output = self.model.forward(new_states).max(1)[0].detach()  # max(Q(a_{t}, s_{t}))
        expected = rewards + self.gamma * new_output  # R(a_{t}, s_{t},) + y * max(Q(a_{t}, s_{t+1}))

        # Note: Here, I used 'torch.nn.functional.smooth_l1_loss' instead of 'torch.nn.SmoothL1Loss'. The reason is
        # that the latter is actually calling the former itself but because of its parent class can have a reduction
        # that is different than mean reduction. For this project's agent, this can lead to very inefficient learning.
        # Reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#SmoothL1Loss
        loss = F.smooth_l1_loss(output.squeeze(1), expected)
        self.optimizer.zero_grad()  # setting all gradients to zero, so it does not accumulate over time
        loss.backward()  # calculate backpropagation
        self.optimizer.step()  # update weights according to backpropagation

    # Method 'update':
    #   Purpose: Every time the agent has selected an action, it changes its state in the model. Concrete,
    #       this means new input data is available, which should be used to select the next action, if the state is not
    #       a final state. This method updates all relevant variables and initiates a new learning iteration by calling
    #       'optimize_model'.
    #   Parameters:
    #       'reward': reward that resulted from the last action, calculated by class 'Environment'
    #       'new_signal': information about the current state of the agent in the model, provided by class
    #           'AgentVisualization'
    def update(self, reward, new_signal):
        new_state = torch.Tensor([new_signal], device=self.device).float()
        self.last_transition = Transition(self.last_transition[0], new_state,
                                          self.last_transition[2], self.last_transition[3])
        self.memory.push(self.last_transition)
        self.optimize_model()

        # compute and update current state
        new_action = self.select_action(new_state)
        self.last_transition = (new_state, torch.Tensor([StaticParameters.INPUT], device=self.device).float(),
                                new_action, torch.tensor([reward], device=self.device))

        # return action of type enum 'Action'
        # not implemented as match pattern (switch/case) because this is only supported from Python 3.10
        if new_action == 0:
            return Direction.STRAIGHT
        if new_action == 1:
            return Direction.RIGHT
        return Direction.LEFT

    # Method 'save':
    # Saving the current state of the neural net to a file so it can be reused and training does not have to start at 0
    # every time.
    def save(self):
        FileManager.save_model(self.model, self.optimizer, self.filename)
        logging.info('Model successfully saved.')

    # Method 'load':
    # 'model' and 'optimizer' are passed as reference, so it is not necessary for 'FileManager.load_model' to return
    # anything. Instead it directly sets them to the stored values.
    def load(self):
        if FileManager.load_model(self.model, self.optimizer, self.filename):
            logging.info('Model successfully loaded.')
