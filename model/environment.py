from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.vector import Vector

import numpy as np
import logging

from staticParameters import StaticParameters
from agent import Agent
from iterationManager import IterationManager


# Class 'Environment':
#   This class represents the grid-world environment displayed in the GUI. Since the q-learning agent itself is part of
#   the environment, this class contains an instance of class Agent from 'agent.py'.
class Environment(Widget):

    # loading required static parameters
    GUI_WIDTH = StaticParameters.GUI_WIDTH
    GUI_HEIGHT = StaticParameters.GUI_HEIGHT
    AGENT_STEP_SIZE = StaticParameters.AGENT_STEP_SIZE
    wall = StaticParameters.wall

    # indirectly creating an object of class 'agentVisualization' by creating a kivy agent visualization, which is
    # specified in file 'rlagent.kv'
    agentVisualization = ObjectProperty(None)

    # 'agent' represents the reinforcement learning agent from the module 'deepQLearning'
    agent = Agent()

    # goal for the agent as a position in the model (actual goal is a circle of certain diameter around this
    # position)
    goal = (GUI_WIDTH - 60, GUI_HEIGHT - 60)

    # Specifies the maximal number of iterations for the agent to learn. After the agent has gone through this number of
    # iterations the application will save the model and exit.
    max_iterations = StaticParameters.MAX_ITERATIONS

    # The model has an iteration manager, which determines when an iteration is finished and how many iterations
    # the agent goes through.
    iteration_manager = IterationManager(max_iterations)

    # The 'current_reward' denotes the received reward for the last action that the agent took.
    current_reward = 0
    # The 'cumulative_reward' stands for the sum of all current rewards over an entire iteration and is hence set back
    # to 0 after each iteration.
    cumulative_reward = 0

    # how often has the agent touched a wall in the current iteration
    walls_touched = 0

    # current direct distance to the goal, not regarding walls
    distance_to_goal = 0

    def init_wall(self):
        StaticParameters.wall = np.zeros((self.GUI_WIDTH, self.GUI_HEIGHT))
        self.wall = StaticParameters.wall

    # Method 'start':
    # Starting point for every iteration. The agent is set to its starting position in the model and receives a
    # initial speed and direction.
    def start(self):
        self.agentVisualization.start(Vector(100, 100), Vector(self.AGENT_STEP_SIZE, 0))
        self.cumulative_reward = 0
        self.walls_touched = 0

    # Method 'update':
    #   Purpose: This method is called in a regular time interval from the kivy application. Every call leads to the
    #       agent taking an action in the model and hence leads to a new state. This process is realized by the
    #       following steps:
    #           1. get signal from agent and use it as new input signal for the dqn neural net
    #           2. retrieving the next direction from the neural net. 'next_direction' is of type
    #           helperClasses.Direction.
    #           3. trigger the agent visualization to change its position according to the calculated 'next_direction'
    #           4. checking if the agent has reached the goal
    #           5. setting the new reward
    #           6. updating the current distance to the goal
    #   Parameters: The kivy.clock.Clock interval scheduler calls a method with one parameter, which in this case
    #       is not required. An IDE warning can be ignored but the parameter should not be removed.
    def update(self, dt):
        nn_input = self.agentVisualization.get_signal(self.goal)  # 1.
        next_direction = self.agent.update(self.current_reward, nn_input)  # 2.
        self.agentVisualization.move(next_direction)  # 3.

        # The iteration manager is called to check if a new iteration is supposed to begin. This is the case whenever
        # the agent reached its goal or comes sufficiently close to it. If this is the case the agent starts again at
        # the same start position.
        new_distance_to_goal = self.agentVisualization.distance_to_goal(self.goal)

        iteration_finished = self.iteration_manager.check_iteration(new_distance_to_goal, self.agent,
                                                                    self.cumulative_reward, self.walls_touched)  # 4.

        self.calculate_reward(iteration_finished)  # 5.
        self.distance_to_goal = new_distance_to_goal  # 6.

        if iteration_finished:
            self.start()

    # Method 'calculate_reward':
    # For every action it takes, the agent receives a reward which can be positive or negative. This method calculates
    # the rewards after every taken action and differentiates between wall and no wall.
    def calculate_reward(self, iteration_finished):
        self.agentVisualization.avoid_out_of_bounce(20)

        # reward for goal
        if iteration_finished:
            self.update_rewards(200)

        # reward for wall
        elif self.wall[int(self.agentVisualization.x), int(self.agentVisualization.y)] > 0:
            self.update_rewards(-20)
            self.walls_touched = self.walls_touched + 1

        # reward for regular field
        else:
            self.update_rewards(-0.5)  # living penalty, so that agent has an incentive to take the fastest way
            if self.agentVisualization.distance_to_goal(self.goal) < self.distance_to_goal:
                self.update_rewards(0.25)

    # Method 'update_rewards':
    # Updates both the current reward from the latest action and the cumulative reward for the entire iteration.
    def update_rewards(self, new_reward):
        self.current_reward = new_reward
        self.cumulative_reward = self.cumulative_reward + new_reward
