import numpy as np


class StaticParameters:

    # 1. GUI parameters

    # Specifies how often the agent updates its direction (in seconds). The lower this number, the faster and more
    # sensible will the agent move through the model.
    # Note: The kivy clock does not work with actual time but with the frame rate of the application. Because of this,
    # 'AGENT_MOVING_ABILITY' has a certain minimum, which depends on the set frame rate and the machine that the
    # application runs on. To make the agent move even faster beyond the minimum, you can either increase the
    # 'AGENT_STEP_SIZE' or start the kivy clock scheduler multiple times. However, both of these options only make the
    # agent visualization move faster but do not translate to more learning iterations per second for the agent.
    AGENT_MOVING_ABILITY = 0.001

    # Specifies the speed with which the agent moves through the model, more specifically, the distance it will
    # cover each defined time step.
    AGENT_STEP_SIZE = 8

    # Specifies the maximal number of iterations for the agent to learn. After the agent has gone through this number of
    # iterations the application will save the model and exit.
    MAX_ITERATIONS = 1000

    # Width and length of the GUI.
    GUI_WIDTH = 2000
    GUI_HEIGHT = 1500

    # 2. Deep Learning parameters:

    # Path to file where the last saved model can be found and a new model should be saved.
    # This path will be "overwritten" in case an argument is specified when running the application.
    MODEL_FILENAME = 'lastModel/trained_model.pt'

    # discount factor (set to 1 to erase its effect)
    GAMMA = 0.9

    # learning rate for the neural network optimizer (minimum = 0)
    LEARNING_RATE = 0.001

    # number of samples to be taken from replay memory for one learning iteration
    BATCH_SIZE = 100

    # parameters for epsilon-greedy-algorithm
    # https://www.baeldung.com/cs/epsilon-greedy-q-learning
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 200

    # maximum number of transitions to be stored in replay memory
    REPLAY_MEMORY_CAPACITY = 100000

    # 3. Neural net layer sizes

    # When input size is changed, the actual input array created in class 'AgentVisualization' has to be updated to
    # contain the correct number of elements.
    INPUT = 5
    HIDDEN_1 = 32
    HIDDEN_2 = 16
    OUTPUT = 3

    # 4. not static variable wall (do not change to an array bigger than '(GUI_WIDTH, GUI_HEIGHT)')

    # Array of same size as the application window. When a wall is drawn onto the model, the respective points
    # in the array will be set to 1.
    wall = np.zeros((GUI_WIDTH, GUI_HEIGHT))
