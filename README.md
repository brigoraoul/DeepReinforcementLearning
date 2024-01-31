# DeepReinforcementLearning

The folder "DeepQLearningAgent" contains the implementation of a grid-world environment and a reinforcement learning agent, build with PyTorch.

### Required Installations:

With the following installations, the application can be launched from the command line:

- **Python** (developed with 2.7.0 and 3.9.7, minimum version according to vermin: 2.5 or 3.0)

- **Kivy** (minimum version 2.0.0): 
Kivy can be installed via the Python package installer "pip" with the following command: 
```
python -m pip install kivy[base]
```
Alternatively, the following command can be used with "Anaconda":
```
conda install kivy -c conda-forge
```
More extensive information about the installation process can be found on the Kivy-getting started page at https://kivy.org/doc/stable/gettingstarted/installation.html.

- **PyTorch** (minimum version 1.7.1):
With "pip", the PyTorch library can be installed on macOS and Windows via:
```
pip3 install torch torchvision torchaudio
```
On a Linux-machine, a different command is necessary: 
```
pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch\_stable.html
```
Alternatively, with "Anaconda" the installation command for macOS is: 
```
conda install pytorch torchvision torchaudio -c pytorch
```
For Windows or Linux with "Anaconda": 
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
More extensive information about the installation process can be found on the Pytorch-getting started page at https://pytorch.org/get-started/locally/.

- **Numpy** (minimum version 1.20.0): The Numpy library can also be installed with "pip":
```
pip install numpy
```
or with "Anaconda":
```
conda install numpy
```

### Starting the application:

The starting point for the application, is the file _DeepQLearningAgent/model/rlAgent.py_. To launch it from the console type 

```python3 rlAgent.py``` or ```python rlAgent.py```

It is possible to pass two filenames as arguments to the application to store and load a model. For example, if the application is started with 
```
python3 rlAgent.py new_model.pt old_model.pt
```
, it will save the model to _DeepQLearningAgent/model/lastModel/new_model.pt_ and whenever requested, load the model stored in _DeepQLearningAgent/model/lastModel/old_model.pt_, given that this file exists.

To be able to execute the program, it is necessary to have Python, PyTorch and Kivy installed. 

### Using the GUI:

Once started, the GUI window shows a red dot in the lower left corner and a bigger, green dot in the upper right corner. The red dot represents the agent, while the green dot represents the agent's goal which it has to get to. The entire window represents the environment in which the agent can move.

The GUI also displays five user buttons:
- "start": Gives the commando for the agent to start exploring the environment and find the goal. The red dot will start to move.
- "reset": Resets the current iteration, meaning the agent will be set back to its starting point in the lower left corner. However, the agent itself and the environment are not reset, so no learning progress is lost. After pressing "reset" the agent will continue exploring right away, the user does not have to press "start" again.
- "load": Loads a previously trained and stored version of the agent from a file. Alternatively to the option of passing the file location and name as a parameter, they can be specified in the class "StaticParameters". The program automatically saves the current state of the agent after every fifth iteration to the same file.
- "exit": Shuts down the application.

By right-clicking and dragging over the application window, the user can draw "walls". Walls should be drawn rather slowly, as the thickness of a wall depends on the speed with which the user drags the courser. In case the drawn lines become too thin, a warning occurs in the command line. Whenever the agent reaches the goal, it immediately starts the next iteration and the command line logs information about the finished iteration.

### Changing application parameters:

To enable quick parameter tuning and general testing of the application, many parameters, regarding both q-learning and the GUI, are stored in the class "StaticParameters". These parameters only need to be changed in "StaticParameters" and will be dynamically updated everywhere else in the application.

### Getting more information about the implementation

Various parts of the code were inspired by publicly available examples and library documentation. These parts are indicated with a comment at the respective locations in the code. The linked references can be code examples but also explanations of an abstract concept.
