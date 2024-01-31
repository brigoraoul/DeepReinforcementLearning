import numpy as np

from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.vector import Vector

from staticParameters import StaticParameters


# Class 'AgentVisualization':
#   This class represents the visual representation of the agent and its actions in the kivy GUI. Additionally, this
#   class is responsible for composing a signal which is then used as input for the neural net. Part of this signal are
#   multiple sensors which scan the surroundings of the agent's current state. I have taken inspiration for the general
#   implementation of these sensors from this udemy course:
#   https://www.udemy.com/course/artificial-intelligence-az/
class AgentVisualization(Widget):

    # loading required static parameters
    GUI_WIDTH = StaticParameters.GUI_WIDTH
    GUI_HEIGHT = StaticParameters.GUI_HEIGHT
    AGENT_STEP_SIZE = StaticParameters.AGENT_STEP_SIZE
    wall = StaticParameters.wall

    # Kivy properties are used to check for errors of changing variables that are displayed in a kivy application.
    # For example kivy.properties.NumericProperty checks if a value is of numeric type.
    # kivy.properties.ReferenceListProperty can be used to represent a position composed of two NumericProperties.
    # Reference: https://kivy.org/doc/stable/api-kivy.properties.html#

    # angle between the x-axis of the map and the orientation of the agent
    angle = NumericProperty(0)
    # last turn the agent made
    rotation = NumericProperty(0)

    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    def start(self, start_position, direction):
        self.center = start_position
        self.velocity = direction

    # Method 'move':
    # Every time the agent takes an action, the visible agent in the GUI and its sensors should change their location.
    # Additionally, because the agent is now in a different state, the signals have to be recalculated.
    def move(self, direction):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = direction.value  # Transform direction from type helperClasses.direction.Direction to Integer
        self.angle = self.angle + self.rotation

        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

        # checking for application edges
        self.avoid_out_of_bounce(20)

        # readjust direction
        self.velocity = Vector(self.AGENT_STEP_SIZE, 0).rotate(self.angle)

    # Method 'get_signal':
    # Depending on the current state of the agent, it composes a signal from all its current information about its
    # state in the model. This signal is the basis for the neural net to decide on what action to take next.
    def get_signal(self, goal):
        signal1 = self.calculate_sensor_signal(self.sensor1_x, self.sensor1_y)
        signal2 = self.calculate_sensor_signal(self.sensor2_x, self.sensor2_y)
        signal3 = self.calculate_sensor_signal(self.sensor3_x, self.sensor3_y)
        orientation = self.get_orientation(goal)

        collective_signal = [signal1, signal2, signal3, orientation, -orientation]
        return collective_signal

    # Method 'get_orientation':
    # Returns the angle between the current direction the agent is headed and the direction to the goal.
    def get_orientation(self, goal):
        direction_of_goal = Vector((goal[0] - self.x), (goal[1] - self.y)).normalize()
        orientation = Vector(self.velocity).angle(direction_of_goal) / 180
        return orientation

    # Method 'calculate_sensor_signal':
    # If the signal of a sensor would only stand for the one position that the sensor is located on, this information
    # would not be very useful for detecting walls. So, instead, a sensor measures an area of 20 x 20 around its
    # location and produces a signal which stands for the ratio of fields which are wall and fields which are no wall
    # in this area.
    def calculate_sensor_signal(self, x, y):
        x = int(x)
        y = int(y)
        area = self.wall[x - 10:x + 10, y - 10:y + 10]
        return int(np.sum(area)) / 400

    # Method 'avoid_out_of_bounce':
    # If the agent comes too close to one of the application window's edges, its position is corrected away from the
    # edge. This is necessary, so there is so it does not try to access an out-of-bounce-element of
    # StaticParameters.wall.
    def avoid_out_of_bounce(self, safety_distance):
        if self.x < safety_distance:
            self.x = safety_distance
        elif self.x > self.GUI_WIDTH - safety_distance:
            self.x = self.GUI_WIDTH - safety_distance
        elif self.y < safety_distance:
            self.y = safety_distance
        elif self.y > self.GUI_HEIGHT - safety_distance:
            self.y = self.GUI_HEIGHT - safety_distance

    # Method 'distance_to_goal':
    # Returns the current distance from the visualized agent's center to the goal, using the Pythagorean theorem.
    def distance_to_goal(self, goal):
        a = self.x - goal[0]
        b = self.y - goal[1]
        return np.sqrt(a**2 + b**2)
