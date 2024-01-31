import numpy as np

import logging

from kivy.uix.widget import Widget
from kivy.graphics import Color, Line

from staticParameters import StaticParameters


# Class 'WallVisualization':
#   The WallVisualization is responsible for handling the representation of walls in the gui. A user has to be able to
#   draw walls by right-clicking and dragging over the gui. This class provides the required functionality and makes use
#   of the kivy graphics library. It inherits from kivy.uix.widget.Widget
#   References:
#       https://kivy.org/doc/stable/tutorials/firstwidget.html
#       https://progi.pro/kivi-obekt-sozdayushiyrisuyushiy-sled-za-hodom-dvizheniya-7899881
class WallVisualization(Widget):
    # Coordinates of the last point drawn
    position = (0, 0)
    # Number of points in the last drawing
    number_of_points = 0
    # Length of the last drawing
    length = 0
    # Determines how thick lines will be displayed in the GUI (the higher this number, the thicker the lines).
    line_width = 30

    # Method 'on_touch_down':
    # This method gets called whenever the user right-clicks at any position in the application window. In anticipation
    # of the users wanting to draw a line, the line parameters are reset.
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1)
            touch.ud['line'] = Line(points=(touch.x, touch.y))
            self.position = (int(touch.x), int(touch.y))
            self.number_of_points = 0
            self.length = 0

            # set the touched square to 1 in the wall array used by the agent
            StaticParameters.wall[int(touch.x), int(touch.y)] = 1

    # Method 'on_touch_move':
    # This method is called whenever the user has right-clicked on the application window and drags over it. It also
    # sets a reasonable size for the drawn lines, depending on the speed with which the user draws.
    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

        new_position = (int(touch.x), int(touch.y))
        self.length = self.length + np.sqrt(max((new_position[0] - self.position[0]) ** 2
                                                + (new_position[1] - self.position[1]) ** 2, 2))
        self.number_of_points = self.number_of_points + 1

        try:
            touch.ud['line'].width = int(self.line_width * self.number_of_points / np.max([self.length, 1]))
        except:
            logging.warning('User has drawn on GUI to fast!')

        self.position = new_position

        # set all touched squares to 1 in the wall array used by the agent
        StaticParameters.wall[int(touch.x) - int(self.line_width / 2): int(touch.x) + int(self.line_width / 2),
                            int(touch.y) - int(self.line_width / 2): int(touch.y) + int(self.line_width / 2)] = 1
