import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse
from kivy.clock import Clock
from kivy.config import Config

from agentVisualization import AgentVisualization  # ignore IDE warning of unused import because it is used by kivy
from wallVisualization import WallVisualization
from staticParameters import StaticParameters
from environment import Environment

# require installed kivy version
kivy.require(kivy.__version__)

# set width and height of the application window
width = str(int(StaticParameters.GUI_WIDTH / 2))
height = str(int(StaticParameters.GUI_HEIGHT / 2))
Config.set('graphics', 'width', width)
Config.set('graphics', 'height', height)
# set maximum number of frames per second
Config.set('postproc', 'maxfps', '0')


# Class 'RLAgentApp'
#   This marks the entry point for the entire application. It inherits from kivy.app.App and is thus a GUI-application.
#   It contains a build method, which is executed by the call 'RLAgentApp.run()' and methods to be called when a button
#   in the GUI is pressed.
#   Reference: https://kivy.org/doc/stable/tutorials/pong.html
class RLAgentApp(App):

    clock_active = False
    clock = None

    def restart_iteration(self, obj):
        self.parent.start()

    def load(self, obj):
        self.parent.agent.load()

    def start(self, obj):
        if not self.clock_active:
            self.clock = Clock.schedule_interval(self.parent.update, StaticParameters.AGENT_MOVING_ABILITY)
            self.clock_active = True

    def on_stop(self):
        import sys
        sys.exit()

    def draw_goal(self):
        with self.walls.canvas:
            Color(0, 1, 0)
            Ellipse(
                pos=(StaticParameters.GUI_WIDTH - 120, StaticParameters.GUI_HEIGHT - 120),
                size=(100, 100)
            )

    # Method 'build'
    # This builds the GUI window with all its components (buttons, agent visualization, wall visualization, goal
    # visualization)
    def build(self):

        self.parent = Environment()
        self.walls = WallVisualization()
        self.clock = None

        # drawing goal for the agent as a circle in the upper right corner
        self.draw_goal()

        self.parent.add_widget(self.walls)

        # define visible user buttons with text, position and method to call 'on_release' and add them to the root
        # widget
        start_btn = Button(
            text='start',
            pos=(1 * self.parent.width, StaticParameters.GUI_HEIGHT - self.parent.height * 2),
            background_color=(0, 1, 0)
        )
        start_btn.bind(on_release=self.start)
        self.parent.add_widget(start_btn)

        load_btn = Button(
            text='load',
            pos=(3 * self.parent.width, StaticParameters.GUI_HEIGHT - self.parent.height * 2),
            background_color=(0, 1, 0)
        )
        load_btn.bind(on_release=self.load)
        self.parent.add_widget(load_btn)

        stop_btn = Button(
            text='exit',
            pos=(4 * self.parent.width, StaticParameters.GUI_HEIGHT - self.parent.height * 2),
            background_color=(0, 1, 0)
        )
        stop_btn.bind(on_release=self.stop)
        self.parent.add_widget(stop_btn)

        reset_btn = Button(
            text='reset',
            pos=(2 * self.parent.width, StaticParameters.GUI_HEIGHT - self.parent.height * 2),
            background_color=(0, 1, 0)
        )
        reset_btn.bind(on_release=self.restart_iteration)
        self.parent.add_widget(reset_btn)

        return self.parent


# starting the application
if __name__ == '__main__':
    RLAgentApp().run()
