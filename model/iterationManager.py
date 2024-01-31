import logging

logging.basicConfig(filename='model.log', encoding='utf-8', level=logging.DEBUG)


# Class 'IterationManager':
# Whenever the agent reaches its goal, a new iteration is supposed to start, but only if the maximum number of
# iterations is not yet exceeded. This class ensures handles all functionality related to these iterations.
# It also stores iteration specific data in a logfile.
class IterationManager:

    def __init__(self, max_iterations, current_iteration=0):
        self.max_iterations = max_iterations
        self.current_iteration = current_iteration

    # Method 'check_iteration':
    #   Purpose: Check, if the agent is currently close enough to the goal, so a new iteration can begin.
    #   Returns: Boolean to indicate whether a new iteration should begin (True) or not (False).
    def check_iteration(self, distance, agent, cumulative_reward, walls_touched):
        next_iteration = self.iteration_finished(distance)
        if next_iteration:
            self.max_iterations_reached(agent)
            logging_message = 'Iteration ' + str(self.current_iteration) + ': '\
                     + 'Cumulative reward: ' + str(cumulative_reward+200) + ", Walls touched: " + str(walls_touched)
            logging.info(logging_message)
            self.safe_model_iteration(agent)
            return True

        return False

    def iteration_finished(self, distance):
        # number defines how close the agent has to come to the goal position for the model to count it as
        # goal reached
        if distance < 60:
            self.current_iteration = self.current_iteration + 1
            return True

        return False

    # ending the application, if the max number of iterations is reached
    def max_iterations_reached(self, agent):
        if self.current_iteration > self.max_iterations:
            agent.save()
            logging.info('Max number of iterations is reached.')
            import sys
            sys.exit()

    # saving the model every five iterations
    def safe_model_iteration(self, agent):
        if self.current_iteration % 5 == 0:
            agent.save()

