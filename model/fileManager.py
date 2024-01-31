import os
import sys
import logging
import torch

logging.basicConfig(filename='model.log', encoding='utf-8', level=logging.DEBUG)


# Class 'FileManager':
#   Purpose: The file manager provides static methods to store and load a pytorch model. This makes it possible to
#       reuse an already trained agent. However, not the entire agent is saved, but only the parameters of the model
#       and the optimizer.
class FileManager:

    # Save model and optimizer to a separate file with name 'filename'. 'filename' is either overwritten or newly
    # created in the same directory.
    @staticmethod
    def save_model(model, optimizer, filename):
        if len(sys.argv) > 1:
            filename = 'lastModel/' + sys.argv[1]

        parameter_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(parameter_dict, filename)

    # Load model and optimizer from a file with name 'filename' in the same directory, if the file exists
    @staticmethod
    def load_model(model, optimizer, filename):
        if len(sys.argv) > 2:
            filename = 'lastModel/' + sys.argv[2]

        if not os.path.isfile(filename):
            logging.warning('File ' + filename + ' was not found when trying to load the model.')
            return False

        file_data = torch.load(filename)
        model.load_state_dict(file_data['model'])
        optimizer.load_state_dict(file_data['optimizer'])
        return True
