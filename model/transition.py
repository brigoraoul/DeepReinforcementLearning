from collections import namedtuple

# All elements of a transition have to be torch tensors.
Transition = namedtuple('Transition',
                        ['state', 'new_state', 'action', 'reward'])
