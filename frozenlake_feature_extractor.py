import numpy as np
from feature_extractor import FeatureExtractor

class Actions:
    '''
    Actions for FrozenLake
      0 : mover para baixo (DOWN)
      1 : mover para cima (UP)
      2 : mover para a direita (RIGHT)
      3 : mover para a esquerda (LEFT)
    '''
    DOWN = 0
    UP = 1
    RIGHT = 2
    LEFT = 3

class FrozenLakeFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.DOWN:   np.array([1,0,0,0]), 
        Actions.UP:     np.array([0,1,0,0]), 
        Actions.RIGHT:  np.array([0,0,1,0]), 
        Actions.LEFT:   np.array([0,0,0,1])
    }

    def __init__(self, env):
        '''
        Initializes the FrozenLakeFeatureExtractor object. 
        It adds feature extraction methods to the features_list attribute.
        '''
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)
        self.features_list.append(self.f1)
        self.features_list.append(self.f2)
        self.features_list.append(self.f3)

    def get_num_features(self):
        '''
        Returns the number of features extracted by the feature extractor.
        '''
        return len(self.features_list) + self.get_num_actions()

    def get_num_actions(self):
        '''
        Returns the number of actions available in the environment.
        '''
        return len(self.get_actions())

    def get_action_one_hot_encoded(self, action):
        '''
        Returns the one-hot encoded representation of an action.
        '''
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        '''
        Checks if the given state is terminal. 
        In FrozenLake, the episode ends if the agent reaches the goal or falls into a hole.
        '''
        # Determine terminal states based on environment dynamics.
        # You might need to customize this based on your specific FrozenLake environment configuration.
        terminal_states = [0, 15, 5, 11]  # Example; Adjust this based on actual terminal states in your environment.
        return state in terminal_states

    def get_actions(self):
        '''
        Returns a list of available actions in the environment.
        '''
        return [Actions.DOWN, Actions.UP, Actions.RIGHT, Actions.LEFT]

    def get_features(self, state, action):
        '''
        Takes a state and an action as input and returns the feature vector for that state-action pair. 
        It calls the feature extraction methods and constructs the feature vector.
        '''
        feature_vector = np.zeros(len(self.features_list))
        for index, feature in enumerate(self.features_list):
            feature_vector[index] = feature(state, action)

        action_vector = self.get_action_one_hot_encoded(action)
        feature_vector = np.concatenate([feature_vector, action_vector])

        return feature_vector

    def f0(self, state, action):
        '''
        This is just the bias term.
        '''
        return 1.0

    def f1(self, state, action):
        '''
        This feature computes the proximity to the goal state. 
        It provides a higher value as the agent gets closer to the goal.
        '''
        size = int(np.sqrt(self.env.observation_space.n))
        goal_states = [self.env.desc.flatten().tolist().index(b) for b in self.env.desc.flatten() if b == b'G']
        state_pos = (state // size, state % size)
        min_dist = float('inf')
        for goal_state in goal_states:
            goal_pos = (goal_state // size, goal_state % size)
            dist = self.__manhattanDistance(state_pos, goal_pos)
            if dist < min_dist:
                min_dist = dist
        return 1 / (min_dist + 1) 

    def f2(self, state, action):
        '''
        This feature could indicate if the action moves the agent closer to the goal.
        '''
        # Placeholder logic; needs to be adapted based on the environment
        return 0.0

    def f3(self, state, action):
        '''
        This feature could indicate if the action hits a wall (if applicable).
        '''
        # Placeholder logic; needs to be adapted based on the environment
        return 0.0

    @staticmethod
    def __manhattanDistance(xy1, xy2):
        '''
        Computes the Manhattan distance between two points.
        '''
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
