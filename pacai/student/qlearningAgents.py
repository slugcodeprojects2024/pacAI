from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection, probability
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # Initialize the Q-values as a dictionary mapping (state, action) pairs to values
        self.qValues = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        return self.qValues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        
        # Find the maximum Q-value for the best action in this state
        return max(self.getQValue(state, action) for action in legalActions)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        
        # Break ties randomly to avoid deterministic behavior
        best_actions = []
        best_value = float('-inf')
        
        for action in legalActions:
            q_value = self.getQValue(state, action)
            if q_value > best_value:
                best_actions = [action]
                best_value = q_value
            elif q_value == best_value:
                best_actions.append(action)
                
        return random.choice(best_actions)

    def getAction(self, state):
        """
        Compute the action to take in the current state.
        With probability `self.getEpsilon()`, we should take a random action
        and take the best policy action otherwise.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
            
        # With probability epsilon, take a random action (exploration)
        if probability.flipCoin(self.getEpsilon()):
            return random.choice(legalActions)
        # Otherwise, follow the best policy (exploitation)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Q-Learning update rule:
        Q(s,a) = (1-alpha) * Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a'))
        """
        # Current Q-value
        current_q = self.getQValue(state, action)
        
        # Max Q-value for next state (0 if nextState is terminal)
        next_max_q = self.getValue(nextState)
        
        # Calculate the new Q-value using the Q-learning update rule
        updated_q = ((1 - self.getAlpha()) * current_q) + (self.getAlpha() * 
                     (reward + self.getDiscountRate() * next_max_q))
        
        # Update the Q-value in our dictionary
        self.qValues[(state, action)] = updated_q

class PacmanQAgent(QLearningAgent):
    """
    A Q-Learning agent for Pacman.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Compute the action to take in the current state.
        With probability `self.getEpsilon()`, we should take a random action
        and take the best policy action otherwise.
        Note: To pick randomly from a list, use random.choice(list).
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
            
        # With probability epsilon, take a random action (exploration)
        if probability.flipCoin(self.getEpsilon()):
            return random.choice(legalActions)
        # Otherwise, follow the best policy (exploitation)
        else:
            return self.getPolicy(state)

    def final(self, state):
        """
        Called at the end of each game.
        """
        # Call the super-class
        
class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent that uses feature extraction.
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)()
        
        # Initialize weights for features
        self.weights = {}

    def getQValue(self, state, action):
        """
        Should return Q(state, action) = w * featureVector
        where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        return sum(self.weights.get(feature, 0) * value for feature, value in features.items())

    def update(self, state, action, nextState, reward):
        """
        Should update weights based on transition
        """
        # Get features for this state-action pair
        features = self.featExtractor.getFeatures(state, action)
        
        # Calculate the TD-error (temporal difference)
        # difference = (reward + discount * max_a' Q(s',a')) - Q(s,a)
        correction = reward + self.getDiscountRate() * self.getValue(nextState) - self.getQValue(state, action)
        
        # Update each weight based on the TD-error and feature values
        for feature, value in features.items():
            self.weights[feature] = self.weights.get(feature, 0) + self.getAlpha() * correction * value

    def final(self, state):
        """
        Called at the end of each game.
        """
        # Call the super-class final method
        super().final(state)
        
        # Print weights when training is complete
        if self.episodesSoFar == self.numTraining:
            print("Final weights:", self.weights)