from pacai.agents.capture.capture import CaptureAgent
import random
import time
import logging
from pacai.util import util
from pacai.core.directions import Directions
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def createTeam(firstIndex, secondIndex, isRed,
        first = 'OffensiveAgent',
        second = 'DefensiveAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created, and will be False if the
    blue team is being created.
    """
    return [
        eval(first)(firstIndex),
        eval(second)(secondIndex),
    ]

class OffensiveAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

    def chooseAction(self, gameState):

        # Retrieve all possible legal actions for the current state
        actions = gameState.getLegalActions(self.index)

        # Performance tracking
        start_time = time.time()

        # Evaluate each possible action
        action_values = {}
        for action in actions:
            action_values[action] = self.evalFcn(gameState, action)

        # Debugging
        elapsed_time = time.time() - start_time
        logging.debug(f'Evaluation time for agent {self.index}: {elapsed_time:.4f} seconds')

        # Find the maximum evaluation score
        max_value = max(action_values.values())

        # Filter for maximum score
        best_actions = [action for action, value in action_values.items() if value == max_value]

        # Randomly select from best_actions
        return random.choice(best_actions)


    def evalFcn(self, state, action):
        weights = self.score(state, action)
        features = self.featureGenerator(state, action)
        return sum(features.get(f, 0) * w for f, w in weights.items())

    def featureGenerator(self, state, action):

        # Generate the successor
        successor = self.getSuccessor(state, action)

        # Get the agent current state and position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features = {}

        # Tracks the agent's success in capturing points.
        features['successorScore'] = self.getScore(successor)

        # Distance to the Nearest Food
        # Encourages agent to prioritize closer food
        foodList = self.getFood(successor).asList()
        if foodList:
            features['distanceToFood'] = min(self.getMazeDistance(myPos, food) for food in foodList)

        # Distance to the Nearest Ghost (Guard)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        guards = [enemy for enemy in enemies if enemy.isGhost() and enemy.getPosition() is not None]

        guardDistances = [self.getMazeDistance(myPos, guard.getPosition()) for guard in guards]
        if guardDistances:
            features['guardDistance'] = min(guardDistances)

        # Distance to the Nearest Capsule
        capsuleList = self.getCapsules(successor)
        if capsuleList:
            features['distanceToCapsule'] = min(self.getMazeDistance(myPos, capsule) for capsule in capsuleList)

        # Distance to the Nearest Ally
        allies = [successor.getAgentState(i) for i in self.getTeam(successor)]
        allyDistances = [self.getMazeDistance(myPos, ally.getPosition()) for ally in allies]
        if allyDistances:
            features['allyDistance'] = min(allyDistances)

        return features


    def score(self, state, action):
        scores = {
            'successorScore': 100,
            'distanceToFood': -1,
            'guardDistance': 0.75,
            'allyDistance': 0,
            'distanceToCapsule': -0.2
        }
        # Forces logger to output this line
        # logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(f"Score Calculation for {action}: {scores}")
        return scores





    def getSuccessor(self, gameState, action):
       
        # If the position is not matching with the nearest grid point, generate an additional successor
        successor = gameState.generateSuccessor(self.index, action)
        newPos = successor.getAgentState(self.index).getPosition()

        # Makes sure the agent is on a valid grid point before returning the state
        return successor.generateSuccessor(self.index, action) if newPos != util.nearestPoint(newPos) else successor



class DefensiveAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

    def chooseAction(self, gameState):

        possible_actions = gameState.getLegalActions(self.index)

        action_values = {action: self.evalFcn(gameState, action) for action in possible_actions}

        # Debugging
        start_time = time.time()
        logging.debug(f'Evaluation time for agent {self.index}: {time.time() - start_time:.4f} seconds')

        # Get the maximum score
        max_value = max(action_values.values())

        # Get all actions that have the maximum score
        best_actions = [action for action, value in action_values.items() if value == max_value]

        # Randomly select from best_actions
        return random.choice(best_actions)


    def evalFcn(self, state, action):
        weights = self.score(state, action)
        features = self.featureGenerator(state, action)
        return sum(features.get(f, 0) * w for f, w in weights.items())

    def featureGenerator(self, state, action):

        # Evaluate the potential outcome of the action
        successor = self.getSuccessor(state, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features = {}

        # Defense Status
        # Tracks if the agent is playing defense or offense
        features['onDefense'] = 1 if not myState.isPacman() else 0

        # Getting the number of Visible Invaders
        # Prioritize defending against visible enemy Pacman
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [enemy for enemy in enemies if enemy.isPacman() and enemy.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        # Distance to the Nearest Invader
        # Makes the agent to move closer to invaders to intercept
        if invaders:
            features['invaderDistance'] = min(self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders)

        # Distance to the Nearest Enemy
        # Make agent aware of nearby opponents
        if enemies:
            features['enemyDistance'] = min(self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies)

        # Penalty for Stopping
        if action == Directions.STOP:
            features['stop'] = 1

        # Penalty for Reversing, unless needed
        current_direction = state.getAgentState(self.index).getDirection()
        if action == Directions.REVERSE[current_direction]:
            features['reverse'] = 1

        return features


    def score(self, state, action):
        scores = {
            'numInvaders': -800,    # Still a high priority, but less aggressive
            'onDefense': 100,       # Encourage defensive positioning
            'invaderDistance': -100, # Balanced priority for interception
            'stop': -10,            # Stronger penalty for inactivity
            'reverse': -5,          # Mild penalty to prevent excessive reversing
            'enemyDistance': -30    # More emphasis on enemy tracking
        }
        logging.debug(f"Score Calculation for {action}: {scores}")
        return scores



    def getSuccessor(self, gameState, action):
       
        # If the position is not matching with the nearest grid point, generate an additional successor
        successor = gameState.generateSuccessor(self.index, action)
        newPos = successor.getAgentState(self.index).getPosition()

        # Makes sure the agent is on a valid grid point before returning the state
        return successor.generateSuccessor(self.index, action) if newPos != util.nearestPoint(newPos) else successor



