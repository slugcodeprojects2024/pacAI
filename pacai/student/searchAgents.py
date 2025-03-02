"""
This file contains all of the agents that can be selected to control Pacman.
To select an agent, use the '-p' option when running pacman.py.
"""

import logging
from pacai.core.actions import Actions
from pacai.core.directions import Directions
from pacai.core.distance import manhattan
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        self._numExpanded = 0

        # The state consists of pacman's position and a tuple of corners that
        # haven't been visited yet
        self._startState = (self.startingPosition, self.corners)

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        """
        if actions is None:
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

    def startingState(self):
        """
        Returns the start state (in your state space, not the full Pacman state space)
        """
        return self._startState

    def isGoal(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        return len(state[1]) == 0  # All corners have been visited

    def successorStates(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
        """
        successors = []
        x, y = state[0]  # Current position
        unvisited_corners = state[1]  # Corners that haven't been visited

        self._numExpanded += 1

        for action in Directions.CARDINAL:
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            next_pos = (nextx, nexty)

            if not self.walls[nextx][nexty]:
                remaining_corners = tuple(
                    corner for corner in unvisited_corners
                    if corner != next_pos)
                nextState = (next_pos, remaining_corners)
                successors.append((nextState, action, 1))

        return successors

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem.
    This heuristic must be consistent and admissible.
    """
    pos, unvisited_corners = state
    
    if len(unvisited_corners) == 0:
        return 0

    # Calculate distances to all corners and find minimal spanning path
    total_distance = 0
    current_pos = pos
    remaining_corners = list(unvisited_corners)

    # While there are corners left to visit
    while remaining_corners:
        # Find the closest corner from current position
        distances = [manhattan(current_pos, corner) for corner in remaining_corners]
        min_dist = min(distances)
        min_idx = distances.index(min_dist)
        
        # Add distance to closest corner
        total_distance += min_dist
        
        # Move to that corner and remove it from remaining corners
        current_pos = remaining_corners[min_idx]
        remaining_corners.pop(min_idx)

    return total_distance

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.
    
    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic; 
    almost all admissible heuristics will be consistent as well.
    """
    position, foodGrid = state
    foodList = foodGrid.asList()
    
    if not foodList:
        return 0
    
    # Find farthest and nearest food distances
    farthest_dist = 0
    nearest_dist = float('inf')
    
    for food in foodList:
        dist = manhattan(position, food)
        farthest_dist = max(farthest_dist, dist)
        nearest_dist = min(nearest_dist, dist)
    
    # If there are multiple food dots, also consider distance between them
    if len(foodList) > 1:
        max_food_dist = 0
        for i, food1 in enumerate(foodList):
            for j in range(i + 1, len(foodList)):
                food2 = foodList[j]
                max_food_dist = max(max_food_dist, manhattan(food1, food2))
        
        # Combine the distances: must reach nearest food + account for food spread
        return nearest_dist + max_food_dist
    
    # If only one food left, just use distance to it
    return farthest_dist

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, fn='pacai.student.search.breadthFirstSearch', **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot.
        """
        problem = AnyFoodSearchProblem(gameState)
        from pacai.student.search import breadthFirstSearch
        return breadthFirstSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.
    """
    def __init__(self, gameState, start=None):
        super().__init__(gameState, goal=None, start=start)
        self.food = gameState.getFood()

    def isGoal(self, state):
        """
        The state is Pacman's position.
        Returns whether there's food at this position.
        """
        x, y = state
        return self.food[x][y]

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.
    """
    def registerInitialState(self, state):
        """
        This method is called before any moves are made.
        """
        self.food = state.getFood()
        self.walls = state.getWalls()

    def getAction(self, state):
        """
        From game.py: The Agent will receive a GameState and must return an action
        from Directions.{North, South, East, West, Stop}
        """
        legal = state.getLegalActions()
        if not legal:
            return Directions.STOP

        position = state.getPacmanPosition()
        foodList = self.food.asList()

        if not foodList:
            return Directions.STOP

        # Choose action that gets us closest to nearest food
        bestDist = float('inf')
        bestAction = Directions.STOP

        for action in legal:
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(position[0] + dx), int(position[1] + dy)

            if not self.walls[nextx][nexty]:
                # Find closest food from this position
                dist = min(manhattan((nextx, nexty), food) for food in foodList)
                if dist < bestDist:
                    bestDist = dist
                    bestAction = action

        return bestAction