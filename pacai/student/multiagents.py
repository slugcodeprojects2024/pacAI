import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Chooses among the best options according to the evaluation function.
        """
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Evaluates a state-action pair.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        food = successorGameState.getFood()
        foodList = food.asList()
        ghostStates = successorGameState.getGhostStates()
        capsules = successorGameState.getCapsules()

        def manhattanDistance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        # Food evaluation with higher weight
        foodScore = 0
        if foodList:
            minFoodDistance = min(
                manhattanDistance(newPosition, food) for food in foodList
            )
            foodScore = 20.0 / (minFoodDistance + 1)
            # Additional reward for eating food
            if len(foodList) < len(currentGameState.getFood().asList()):
                foodScore += 100

        # Ghost evaluation with more sophisticated distance consideration
        ghostScore = 0
        for ghost in ghostStates:
            dist = manhattanDistance(newPosition, ghost.getPosition())
            if ghost.getScaredTimer() > 0:
                ghostScore += 200.0 / (dist + 1)
            else:
                if dist < 2:
                    ghostScore -= 500
                elif dist < 4:
                    ghostScore -= 200 / (dist + 1)
                else:
                    ghostScore -= 50 / (dist + 1)

        # Power pellet consideration
        capsuleScore = 0
        if capsules:
            minCapsuleDistance = min(
                manhattanDistance(newPosition, caps) for caps in capsules
            )
            capsuleScore = 100.0 / (minCapsuleDistance + 1)
            if len(capsules) < len(currentGameState.getCapsules()):
                capsuleScore += 150

        # Reward for staying in motion
        stationaryPenalty = -50 if action == 'Stop' else 0

        improvedScore = (successorGameState.getScore()
                         + foodScore
                         + ghostScore
                         + capsuleScore
                         + stationaryPenalty)
        return improvedScore


class MinimaxAgent(MultiAgentSearchAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState.
        """

        def minimax(state, depth, agentIndex):
            if depth == self.getTreeDepth() or state.isWin() or state.isLose():
                return self.getEvaluationFunction()(state)

            actions = state.getLegalActions(agentIndex)
            # For Pac-Man, filter out the "Stop" action if alternatives exist.
            if agentIndex == 0:
                filtered = [a for a in actions if a != 'Stop']
                if filtered:
                    actions = filtered

            if agentIndex == 0:
                value = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, minimax(successor, depth, 1))
                return value
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth + 1
                value = float("inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, minimax(successor, nextDepth, nextAgent))
                return value

        bestAction = None
        bestValue = float("-inf")
        actions = gameState.getLegalActions(0)
        filtered = [a for a in actions if a != 'Stop']
        if filtered:
            actions = filtered
        # Order actions based on the evaluation function to help with move ordering.
        actions = sorted(
            actions,
            key=lambda action: self.getEvaluationFunction()(
                gameState.generateSuccessor(0, action)
            ),
            reverse=True
        )
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 0, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using alpha-beta pruning.
        """

        def alphabeta(state, depth, agentIndex, alpha, beta):
            if depth == self.getTreeDepth() or state.isWin() or state.isLose():
                return self.getEvaluationFunction()(state)

            actions = state.getLegalActions(agentIndex)
            if agentIndex == 0:
                filtered = [a for a in actions if a != 'Stop']
                if filtered:
                    actions = filtered

            if agentIndex == 0:  # Maximizer (Pac-Man)
                value = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(successor, depth, 1, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Minimizer (Ghosts)
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth + 1
                value = float("inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        bestAction = None
        alpha = float("-inf")
        beta = float("inf")
        bestValue = float("-inf")
        actions = gameState.getLegalActions(0)
        filtered = [a for a in actions if a != 'Stop']
        if filtered:
            actions = filtered
        actions = sorted(
            actions,
            key=lambda action: self.getEvaluationFunction()(
                gameState.generateSuccessor(0, action)
            ),
            reverse=True
        )
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, 0, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the expectimax action using expected values for ghost moves.
        """

        def expectimax(state, depth, agentIndex):
            if depth == self.getTreeDepth() or state.isWin() or state.isLose():
                return self.getEvaluationFunction()(state)

            actions = state.getLegalActions(agentIndex)
            if agentIndex == 0:
                filtered = [a for a in actions if a != 'Stop']
                if filtered:
                    actions = filtered

            if agentIndex == 0:  # Maximizer (Pac-Man)
                value = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(successor, depth, 1))
                return value
            else:  # Expectation (Ghosts move uniformly at random)
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth + 1

                actions = state.getLegalActions(agentIndex)
                if not actions:
                    return self.getEvaluationFunction()(state)
                total = 0
                probability = 1.0 / len(actions)
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    total += probability * expectimax(successor, nextDepth, nextAgent)
                return total

        bestAction = None
        bestValue = float("-inf")
        actions = gameState.getLegalActions(0)
        filtered = [a for a in actions if a != 'Stop']
        if filtered:
            actions = filtered
        actions = sorted(
            actions,
            key=lambda action: self.getEvaluationFunction()(
                gameState.generateSuccessor(0, action)
            ),
            reverse=True
        )
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 0, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: This evaluation function considers several key features:
    1. Current game score
    2. Distance to closest food
    3. Number of remaining food pellets
    4. Ghost positions and states (scared vs normal)
    5. Number of power pellets remaining

    Each feature is weighted and combined into a final score, with:
    - Higher weight for being close to food when there's less food remaining
    - Positive weight for being near scared ghosts
    - Negative weight for being near active ghosts
    - Bonus for power pellets when ghosts are nearby
    """
    pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()

    # Food information
    food = currentGameState.getFood()
    foodList = food.asList()
    numFood = len(foodList)

    # Ghost information
    ghostStates = currentGameState.getGhostStates()
    scaredGhosts = [g for g in ghostStates if g.getScaredTimer() > 0]
    activeGhosts = [g for g in ghostStates if g.getScaredTimer() <= 0]

    # Power pellet information
    capsules = currentGameState.getCapsules()
    numCapsules = len(capsules)

    def manhattanDistance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # Calculate distance to closest food
    if foodList:
        closestFoodDist = min(manhattanDistance(pos, food) for food in foodList)
        foodScore = 10.0 / (closestFoodDist + 1) * (1 + 1.0 / (numFood + 1))
    else:
        foodScore = 100  # Big bonus for clearing all food

    # Ghost influence
    ghostScore = 0
    for ghost in activeGhosts:
        dist = manhattanDistance(pos, ghost.getPosition())
        if dist < 2:  # Extremely close ghost
            ghostScore -= 500
        else:
            ghostScore -= 100 / (dist + 1)

    # Scared ghost bonus
    scaredScore = 0
    for ghost in scaredGhosts:
        dist = manhattanDistance(pos, ghost.getPosition())
        scaredScore += 200 / (dist + 1)

    # Power pellet consideration
    capsuleScore = 0
    if numCapsules > 0 and activeGhosts:
        minGhostDistance = min(manhattanDistance(pos, ghost.getPosition())
                               for ghost in activeGhosts)
        if minGhostDistance < 4:  # Ghosts nearby
            capsuleScore = 100 / (
                min(manhattanDistance(pos, capsule) for capsule in capsules) + 1
            )

    finalScore = score + foodScore + ghostScore + scaredScore + capsuleScore
    return finalScore


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.
    Ghosts don't behave randomly anymore, but they aren't perfect either --
    they'll usually just make a beeline straight towards Pac-Man (or away if they're scared!)
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
