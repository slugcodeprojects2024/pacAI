from pacai.agents.capture.capture import CaptureAgent
from pacai.core.directions import Directions
import random

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
    def __init__(self, index, timeForComputing = 0.1):
        super().__init__(index, timeForComputing)
        self.last_position = None
        self.stuck_count = 0
        self.last_food_eaten = None
        self.same_food_count = 0

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if len(actions) == 0:
            return None

        myPos = gameState.getAgentPosition(self.index)
        
        # Enhanced stuck detection
        if self.last_position == myPos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.last_position = myPos

        # Get enemies state
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
        scared_ghosts = [g for g in ghosts if g.isScared()]

        # Get food and capsule locations
        food = self.getFood(gameState).asList()
        capsules = self.getCapsules(gameState)

        # Track food carrying - but don't use it to stop collecting
        food_carrying = (
            len(self.getFoodYouAreDefending(gameState).asList())
            - len(self.getFood(gameState).asList())
        )

        # Return home only if in danger or carrying a lot of food
        if (food_carrying >= 6  # Increased threshold
                or (food_carrying > 0 and any(
                    g for g in ghosts if self.getMazeDistance(myPos, g.getPosition()) < 3))):
            return self.getReturnHomeAction(gameState, actions, myPos, ghosts)

        # If we're stuck, try to move to a new position
        if self.stuck_count > 3:
            self.stuck_count = 0
            valid_actions = [a for a in actions if a != Directions.STOP]
            if valid_actions:
                return random.choice(valid_actions)

        # Handle ghost avoidance
        if ghosts and not scared_ghosts:
            ghost_dists = [self.getMazeDistance(myPos, g.getPosition()) for g in ghosts]
            if min(ghost_dists) < 3:
                return self.getEscapeAction(gameState, actions, myPos, ghosts)

        # Go for power capsule if ghosts are near and we're in dangerous territory
        if capsules and ghosts:
            ghost_dists = [self.getMazeDistance(myPos, g.getPosition()) for g in ghosts]
            if min(ghost_dists) < 6:  # Increased range for capsule consideration
                capsule_dists = [self.getMazeDistance(myPos, caps) for caps in capsules]
                closest_capsule = capsules[capsule_dists.index(min(capsule_dists))]
                return self.getActionToTarget(gameState, actions, myPos, closest_capsule)

        # Always try to get more food if we can do so safely
        if food:
            food_distances = [(f, self.getMazeDistance(myPos, f)) for f in food]
            safe_foods = [(f, d) for f, d in food_distances
                         if not any(self.getMazeDistance(f, g.getPosition()) < 2
                                  for g in ghosts if not g.isScared())]
            
            if safe_foods:  # If there's safe food, go for the closest one
                target_food = min(safe_foods, key=lambda x: x[1])[0]
                return self.getActionToTarget(gameState, actions, myPos, target_food)
            elif food_distances:  # If no safe food, try for closest food if not too dangerous
                closest_food = min(food_distances, key=lambda x: x[1])[0]
                return self.getActionToTarget(gameState, actions, myPos, closest_food)

        # If no clear goal, make a random move (avoid stopping)
        valid_actions = [a for a in actions if a != Directions.STOP]
        return random.choice(valid_actions if valid_actions else actions)

    def getActionToTarget(self, gameState, actions, myPos, targetPos):
        """Get best action to reach a target position."""
        best_dist = float('inf')
        best_action = random.choice(actions)
        
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(new_pos, targetPos)
            
            if dist < best_dist:
                best_dist = dist
                best_action = action
                
        return best_action
        
    def getRetreatAction(self, gameState, actions, myPos, enemies):
        """Get best action to retreat from enemies while staying in defensive position."""
        best_score = float('-inf')
        best_action = random.choice(actions)
        
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            
            # Calculate minimum distance to any enemy
            enemy_dists = [self.getMazeDistance(new_pos, e.getPosition()) for e in enemies]
            min_enemy_dist = min(enemy_dists)
            
            # Calculate distance to defensive position
            defense_dist = self.getMazeDistance(new_pos, self.target_position)
            
            # Score favors positions that are:
            # 1. Further from enemies when scared
            # 2. Not too far from defensive position
            score = min_enemy_dist - (defense_dist * 0.5)
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def getEscapeAction(self, gameState, actions, myPos, ghosts):
        """Get best action to escape from ghosts."""
        best_score = float('-inf')
        best_action = random.choice(actions)
        
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            
            # Score based on ghost distances and direction to home
            ghost_distances = [self.getMazeDistance(new_pos, g.getPosition()) for g in ghosts]
            min_ghost_dist = min(ghost_distances)
            home_dist = self.getMazeDistance(new_pos, self.start)
            
            # Prefer positions further from ghosts and closer to home
            score = min_ghost_dist - (home_dist * 0.5)
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def getReturnHomeAction(self, gameState, actions, myPos, ghosts=[]):
        """Get best action to return to our side."""
        best_score = float('-inf')
        best_action = random.choice(actions)
        
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            home_dist = self.getMazeDistance(new_pos, self.start)
            
            # Consider ghost positions in scoring
            ghost_penalty = 0
            if ghosts:
                min_ghost_dist = min(
                    self.getMazeDistance(new_pos, g.getPosition())
                    for g in ghosts
                )
                ghost_penalty = -10 if min_ghost_dist < 2 else 0
            
            # Score favors positions closer to home and away from ghosts
            score = -home_dist + ghost_penalty
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

class DefensiveAgent(CaptureAgent):
    def __init__(self, index, timeForComputing = 0.1):
        super().__init__(index, timeForComputing)
        self.target_position = None

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

        # Calculate defensive position based on map analysis
        walls = gameState.getWalls()
        self.mapWidth = walls._width
        self.mapHeight = walls._height
        
        # Find the best defensive position by analyzing the map
        self.target_position = self.findDefensivePosition(gameState)

    def findDefensivePosition(self, gameState):
        """Find a good defensive position based on map layout."""
        walls = gameState.getWalls()
        mid_x = (self.mapWidth - 2) // 2
        
        # For red team, we want to be slightly on our side
        defend_x = mid_x - 2
        
        # Look for a position that has good visibility and mobility
        best_pos = None
        best_score = float('-inf')
        
        # Check positions around the vertical middle
        mid_y = (self.mapHeight - 2) // 2
        search_range = min(5, mid_y)  # Don't search too far from middle
        
        for y_offset in range(-search_range, search_range + 1):
            test_y = mid_y + y_offset
            if test_y < 1 or test_y >= self.mapHeight - 1:
                continue
                
            # Try positions at different x coordinates
            for x_offset in [-1, 0, 1]:
                test_x = defend_x + x_offset
                if test_x < 1 or test_x >= self.mapWidth - 1:
                    continue
                    
                if not walls[test_x][test_y]:
                    # Score this position based on several factors
                    score = self.evaluateDefensivePosition(gameState, (test_x, test_y))
                    if score > best_score:
                        best_score = score
                        best_pos = (test_x, test_y)
        
        # If no good position found, fall back to simple middle position
        if best_pos is None:
            for y in range(self.mapHeight):
                if not walls[defend_x][y]:
                    best_pos = (defend_x, y)
                    break
                    
        return best_pos

    def evaluateDefensivePosition(self, gameState, pos):
        """Score a potential defensive position."""
        walls = gameState.getWalls()
        score = 0
        
        # Prefer positions with more open adjacent squares (more mobility)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            test_x = pos[0] + dx
            test_y = pos[1] + dy
            if 0 <= test_x < self.mapWidth and 0 <= test_y < self.mapHeight:
                if not walls[test_x][test_y]:
                    score += 1
                    
        # Prefer positions closer to the middle height
        mid_y = self.mapHeight // 2
        score -= abs(pos[1] - mid_y) * 0.5
        
        # Prefer positions not too close to walls
        if any(not walls[pos[0] + dx][pos[1] + dy]
               for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
               if 0 <= pos[0] + dx < self.mapWidth and 0 <= pos[1] + dy < self.mapHeight):
            score += 2
            
        return score

    def getRetreatAction(self, gameState, actions, myPos, enemies):
        """Get best action to retreat while maintaining defensive presence."""
        best_score = float('-inf')
        best_action = random.choice(actions)
        
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            
            # Evaluate retreat position
            # Consider: distance from enemies, distance to defensive position,
            # and whether position maintains good defensive coverage
            enemy_distances = [self.getMazeDistance(new_pos, e.getPosition())
                             for e in enemies]
            min_enemy_dist = min(enemy_distances)
            
            # Distance to defensive position
            defense_dist = self.getMazeDistance(new_pos, self.target_position)
            
            # Score based on safety and defensive utility
            score = min_enemy_dist - (defense_dist * 0.5)
            
            # Bonus for positions closer to center
            mid_x = self.mapWidth // 2
            if abs(new_pos[0] - mid_x) < abs(myPos[0] - mid_x):
                score += 2
                
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if len(actions) == 0:
            return None

        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        
        # Get positions of all visible enemies, whether they're Pacman or not
        visible_enemies = [a for a in enemies if a.getPosition() is not None]
        
        # Check if we're scared
        my_state = gameState.getAgentState(self.index)
        is_scared = my_state.isScared()
        
        if visible_enemies:
            enemy_dists = [self.getMazeDistance(myPos, a.getPosition()) for a in visible_enemies]
            closest_enemy = visible_enemies[enemy_dists.index(min(enemy_dists))]
            closest_dist = min(enemy_dists)
            
            # Define the border x-coordinate (middle of the map)
            border_x = self.mapWidth // 2
            
            # If enemy is on our side (left half for red team)
            enemy_pos = closest_enemy.getPosition()
            if enemy_pos[0] < border_x:
                # If we're not scared or if the enemy is very close, chase them
                if not is_scared or closest_dist <= 2:
                    return self.getActionToTarget(gameState, actions, myPos, enemy_pos)
                # If we are scared, retreat to a safe position
                else:
                    return self.getRetreatAction(gameState, actions, myPos, visible_enemies)
            
            # More aggressive interception
            elif closest_dist <= 6 and enemy_pos[0] <= border_x + 3:
                # Move to intercept earlier and ensure coordinates are integers
                intercept_x = int(min(enemy_pos[0] - 1, border_x - 1))  # Convert to int
                intercept_y = int(enemy_pos[1])
                intercept_pos = (intercept_x, intercept_y)
                
                # Verify the position is valid (not a wall)
                walls = gameState.getWalls()
                if not walls[int(intercept_pos[0])][int(intercept_pos[1])]:
                    return self.getActionToTarget(gameState, actions, myPos, intercept_pos)
                
                # If wall, try positions above and below
                for y_offset in [1, -1, 2, -2]:
                    test_y = int(intercept_y + y_offset)
                    test_x = int(intercept_x)
                    if 0 <= test_y < self.mapHeight and not walls[test_x][test_y]:
                        return self.getActionToTarget(gameState, actions, myPos, (test_x, test_y))
                
                # Verify the position is valid (not a wall)
                walls = gameState.getWalls()
                if not walls[intercept_pos[0]][intercept_pos[1]]:
                    return self.getActionToTarget(gameState, actions, myPos, intercept_pos)
                
                # If wall, try positions above and below
                for y_offset in [1, -1, 2, -2]:
                    test_pos = (border_x - 1, intercept_y + y_offset)
                    if 0 <= test_pos[1] < self.mapHeight and not walls[test_pos[0]][test_pos[1]]:
                        return self.getActionToTarget(gameState, actions, myPos, test_pos)
                        
        # If no valid intercept position or no enemies in sight, return to defensive position
        return self.getActionToTarget(gameState, actions, myPos, self.target_position)

        # Otherwise, maintain defensive position
        return self.getActionToTarget(gameState, actions, myPos, self.target_position)

    def getActionToTarget(self, gameState, actions, myPos, targetPos):
        """Get best action to reach a target position."""
        best_dist = float('inf')
        best_action = random.choice(actions)
        
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(new_pos, targetPos)
            
            if dist < best_dist:
                best_dist = dist
                best_action = action
                
        return best_action