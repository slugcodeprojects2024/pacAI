"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first (DFS).
    Returns a list of actions that reaches the goal.
    """
    # A Stack for frontier states; each element is (state, path_taken).
    frontier = Stack()
    start_state = problem.startingState()
    frontier.push((start_state, []))
    
    visited = set()  # Keep track of visited states to avoid revisiting
    
    while not frontier.isEmpty():
        state, path = frontier.pop()
        
        # If this state is the goal, return the path (actions) to get here
        if problem.isGoal(state):
            return path
        
        if state not in visited:
            visited.add(state)
            
            # Expand the node: get successors, push them if unvisited
            for successor, action, _ in problem.successorStates(state):
                if successor not in visited:
                    frontier.push((successor, path + [action]))
    
    # If somehow no solution is found (should not happen if problem is solvable)
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first (BFS).
    Returns a list of actions that reaches the goal.
    """
    # A Queue for frontier states; each element is (state, path_taken).
    frontier = Queue()
    start_state = problem.startingState()
    frontier.push((start_state, []))
    
    visited = set()  # Keep track of visited states to avoid revisiting
    
    while not frontier.isEmpty():
        state, path = frontier.pop()
        
        # If this state is the goal, return the path (actions) to get here
        if problem.isGoal(state):
            return path
        
        if state not in visited:
            visited.add(state)
            
            # Expand the node: get successors, enqueue them if unvisited
            for successor, action, _ in problem.successorStates(state):
                if successor not in visited:
                    frontier.push((successor, path + [action]))
    
    # If no solution is found (should not happen if problem is solvable)
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    # Priority queue of (state, path, total_cost); priority is total_cost
    frontier = PriorityQueue()
    start_state = problem.startingState()
    frontier.push((start_state, [], 0), 0)
    
    visited = {}  # state -> lowest cost to reach this state so far
    
    while not frontier.isEmpty():
        state, path, cost = frontier.pop()
        
        # If we found the goal, return the path
        if problem.isGoal(state):
            return path
        
        # Only expand if we haven't found a cheaper path to this state
        if state not in visited or cost < visited[state]:
            visited[state] = cost
            
            # Add successors to frontier with updated costs
            for successor, action, step_cost in problem.successorStates(state):
                new_cost = cost + step_cost
                if successor not in visited or new_cost < visited[successor]:
                    frontier.push((successor, path + [action], new_cost), new_cost)
    
    # No solution found
    return []

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    # Priority queue of (state, path, g_cost); priority is f_cost = g_cost + h_cost
    frontier = PriorityQueue()
    start_state = problem.startingState()
    frontier.push((start_state, [], 0), heuristic(start_state, problem))
    
    visited = {}  # state -> lowest g_cost to reach this state so far
    
    while not frontier.isEmpty():
        state, path, g_cost = frontier.pop()
        
        # If we found the goal, return the path
        if problem.isGoal(state):
            return path
        
        # Only expand if we haven't found a cheaper path to this state
        if state not in visited or g_cost < visited[state]:
            visited[state] = g_cost
            
            # Add successors to frontier with updated costs
            for successor, action, step_cost in problem.successorStates(state):
                new_g_cost = g_cost + step_cost
                if successor not in visited or new_g_cost < visited[successor]:
                    # f_cost = g_cost + h_cost
                    f_cost = new_g_cost + heuristic(successor, problem)
                    frontier.push((successor, path + [action], new_g_cost), f_cost)
    
    # No solution found
    return []