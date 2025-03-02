from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the values for each state.

        # Perform value iteration for the specified number of iterations
        for i in range(self.iters):
            # Create a new value dictionary for updating (to avoid affecting the current iteration)
            new_values = self.values.copy()
            
            # For each state in the MDP
            for state in self.mdp.getStates():
                # Skip terminal states as they have a value of 0
                if self.mdp.isTerminal(state):
                    continue
                
                # Get all possible actions from this state
                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    continue
                
                # Find the maximum Q-value across all possible actions
                max_q_value = max(self.getQValue(state, action) for action in actions)
                
                # Update the value of this state
                new_values[state] = max_q_value
            
            # Update our values for the next iteration
            self.values = new_values

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values.get(state, 0.0)

    def getQValue(self, state, action):
        """
        The q-value of the state action pair.
        Q(s,a) = Σ_s' T(s,a,s') * [R(s,a,s') + γV(s')]
        """
        q_value = 0.0
        
        # For each possible next state and associated probability
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # Get the reward for this transition
            reward = self.mdp.getReward(state, action, nextState)
            
            # Get the current value of the next state
            next_value = self.getValue(nextState)
            
            # Sum up weighted by transition probability: T(s,a,s') * [R(s,a,s') + γV(s')]
            q_value += prob * (reward + self.discountRate * next_value)
            
        return q_value

    def getPolicy(self, state):
        """
        The policy is the best action in the given state according to the values.
        If there are no legal actions, return None.
        """
        if self.mdp.isTerminal(state):
            return None
            
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
            
        # Find the action with the maximum Q-value
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            q_value = self.getQValue(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
                
        return best_action

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """
        return self.getPolicy(state)