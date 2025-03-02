"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    For the Bridge Grid problem, we need to change either discount or noise
    so that the agent will cross the bridge.
    
    Lowering the noise to 0 makes the environment deterministic, removing
    the risk of falling into negative terminal states when crossing the bridge.
    """
    answerDiscount = 0.9
    answerNoise = 0.0

    return answerDiscount, answerNoise

def question3a():
    """
    Prefer the close exit (+1), risking the cliff (-10).
    
    We use:
    - Low discount to prioritize immediate rewards
    - No noise for deterministic movements
    - Negative living reward to encourage reaching exits quickly
    """
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Prefer the close exit (+1), but avoiding the cliff (-10).
    
    We use:
    - Low discount to prioritize immediate rewards
    - Some noise to make the cliff risky and encourage the safer path
    - Small negative living reward to encourage reaching exits
    """
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Prefer the distant exit (+10), risking the cliff (-10).
    
    We use:
    - High discount to value the distant high reward
    - No noise for deterministic movements
    - Small negative living reward to encourage movement
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Prefer the distant exit (+10), avoiding the cliff (-10).
    
    We use:
    - High discount to value the distant high reward
    - Some noise to make the cliff risky and encourage the safer path
    - Small negative living reward to encourage movement
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Avoid both exits and the cliff.
    
    We use:
    - High discount to value future states
    - Low noise for more control
    - Positive living reward to encourage staying alive rather than reaching exits
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 10.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    For BridgeGrid with Q-learning, we need parameters that ensure
    the optimal policy is learned reliably (>99%) within 50 iterations.
    
    We use:
    - Low epsilon for less random exploration
    - High learning rate to learn quickly from limited experiences
    """
    answerEpsilon = 0.1
    answerLearningRate = 0.9

    return answerEpsilon, answerLearningRate

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))