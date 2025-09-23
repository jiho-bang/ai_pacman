# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodLst = newFood.asList()
        closestGhost = 999999
        for g in newGhostStates:
            if g.scaredTimer == 0:
                closestGhost = min(closestGhost, manhattanDistance(newPos, g.configuration.pos))

        if closestGhost == 0:
            return -999999

        closestFood = 999999
        if currentGameState.getFood()[newPos[0]][newPos[1]]:
            closestFood = 0
            score = (-(1 / (closestGhost + 0.5)) + 1 / (closestFood + 0.5))
            return score

        for f in foodLst:
            closestFood = min(closestFood, manhattanDistance(newPos, f))

        score = (-(1 / (closestGhost + 0.5)) + 1 / (closestFood + 0.5))
        #print(closestFood)
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        states = [gameState.generateSuccessor(0, action) for action in actions]
        minimax = [self.minimize(state, 0, 1) for state in states]
        best = [x for x in range(len(minimax)) if max(minimax) == minimax[x]]
        return actions[best[0]]

        util.raiseNotDefined()

    def minimize(self, gameState, depth, agentIndex):
        ghosts = gameState.getNumAgents() - 1
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        states = [gameState.generateSuccessor(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]

        if agentIndex == ghosts:
            best = [self.maximize(state, depth+1) for state in states]
        else:
            best = [self.minimize(state, depth, agentIndex+1) for state in states]
        return min(best)
    
    def maximize(self, gameState, depth):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        states = [gameState.generateSuccessor(0, action) for action in actions]
        best = [self.minimize(state, depth, 1) for state in states]
        return max(best)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        beta = float("inf")
        score = float("-inf")
        actions = gameState.getLegalActions(0)
        #states = [gameState.generateSuccessor(0, action) for action in actions]
        move = 0
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            tempScore = self.min_value(state, 0, alpha, beta, 1)
            if tempScore > score:
                score = tempScore
                move = action
            alpha = max(alpha, tempScore)
        return move
        util.raiseNotDefined()

    def max_value(self, gameState, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = float("-inf")
        actions = gameState.getLegalActions(0)
        #states = [gameState.generateSuccessor(0, action) for action in actions]
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            v = max(v, self.min_value(state, depth, alpha, beta, 1))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def min_value(self, gameState, depth, alpha, beta, agentIndex):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = float("inf")
        #actions = gameState.getLegalActions(agentIndex)
        #states = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        for action in gameState.getLegalActions(agentIndex):
            ghosts = gameState.getNumAgents() - 1
            state = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == ghosts:
                v = min(v, self.max_value(state, depth+1, alpha, beta))
            else:
                v = min(v, self.min_value(state, depth, alpha, beta, agentIndex+1))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        actions = gameState.getLegalActions(0)
        states = [gameState.generateSuccessor(0, action) for action in actions]
        expectimax = [self.probability(state, 0, 1) for state in states]
        best = [x for x in range(len(expectimax)) if max(expectimax) == expectimax[x]]
        return actions[best[0]]
        util.raiseNotDefined()

    def expectimax(self, gameState, depth):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        states = [gameState.generateSuccessor(0, action) for action in actions]
        best = [self.probability(state, depth, 1) for state in states]
        return max(best)
    
    def avg(self, arr):
        return sum(arr) / len(arr)
    
    def probability(self, gameState, depth, agentIndex):
        ghosts = gameState.getNumAgents() - 1
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        states = [gameState.generateSuccessor(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]

        if agentIndex == ghosts:
            best = [self.expectimax(state, depth+1) for state in states]
        else:
            best = [self.probability(state, depth, agentIndex+1) for state in states]
        return self.avg(best)
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    ghostStates= currentGameState.getGhostStates()
    #scaredTimes = [ghostStates.scaredTimer for ghost in ghostStates]

    closestGhost = 999999
    for g in ghostStates:
        if g.scaredTimer == 0:
            closestGhost = min(closestGhost, manhattanDistance(pos, g.configuration.pos))
        else:
            closestGhost = -10

    if closestGhost == 0:
        return -999999

    closestFood = 999999
    if currentGameState.getFood()[pos[0]][pos[1]]:
        closestFood = 0
        score = (-(1 / (closestGhost + 0.5)) + 1 / (closestFood + 0.5))
        return score
    if not foodList:
        closestFood = 0
    for f in foodList:
        closestFood = min(closestFood, manhattanDistance(pos, f))
    score = currentGameState.getScore() -(1 / (closestGhost + 0.5)) + 1 / (closestFood + 0.5)
    return score

# Abbreviation
better = betterEvaluationFunction
