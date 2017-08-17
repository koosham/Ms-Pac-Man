#!python
r"""GPac - EA Search"""
__author__ = 'Koosha Marashi'
__email__ = 'km89f@mst.edu'

#import used packages
import random
import json
import time
import argparse
import os
import sys
import copy
from decimal import Decimal
import math
import functools
import operator

#Get DIRECTORY of this file. Makes compilation more flexible.
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
NUMBER_OF_GHOSTS = 3
GAME_CONTINUES = 0
GAME_OVER = 1
EMPTY = 0
PILL = 1
WALL = 2
MAX_TREE_DEPTH = 3
FUNCTIONS = ['+', '-', '*', '/', 'rand']
DISTANCE_TO_NEAREST_GHOST = 'G'
DISTANCE_TO_PACMAN = 'M'
DISTANCE_TO_NEAREST_PILL = 'P'
CONSTANT = 'C'
PACMAN_TERMINALS = [DISTANCE_TO_NEAREST_GHOST,DISTANCE_TO_NEAREST_PILL,CONSTANT]
GHOSTS_TERMINALS = [DISTANCE_TO_PACMAN,DISTANCE_TO_NEAREST_GHOST,CONSTANT]
CONSTANT_MIN = -20
CONSTANT_MAX = 20
PRINT = True
#Allowed movements of the units [x,y]
PACMAN_MOVES = [[0,0],[0,1],[0,-1],[1,0],[-1,0]]
GHOST_MOVES = [[0,1],[0,-1],[1,0],[-1,0]]

def deepgetattr(obj,attr):
    """Recurses through an attribute chain to get the ultimate value."""
    return functools.reduce(getattr,attr.split('.'),obj)

def deepsetattr(obj,attr,val):
    """Recurses through an attribute chain to set the ultimate value."""
    pre, _, post = attr.rpartition('.')
    return setattr(deepgetattr(obj, pre) if pre else obj, post, val)

def nodeParent(nodeID):
    """Return ID of parent of a node"""
    if nodeID > 1:
        return int((nodeID - 1) / 2)
    else:
        return 0

def nodeLeftChild(nodeID):
    """Return ID of left child of a node"""
    return (nodeID * 2) + 1

def nodeRightChild(nodeID):
    """Return ID of right child of a node"""
    return (nodeID * 2) + 2

def totalNumberOfNodes(height):
    """Return total number of nodes in a tree with given height"""
    return 2 ** (height + 1) - 1

class Config:
    """Handle configurations"""

    def parseCfg(self,cfgFilePath):
        """Read JSON formatted .cfg file and import configurations"""
        with open(cfgFilePath) as cfgFile:    
            config = json.load(cfgFile)
        #Key: Description
        #output folder: Relative path of output files
        #log file: Name of log file to write
        #highest score game world file: Name of wld file to write the highest-score-game-sequence all-time-step world file
        #pacman - solution file: Name of solution file to write the best solution found for Ms. PacMan's controller
        #ghosts - solution file: Name of solution file to write the best solution found for the Ghosts' controller
        #random generator seed type: Type of seed used for random number generator. Should be 'time' to use system epoch time in milliseconds, or 'int' for manually feeding the seed.
        #random generator seed: If randomSeedType is equal to 'int' then would be used as seed of random number generator
        #number of runs: Total number of runs
        #number of evaluations: Number of fitness evaluations per run
        #world width: number of columns in the grid of the world 
        #world height: number of rows in the grid of the world
        #pill density percentage: cell-wise probability of having a pill
        #pacman - population size
        #pacman - number of offspring
        #pacman - parent selection method: Can be set to 'fps' for Fitness Proportional Selection or 'over-selection'
        #pacman - proportion of population in fitter group: p in over-selection method
        #pacman - survival selection method: Can be set to 'truncation' or 'k-tournament w/o replacement'
        #pacman - tournament size for survival selection: Value of k
        #pacman - mutation probability: Mutation rate for sub-tree mutation
        #pacman - parsimony pressure penalty coefficient: coefficient (c) of penalty function which is c x height(tree)
        #ghosts - population size
        #ghosts - number of offspring
        #ghosts - parent selection method: Can be set to 'fps' for Fitness Proportional Selection or 'over-selection'
        #ghosts - proportion of population in fitter group: p in over-selection method
        #ghosts - survival selection method: Can be set to 'truncation' or 'k-tournament w/o replacement'
        #ghosts - tournament size for survival selection: Value of k
        #ghosts - mutation probability: Mutation rate for sub-tree mutation
        #ghosts - parsimony pressure penalty coefficient: coefficient (c) of penalty function which is c x height(tree)
        
        if config['random generator seed type'] == 'time':
            #Get system epoch time
            config['random generator seed'] = round(time.time() * 1000)
        elif config['random generator seed type'] == 'int':
            #Use manually provided seed
            pass
        return config

class WriteToFile(object):
    """Write outputs to file"""
    def __init__(self,config):
        self.config = config
    def writeHeaderToLog(self):
        """Overwrite header information to log file"""
        logFileObject = open(os.path.join(DIRECTORY,self.config['output folder'],self.config['log file']),'w')
        logFileObject.write('World grid size: W = ' + str(Decimal(self.config['world width'])) + ', H = ' + str(Decimal(self.config['world width'])) + '\n')
        logFileObject.write('Pill density: ' + str(Decimal(self.config['pill density percentage'])) + '%' + '\n')
        logFileObject.write('Random number generator seed: ' + str(Decimal(self.config['random generator seed'])) + '\n')
        logFileObject.write('Number of runs: ' + str(Decimal(self.config['number of runs'])) + '\n')
        logFileObject.write('Number of fitness evaluations per run: ' + str(Decimal(self.config['number of evaluations'])) + '\n')
        logFileObject.write('PacMan population Size: ' + str(Decimal(self.config['pacman - population size'])) + '\n')
        logFileObject.write('PacMan offspring size: ' + str(Decimal(self.config['pacman - number of offspring'])) + '\n')
        logFileObject.write('PacMan parent selection method: ' + str(self.config['pacman - parent selection method']) + '\n')
        if self.config['pacman - parent selection method'] == 'over-selection':
            logFileObject.write('Proportion of PacMan population in fitter group: ' + str(round(self.config['pacman - proportion of population in fitter group'],2)) + '\n')
        logFileObject.write('PacMan survival selection method: ' + str(self.config['pacman - survival selection method']) + '\n')
        if self.config['pacman - survival selection method'] == 'k-tournament w/o replacement':
            logFileObject.write('Tournament size for PacMan survival selection: ' + str(Decimal(self.config['pacman - tournament size for survival selection'])) + '\n')
        logFileObject.write('PacMan mutation probability: ' + str(round(self.config['pacman - mutation probability'],4)) + '\n')
        logFileObject.write('PacMan parsimony pressure penalty coefficient: ' + str(round(self.config['pacman - parsimony pressure penalty coefficient'],1)) + '\n')
        logFileObject.write('Ghosts population Size: ' + str(Decimal(self.config['ghosts - population size'])) + '\n')
        logFileObject.write('Ghosts offspring size: ' + str(Decimal(self.config['ghosts - number of offspring'])) + '\n')
        logFileObject.write('Ghosts parent selection method: ' + str(self.config['ghosts - parent selection method']) + '\n')
        if self.config['ghosts - parent selection method'] == 'over-selection':
            logFileObject.write('Proportion of ghosts population in fitter group: ' + str(round(self.config['ghosts - proportion of population in fitter group'],2)) + '\n')
        logFileObject.write('Ghosts survival selection method: ' + str(self.config['ghosts - survival selection method']) + '\n')
        if self.config['ghosts - survival selection method'] == 'k-tournament w/o replacement':
            logFileObject.write('Tournament size for ghosts survival selection: ' + str(Decimal(self.config['ghosts - tournament size for survival selection'])) + '\n')
        logFileObject.write('Ghosts mutation probability: ' + str(round(self.config['ghosts - mutation probability'],4)) + '\n')
        logFileObject.write('Ghosts parsimony pressure penalty coefficient: ' + str(round(self.config['ghosts - parsimony pressure penalty coefficient'],1)) + '\n')
        logFileObject.close()
        
    def writeResultsLabelToLog(self):
        """Append results header to log file"""
        logFileObject = open(os.path.join(DIRECTORY,self.config['output folder'],self.config['log file']),'a')
        logFileObject.write('\nResult Log\n')
        logFileObject.close()
    
    def writeRunLabelToLog(self,run):
        """Append run label to log file"""
        logFileObject = open(os.path.join(DIRECTORY,self.config['output folder'],self.config['log file']),'a')
        logFileObject.write('\nRun ' + str(run) + '\n')
        logFileObject.close()
        
    def writeResultsToLog(self,eval,*args):
        """Append results to log file"""
        logFileObject = open(os.path.join(DIRECTORY,self.config['output folder'],self.config['log file']),'a')
        logFileObject.write(str(eval))
        for arg in args:
            logFileObject.write('\t' + str(round(arg,2)))
        logFileObject.write('\n')
        logFileObject.close()
        
    def writeToWorldFile(self,pacmanSeq,ghostsSeq,pills):
        """Overwrite final results of the highest score game to wld file"""
        logFileObject = open(os.path.join(DIRECTORY,self.config['output folder'],self.config['highest score game world file']),'w')
        totalTime = 2 * self.config['world width'] * self.config['world height']
        logFileObject.write(str(Decimal(self.config['world width'])) + '\n')
        logFileObject.write(str(Decimal(self.config['world height'])) + '\n')
        for index in range(len(pacmanSeq)):
            logFileObject.write('m ' + str(pacmanSeq[index].x) + ' ' + str(pacmanSeq[index].y) + '\n')
            for i,ghost in enumerate(ghostsSeq[index]):
                logFileObject.write(str(i + 1) + ' ' + str(ghost.x) + ' ' + str(ghost.y) + '\n')
            if index == 0:
                for pill in pills:
                    logFileObject.write('p ' + str(pill[0]) + ' ' + str(pill[1]) + '\n')
            logFileObject.write('t ' + str(totalTime - index) + ' ' + str(pacmanSeq[index].score) + '\n')
        logFileObject.close()
    
    def writeToSolutionFile(self,solution):
        """Overwrite final results to solution file"""
        if solution.type == "Ms. PacMan":
            logFileObject = open(os.path.join(DIRECTORY,self.config['output folder'],self.config['pacman - solution file']),'w')
        elif solution.type == "Ghost": 
            logFileObject = open(os.path.join(DIRECTORY,self.config['output folder'],self.config['ghosts - solution file']),'w')
        logFileObject.write(str(solution))
        logFileObject.close()

class Pacman(object):
    """Object of Ms. Pac-Man"""
    def __init__(self,world,config,**kwargs):
        self.x = 0
        self.y = config['world height'] - 1
        self.score = 0
        for key, value in kwargs.items():
            if key == 'x':
                self.x = value
            elif key == 'y':
                self.y = value
            elif key == 'score':
                self.score = value
    
    def move(self,world,direction,config):
        """Move Ms. Pac-man"""
        self.x += direction[0]
        self.y += direction[1]
        #Only for safety. Controller should not make off-the-grid movements at all.
        self.x = max(0,min(self.x,config['world width'] - 1))
        self.y = max(0,min(self.y,config['world height'] - 1))

class Ghost(object):
    """Object of the ghosts"""
    def __init__(self,world,config,**kwargs):
        self.x = config['world width'] - 1
        self.y = 0
        self.score = 0
        for key, value in kwargs.items():
            if key == 'x':
                self.x = value
            elif key == 'y':
                self.y = value
            elif key == 'score':
                self.score = value
    
    def move(self,world,direction,config):
        """Move the ghosts"""
        self.x += direction[0]
        self.y += direction[1]
        #Only for safety. Controller should not make off-the-grid movements at all.
        self.x = max(0,min(self.x,config['world width'] - 1))
        self.y = max(0,min(self.y,config['world height'] - 1))

class World(object):
    """object of the world grid"""
    def __init__(self,config):
        self.grid = [[EMPTY for x in range(config['world height'])] for x in range(config['world width'])]
        self.nInitialPills = 0
        if config['pill density percentage'] != 0:
            while self.nInitialPills == 0:
                for i in range(config['world width']):
                    for j in range(config['world height']):
                        if random.random() < (config['pill density percentage'] / 100):
                            self.grid[i][j] = PILL
                            self.nInitialPills += 1
        self.nRemainingPills = self.nInitialPills

class GPTree(object):
    """Objects of genetic programming binary trees"""
    def __init__(self,type,value = None):
        self.type = type #Specifies whether the GP tree belongs to a "Ms. PacMan" or a "Ghost" individual
        self.value = value
        self.left = None
        self.right = None
        self.fitness = 0
        self.evals = 0
        self.normalizedFitness = 0
    
    def addLeftChild(self,value = None):
        """Add left child to the current node"""        
        self.left = GPTree(self.type,value)
    
    def addRightChild(self,value = None):
        """Add right child to the current node"""
        self.right = GPTree(self.type,value)
    
    def getNodeByLocation(self,depth,index):
        """Return the node specified by depth and index"""
        if depth is 0:
            return self
        next = self
        for d in reversed(range(1,depth + 1)):
            if next is not None:
                if index < (2 ** (d - 1)):
                    next = next.left
                else:
                    next = next.right
                index = index % (2 ** (d - 1))
            else:
                return None
        return next
    
    def getNodeByID(self,id):
        """Return the node specified by its ID"""
        depth = int(math.log2(id + 1))
        index = id - (2 ** depth) + 1
        return self.getNodeByLocation(depth,index)
    
    def setNodeByLocation(self,depth,index,node):
        """Set the node specified by depth and index"""
        if depth is 0:
            self = node
            return 1
        attr = ''
        next = self
        for d in reversed(range(1,depth + 1)):
            if next is not None:
                if index < (2 ** (d - 1)):
                    next = next.left
                    attr += 'left.'
                else:
                    next = next.right
                    attr += 'right.'
                index = index % (2 ** (d - 1))
                if d is 1:
                    deepsetattr(self,attr[:-1],node)
                    return 1
            else:
                return 0
        return 1
    
    def setNodeByID(self,id,node):
        """Return the node specified by its ID"""
        depth = int(math.log2(id + 1))
        index = id - (2 ** depth) + 1
        return self.setNodeByLocation(depth,index,node)
    
    def isNodeNoneByLocation(self,depth,index):
        """Return True/False if the node specified by depth and index is/is not None"""
        node = self.getNodeByLocation(depth,index)
        if node is None:
            return True
        elif node.value is None:
            return True
        else:
            return False
    
    def isNodeNoneByID(self,id):
        """Return True/False if the node specified by its ID is/is not None"""
        depth = int(math.log2(id + 1))
        index = id - (2 ** depth) + 1
        return self.isNodeNoneByLocation(depth,index)
    
    def height(self):
        """Returns height of the tree"""
        height = 0
        while True:
            if any(not self.isNodeNoneByLocation(height,index) for index in range(2 ** height)):
                height += 1
            else:
                break;
        height -= 1
        return height
    
    def __str__(self):
        """For simple representation of trees"""
        str_ = ''
        height = self.height()
        for depth in range(height + 1):
            str_ += ' ' * 3 * (2 ** (height - depth) - 1)
            for index in range(2 ** depth):
                if self.isNodeNoneByLocation(depth,index):
                    str_ += '  .  '
                    str_ += ' ' * (3 * (2 ** (height - depth + 1) - 1) - 2)
                elif isinstance(self.getNodeByLocation(depth,index).value,(int,float)):
                    str_ += "{: 5.01f}".format(round(self.getNodeByLocation(depth,index).value,1))
                    str_ += ' ' * (3 * (2 ** (height - depth + 1) - 1) - 2)
                else:
                    str_ += "{:^5}".format(self.getNodeByLocation(depth,index).value)
                    str_ += ' ' * (3 * (2 ** (height - depth + 1) - 1) - 2)
            str_ += '\n'
        return str_
    
    def makeTree(self):
        """Populate nodes of an initialized tree by ramped half-and-half method"""
        if self.type == 'Ms. PacMan':
            terminalsSet = PACMAN_TERMINALS
        elif self.type == 'Ghost':
            terminalsSet = GHOSTS_TERMINALS
        #Ramped half-and-half initialization
        if random.random() > 0.5:
            #Full method
            for depth in range(MAX_TREE_DEPTH + 1):
                for index in range(2 ** depth):
                    node = self.getNodeByLocation(depth,index)
                    if depth < MAX_TREE_DEPTH:
                        node.value = random.choice(FUNCTIONS)
                        node.addLeftChild()
                        node.addRightChild()
                    else:
                        node.value = random.choice(terminalsSet)
                    if node.value is CONSTANT:
                        node.value = random.uniform(CONSTANT_MIN,CONSTANT_MAX)
        else:
            #Grow method
            for depth in range(MAX_TREE_DEPTH + 1):
                for index in range(2 ** depth):
                    node = self.getNodeByLocation(depth,index)
                    if depth is 0:
                        node.value = random.choice(FUNCTIONS + terminalsSet)
                        node.addLeftChild()
                        node.addRightChild()
                    else:
                        if depth < MAX_TREE_DEPTH:
                            node.addLeftChild()
                            node.addRightChild()
                            if self.getNodeByID(nodeParent((2 ** depth) + index - 1)).value in FUNCTIONS:
                                node.value = random.choice(FUNCTIONS + terminalsSet)
                        elif self.getNodeByID(nodeParent((2 ** depth) + index - 1)).value in FUNCTIONS:
                            node.value = random.choice(terminalsSet)
                    if node.value is CONSTANT:
                        node.value = random.uniform(CONSTANT_MIN,CONSTANT_MAX)

class GPac:
    """class of methods for simulating Pac-Man"""
    def pacmanController(self,world,unit,ghosts,stateEvaluator,config):
        """control movement of Ms. Pac-Man"""
        validMoves = []
        for move in PACMAN_MOVES:
            if 0 <= unit.x + move[0] < config['world width'] and \
            0 <= unit.y + move[1] < config['world height'] and \
            all((unit.x + move[0] != ghost.x or unit.y + move[1] != ghost.y) for ghost in ghosts):
                validMoves.append(move)
        if stateEvaluator == None:
            #Random controller
            unit.move(world,random.choice(validMoves),config)
        else:
            #GP controller
            newStateGoodness = -100000 #very small number
            bestIdx = 0
            for idx,move in enumerate(validMoves):
                unit.move(world,move,config)
                stateGoodness = self.runTree(stateEvaluator,world,unit,unit,ghosts)
                if newStateGoodness < stateGoodness:
                    newStateGoodness = stateGoodness
                    bestIdx = idx
                reverseMove = [-x for x in move]
                unit.move(world,reverseMove,config)
            unit.move(world,validMoves[bestIdx],config)
    
    def ghostController(self,world,unit,pacman,ghosts,stateEvaluator,config):
        """Control movement of the ghosts"""
        validMoves = []
        for move in GHOST_MOVES:
            if 0 <= unit.x + move[0] < config['world width'] and 0 <= unit.y + move[1] < config['world height']:
                validMoves.append(move)
        if stateEvaluator == None:
            #Random controller
            unit.move(world,random.choice(validMoves),config)
        else:
            #GP controller
            newStateGoodness = -100000 #very small number
            bestIdx = 0
            for idx,move in enumerate(validMoves):
                unit.move(world,move,config)
                stateGoodness = self.runTree(stateEvaluator,world,unit,pacman,ghosts)
                if newStateGoodness < stateGoodness:
                    newStateGoodness = stateGoodness
                    bestIdx = idx
                reverseMove = [-x for x in move]
                unit.move(world,reverseMove,config)
            unit.move(world,validMoves[bestIdx],config)
    
    def manhattanDistance(self,unit1,unit2):
        """Return Manhattan distance between two points"""
        if hasattr(unit1, 'x'):
            x1 = unit1.x
            y1 = unit1.y
        else:
            x1 = unit1[0]
            y1 = unit1[1]
        if hasattr(unit2, 'x'):
            x2 = unit2.x
            y2 = unit2.y
        else:
            x2 = unit2[0]
            y2 = unit2[1]
        distance = abs(x1 - x2) + abs(y1 - y2)
        return distance
    
    def functionOutput(self,parent,child1,child2):
        """Translate the given function and return its output for provided operands"""
        if parent is '+':
            return (child1 + child2)
        elif parent is '-':
            return (child1 - child2)
        elif parent is '*':
            return (child1 * child2)
        elif parent is '/':
            if child2 == 0:
                return 0
            else:
                return (child1 / child2)
        elif parent is 'rand':
            return random.uniform(child1,child2)
    
    def runTree(self,tree,world,unit,pacman,ghosts):
        """Return the overall output value of a tree"""
        treeCpy = copy.deepcopy(tree)
        height = treeCpy.height()
        for id in reversed(range(totalNumberOfNodes(height))):
            if not treeCpy.isNodeNoneByID(id):
                node = treeCpy.getNodeByID(id)
                if node.value is DISTANCE_TO_NEAREST_GHOST:
                    node.value = self.distanceToNearestGhost(unit,ghosts)
                elif node.value is DISTANCE_TO_NEAREST_PILL:
                    node.value = self.distanceToNearestPill(unit,world)
                elif node.value is DISTANCE_TO_PACMAN:
                    node.value = self.distanceToPacman(unit,pacman)
        for id in reversed(range(totalNumberOfNodes(height))):
            if not treeCpy.isNodeNoneByID(id):
                node = treeCpy.getNodeByID(id)
                if node.value in FUNCTIONS:
                    node.value = self.functionOutput(node.value,treeCpy.getNodeByID(nodeLeftChild(id)).value,treeCpy.getNodeByID(nodeRightChild(id)).value)
        return node.value
    
    def distanceToNearestGhost(self,unit,ghosts):
        """Return Manhattan distance of the given unit to the nearest ghost"""
        if isinstance(unit, Pacman):
            return min(self.manhattanDistance(unit,ghost) for ghost in ghosts)
        elif isinstance(unit, Ghost):
            sortedDistances = sorted(self.manhattanDistance(unit,ghost) for ghost in ghosts)
            return sortedDistances[1] # First distance will certainly be zero and corresponds to distance to itself. Next distance value is what we look for.
    
    def distanceToPacman(self,unit,pacman):
        """Return Manhattan distance of the given unit to Ms. PacMan"""
        return self.manhattanDistance(unit,pacman)
    
    def distanceToNearestPill(self,unit,world):
        """Return Manhattan distance of the given unit to the nearest pill"""
        distance = len(world.grid) + len(world.grid[0]) - 2
        for i in range(len(world.grid)):
            for j in range(len(world.grid[0])):
                if world.grid[i][j] is PILL and distance > self.manhattanDistance(unit,[i,j]):
                    distance = self.manhattanDistance(unit,[i,j])
        return distance
    
    def playTurn(self,time,world,pacman,ghosts,pacmanStateEvaluator,ghostsStateEvaluator,config):
        """Play the game for one time step"""
        gameState = GAME_CONTINUES
        self.pacmanController(world,pacman,ghosts,pacmanStateEvaluator,config)
        for ghost in ghosts:
            self.ghostController(world,ghost,pacman,ghosts,ghostsStateEvaluator,config)
        if any(pacman.x == ghost.x and pacman.y == ghost.y for ghost in ghosts):
            #Game-over
            gameState = GAME_OVER
        elif world.grid[pacman.x][pacman.y] == PILL:
            world.grid[pacman.x][pacman.y] = EMPTY
            world.nRemainingPills -= 1
        pacman.score = int((1 - world.nRemainingPills / world.nInitialPills) * 100)
        for ghost in ghosts:
            ghost.score = -pacman.score
        if world.nRemainingPills == 0:
            gameState = GAME_OVER
            pacman.score *= 1 + time / (2 * config['world width'] * config['world height'])
            pacman.score = int(pacman.score)
            for ghost in ghosts:
                ghost.score = -pacman.score
        return gameState
    
    def playGame(self,config,pacmanStateEvaluator = None,ghostsStateEvaluator = None):
        """Run the whole game until a game-over condition occurs"""
        time = 2 * config['world width'] * config['world height']
        world = World(config)
            
        pills = []
        for i in range(config['world width']):
            for j in range(config['world height']):
                if world.grid[i][j] == PILL:
                    pills.append([i,j])
        pacman = Pacman(world,config)
        ghosts=[]
        for index in range(NUMBER_OF_GHOSTS):
            ghosts.append(Ghost(world,config))
        pacmanSeq = []
        ghostsSeq = []
        pacmanSeq.append(Pacman(world,config,x = pacman.x,y = pacman.y,score = pacman.score))
        ghostsSeq.append([Ghost(world,config,x = ghost.x,y = ghost.y,score = ghost.score) for ghost in ghosts])
        gameState = GAME_CONTINUES
        while time != 0 and gameState != GAME_OVER:
            time -= 1
            gameState = self.playTurn(time,world,pacman,ghosts,pacmanStateEvaluator,ghostsStateEvaluator,config)
            pacmanSeq.append(Pacman(world,config,x = pacman.x,y = pacman.y,score = pacman.score))
            ghostsSeq.append([Ghost(world,config,x = ghost.x,y = ghost.y,score = ghost.score) for ghost in ghosts])
        #Plus 1 to give all individuals at least a small chance of survival
        pacmanFitness = pacman.score + 1
        #Parsimony pressure
        pacmanFitness -= pacmanStateEvaluator.height() * config['pacman - parsimony pressure penalty coefficient']
        #Calculate average fitness if more than one evaluation has been done
        pacmanFitness = (pacmanFitness + pacmanStateEvaluator.fitness * pacmanStateEvaluator.evals) / (pacmanStateEvaluator.evals + 1)
        pacmanStateEvaluator.fitness = pacmanFitness
        #Minus 1 to give all individuals at least a small chance of survival
        ghostsFitness = -(pacman.score + 1)
        #Parsimony pressure
        ghostsFitness -= ghostsStateEvaluator.height() * config['ghosts - parsimony pressure penalty coefficient']
        #Calculate average fitness if more than one evaluation has been done
        ghostsFitness = (ghostsFitness + ghostsStateEvaluator.fitness * ghostsStateEvaluator.evals) / (ghostsStateEvaluator.evals + 1)
        ghostsStateEvaluator.fitness = ghostsFitness
        
        pacmanStateEvaluator.evals += 1
        ghostsStateEvaluator.evals += 1
        return (pills,pacmanSeq,ghostsSeq)
    
    def sumFitness(self,population):
        """Calculate sum of all fitness values in the population"""
        return sum(individual.fitness for individual in population)
    
    def averageFitness(self,population):
        """Calculate average fitness in the population"""
        return round(self.sumFitness(population) / len(population),2)
    
    def findFittest(self,population,n = 1):
        """Find the (n) fittest individual(s) in the population and return it.
        Population will be sorted in descending order based on fitness of individuals"""
        population.sort(key = operator.attrgetter('fitness'),reverse = True)
        if n == 1:
            return population[0]
        else:
            return population[0:n]
    
    def NormalizeFitness(self,population):
        """Update normalizedFitness attribute of the individuals given the whole population. Fitness of the ghosts will also be normalized
        in order to make all fitness values positive"""
        if population[0].type == 'Ghost':
            population.sort(key = operator.attrgetter('fitness'))
            minFitness = population[0].fitness
            for individual in population:
                individual.fitness -= minFitness - 1
        sumFitness = self.sumFitness(population)
        for individual in population:
            individual.normalizedFitness = individual.fitness / sumFitness
    
    def recombine(self,parent1,parent2):
        """Recombine two parents using sub-tree crossover and return two children"""
        rand1 = random.randint(0,totalNumberOfNodes(parent1.height()) - 1)
        while parent1.isNodeNoneByID(rand1):
            rand1 = random.randint(0,totalNumberOfNodes(parent1.height()) - 1)
        rand2 = random.randint(0,totalNumberOfNodes(parent2.height()) - 1)
        while parent2.isNodeNoneByID(rand2):
            rand2 = random.randint(0,totalNumberOfNodes(parent2.height()) - 1)
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        crossoverNode1 = child1.getNodeByID(rand1)
        crossoverNode2 = child2.getNodeByID(rand2)
        child1.setNodeByID(rand1,crossoverNode2)
        child2.setNodeByID(rand2,crossoverNode1)
        children = [child1,child2]
        return children
    
    def mutate(self,individual):
        """Apply sub-tree mutation"""
        rand = random.randint(0,totalNumberOfNodes(individual.height()) - 1)
        while individual.isNodeNoneByID(rand):
            rand = random.randint(0,totalNumberOfNodes(individual.height()) - 1)
        subTree = GPTree(individual.type)
        subTree.makeTree()
        individual.setNodeByID(rand,subTree)
    
    def parentSelection(self,parents,nSelectedParents,config):
        """Select nSelectedParents from provided parents using specified method in config file and return them"""
        parents.sort(key = operator.attrgetter('fitness'),reverse = True)
        selectedParents = []
        selectedIndices = []
        if parents[0].type == 'Ms. PacMan':
            parentSelectionMethod = config['pacman - parent selection method']
            fitterGroupProportion = config['pacman - proportion of population in fitter group']
        elif parents[0].type == 'Ghost':
            parentSelectionMethod = config['ghosts - parent selection method']
            fitterGroupProportion = config['ghosts - proportion of population in fitter group']
        
        if parentSelectionMethod == 'fps':
            while len(selectedParents) < nSelectedParents:
                rand = random.random()
                position = 0
                for idx,individual in enumerate(parents):
                    position += individual.normalizedFitness
                    if rand < position:
                        if idx not in selectedIndices:
                            selectedParents.append(individual)
                            selectedIndices.append(idx)
                        break
        elif parentSelectionMethod == 'over-selection':
            fitterIndividualsLastIndex = int(round(fitterGroupProportion * len(parents),0))
            selectedParents.extend(random.sample(parents[:fitterIndividualsLastIndex],int(round(nSelectedParents * 0.8,0))))
            selectedParents.extend(random.sample(parents[fitterIndividualsLastIndex:],int(round(nSelectedParents * 0.2,0))))
        return selectedParents
    
    def survivalSelection(self,population,nSurvivors,config):
        """Select nSurvivors from the population using specified method in config file and return them"""
        populationCpy = copy.deepcopy(population)
        selectedPopulation = []
        if population[0].type == 'Ms. PacMan':
            survivalSelectionMethod = config['pacman - survival selection method']
            tournamentSize = config['pacman - tournament size for survival selection']
        elif population[0].type == 'Ghost':
            survivalSelectionMethod = config['ghosts - survival selection method']
            tournamentSize = config['ghosts - tournament size for survival selection']
        
        if survivalSelectionMethod == 'truncation':
            populationCpy.sort(key = operator.attrgetter('fitness'),reverse = True)
            selectedPopulation = populationCpy[:nSurvivors]
        elif survivalSelectionMethod == 'k-tournament w/o replacement':
            while len(selectedPopulation) < nSurvivors:
                randIndices = random.sample(range(len(populationCpy)),tournamentSize)
                bestFitness = -1000 #very small number
                bestIndividual = populationCpy[0]
                for idx in randIndices:
                    if bestFitness < populationCpy[idx].fitness:
                        bestIndividual = populationCpy[idx]
                        bestFitness = bestIndividual.fitness
                        bestIdx = idx
                selectedPopulation.append(bestIndividual)
                del populationCpy[bestIdx]
        return selectedPopulation
    
    def drawWorld(self,world,pacman,ghosts,config):
        """Draw the world. Used for visualization and debugging"""
        sys.stdout.write((2 * config['world width'] + 1) * '-' + '\n')
        for j in reversed(range(config['world height'])):
            sys.stdout.write('|')
            for i in range(config['world width']):
                if any(ghost.x == i and ghost.y == j for ghost in ghosts):
                    cell = 'G'
                elif pacman.x == i and pacman.y == j:
                    cell = 'P'
                elif world.grid[i][j] is PILL:
                    cell = '+'
                else:
                    cell = ' '
                sys.stdout.write(cell + '|')
            sys.stdout.write('\n')
            sys.stdout.write((2 * config['world width'] + 1) * '-' + '\n')
        sys.stdout.write('\n')
    
    def runExperiment(self,config):
        """Main method for running the experiment"""
        random.seed(a = config['random generator seed'])
        log = WriteToFile(config)
        log.writeHeaderToLog()
        log.writeResultsLabelToLog()
        globalBestPacman = GPTree('Ms. PacMan')
        globalBestGhost = GPTree('Ghost')
        for run in range(config['number of runs']):
            pacmanPopulation = []
            ghostsPopulation = []
            for idx in range(config['pacman - population size']):
                pacmanIndividual = GPTree('Ms. PacMan')
                pacmanIndividual.makeTree()
                pacmanPopulation.append(pacmanIndividual)
            for idx in range(config['ghosts - population size']):
                ghostsIndividual = GPTree('Ghost')
                ghostsIndividual.makeTree()
                ghostsPopulation.append(ghostsIndividual)
            for eval in range(max(config['pacman - population size'],config['ghosts - population size'])):
                if PRINT:
                    progress = 100 * round((run * config['number of evaluations'] + eval + 1) /
                                           (config['number of runs'] * config['number of evaluations'] ),4)
                    progress = min(progress,100)
                    sys.stdout.flush()
                    sys.stdout.write('\r')
                    _ = sys.stdout.write("[%-20s] %5.2f%%" % ('=' * int(progress / 5),progress))
                (pills,pacmanSeq,ghostsSeq) = self.playGame(config,pacmanStateEvaluator = pacmanPopulation[eval % config['pacman - population size']],
                                                            ghostsStateEvaluator = ghostsPopulation[eval % config['ghosts - population size']])
            self.NormalizeFitness(pacmanPopulation)
            self.NormalizeFitness(ghostsPopulation)
            localBestPacman = GPTree('Ms. PacMan')
            localBestGhost = GPTree('Ghost')
            log.writeRunLabelToLog(run + 1)
            log.writeResultsToLog(eval + 1,self.averageFitness(pacmanPopulation),self.findFittest(pacmanPopulation).fitness)
            while eval < config['number of evaluations'] - 1:
                selectedPacmanParents = self.parentSelection(pacmanPopulation,config['pacman - number of offspring'],config)
                pacmanOffspring = []
                for idx in range(int(config['pacman - number of offspring'] / 2)):
                    pacmanChildren = self.recombine(selectedPacmanParents.pop(),selectedPacmanParents.pop())
                    for pacmanChild in pacmanChildren:
                        if random.random() < config['pacman - mutation probability']:
                            self.mutate(pacmanChild)
                    pacmanOffspring.extend(pacmanChildren)
                selectedGhostsParents = self.parentSelection(ghostsPopulation,config['ghosts - number of offspring'],config)
                ghostsOffspring = []
                for idx in range(int(config['ghosts - number of offspring'] / 2)):
                    ghostsChildren = self.recombine(selectedGhostsParents.pop(),selectedGhostsParents.pop())
                    for ghostsChild in ghostsChildren:
                        if random.random() < config['ghosts - mutation probability']:
                            self.mutate(ghostsChild)
                    ghostsOffspring.extend(ghostsChildren)
                random.shuffle(pacmanOffspring)
                random.shuffle(ghostsOffspring)
                for idx in range(max(len(pacmanOffspring),len(ghostsOffspring))):
                    (pills,pacmanSeq,ghostsSeq) = self.playGame(config,pacmanStateEvaluator = pacmanOffspring[idx % len(pacmanOffspring)],
                                                                ghostsStateEvaluator = ghostsOffspring[idx % len(ghostsOffspring)])
                    if PRINT:
                        progress = 100 * round((run * config['number of evaluations'] + eval + 1) /
                                               (config['number of runs'] * config['number of evaluations'] ),4)
                        progress = min(progress,100)
                        sys.stdout.flush()
                        sys.stdout.write('\r')
                        _ = sys.stdout.write("[%-20s] %5.2f%%" % ('=' * int(progress / 5),progress))
                    eval += 1
                    if localBestPacman.fitness < pacmanOffspring[idx % len(pacmanOffspring)].fitness:
                        localBestPacman = pacmanOffspring[idx % len(pacmanOffspring)]
                        localHighestScoreGamePills = pills
                        localHighestScoreGamePacmanSeq = pacmanSeq
                        localHighestScoreGameGhostsSeq = ghostsSeq
                        log.writeToWorldFile(localHighestScoreGamePacmanSeq,localHighestScoreGameGhostsSeq,localHighestScoreGamePills)
                        log.writeToSolutionFile(localBestPacman)
                    if localBestGhost.fitness < ghostsOffspring[idx % len(ghostsOffspring)].fitness:
                        localBestGhost = ghostsOffspring[idx % len(ghostsOffspring)]
                        log.writeToSolutionFile(localBestGhost)
                #Plus strategy:
                pacmanPopulation.extend(pacmanOffspring)
                ghostsPopulation.extend(ghostsOffspring)
                self.NormalizeFitness(pacmanPopulation)
                self.NormalizeFitness(ghostsPopulation)
                pacmanPopulation = self.survivalSelection(pacmanPopulation,config['pacman - population size'],config)
                ghostsPopulation = self.survivalSelection(ghostsPopulation,config['ghosts - population size'],config)
                self.NormalizeFitness(pacmanPopulation)
                self.NormalizeFitness(ghostsPopulation)
                log.writeResultsToLog(eval + 1,self.averageFitness(pacmanPopulation),localBestPacman.fitness)
            if globalBestPacman.fitness < localBestPacman.fitness:
                globalBestPacman = localBestPacman
                globalHighestScoreGamePills = localHighestScoreGamePills
                globalHighestScoreGamePacmanSeq = localHighestScoreGamePacmanSeq
                globalHighestScoreGameGhostsSeq = localHighestScoreGameGhostsSeq
            if globalBestGhost.fitness < localBestGhost.fitness:
                globalBestGhost = localBestGhost
        log.writeToWorldFile(globalHighestScoreGamePacmanSeq,globalHighestScoreGameGhostsSeq,globalHighestScoreGamePills)
        log.writeToSolutionFile(globalBestPacman)
        log.writeToSolutionFile(globalBestGhost)

#Optional config file used by"-config" flag
parser = argparse.ArgumentParser(description='Simulates Pac-Man game where controllers of both Ms. PacMan and the ghosts are evolved using a genetic programming algorithm in a coevolution.')
parser.add_argument("-config",default=os.path.join('config','default.cfg'),help='config file path',metavar='Path')
args = parser.parse_args()

#Instantiate Config class to load and parse configurations (if no config is specified,default is used)
config = Config()
configDict = config.parseCfg(os.path.join(DIRECTORY,args.config))

if PRINT:
    print('\n')
    print('------------------------------')
    print('|           Progress         |')
    print('------------------------------')

#Instantiate GPac class to simulate Pac-Man game
gPac = GPac()
gPac.runExperiment(configDict)

if PRINT:
    print('\n')
