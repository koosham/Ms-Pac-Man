#!python
r"""Pac-Man - Random Search"""
__author__ = 'Koosha Marashi'
__email__ = 'km89f@mst.edu'

#import used packages
import random
import json
import time
import argparse
import os
import sys
from decimal import Decimal


DIRECTORY = os.path.dirname(os.path.abspath(__file__))
NUMBER_OF_GHOSTS = 3
GAME_CONTINUES = 0
GAME_OVER = 1
EMPTY = 0
PILL = 1
WALL = 2
PRINT = True
#Allowed movements of the units [x,y]
PACMAN_MOVES = [[0,0],[0,1],[0,-1],[1,0],[-1,0]]
GHOST_MOVES = [[0,1],[0,-1],[1,0],[-1,0]]

class Config:
    """Handle configurations"""

    def parseCfg(self,cfgFilePath):
        """Read JSON formatted .cfg file and import configurations"""
        with open(cfgFilePath) as cfgFile:    
            config = json.load(cfgFile)
        # Key: Description
        # output folder: Relative path of output files
        # log file: Name of log file to write
        # highest score game world file: Name of wld file to write the highest-score-game-sequence all-time-step world file
        # random generator seed type: Type of seed used for random number generator. Should be 'time' to use system epoch time in milliseconds, or 'int' for manually feeding the seed.
        # random generator seed: If randomSeedType is equal to 'int' then would be used as seed of random number generator
        # number of runs: Total number of runs
        # number of evaluations: Number of fitness evaluations per run
        # world width: number of columns in the grid of the world 
        # world height: number of rows in the grid of the world
        # pill density percentage: cell-wise probability of having a pill
        
        if config['random generator seed type'] == 'time':
            # Get system epoch time
            config['random generator seed'] = round(time.time() * 1000)
        elif config['random generator seed type'] == 'int':
            # Use manually provided seed
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
            logFileObject.write('\t' + str(arg))
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

class Pacman(object):
    """object of Ms. Pac-Man"""
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
        """"""
        self.x += direction[0]
        self.y += direction[1]
        self.x = max(0,min(self.x,config['world width'] - 1))
        self.y = max(0,min(self.y,config['world height'] - 1))

class Ghost(object):
    """object of the ghosts"""
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
        """"""
        self.x += direction[0]
        self.y += direction[1]
        self.x = max(0,min(self.x,config['world width'] - 1))
        self.y = max(0,min(self.y,config['world height'] - 1))

class World(object):
    """object of the world grid"""
    def __init__(self,config):
        """"""
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

class GPac:
    """class of methods for simulating Pac-Man"""
    def pacmanController(self,world,unit,config,mode = 'random'):
        """control movement of Ms. Pac-Man"""
        if mode == 'random':
            validMoves = []
            for move in PACMAN_MOVES:
                if 0 <= unit.x + move[0] < config['world width'] and 0 <= unit.y + move[1] < config['world height']:
                    validMoves.append(move)
            unit.move(world,random.choice(validMoves),config)
    
    def ghostController(self,world,unit,config,mode = 'random'):
        """control movement of the ghosts"""
        if mode == 'random':
            validMoves = []
            for move in GHOST_MOVES:
                if 0 <= unit.x + move[0] < config['world width'] and 0 <= unit.y + move[1] < config['world height']:
                    validMoves.append(move)
            unit.move(world,random.choice(validMoves),config)
    
    def playGame(self,time,world,pacman,ghosts,config):
        """simulate one turn of the game until a game-over condition occurs"""
        gameState = GAME_CONTINUES
        self.pacmanController(world,pacman,config)
        for ghost in ghosts:
            self.ghostController(world,ghost,config)
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
        globalBestFitness = 0
        for run in range(config['number of runs']):
            log.writeRunLabelToLog(run + 1)
            localBestFitness = 0
            for eval in range(config['number of evaluations']):
                if PRINT:
                    progress = 100 * round((run * config['number of evaluations'] + eval + 1) /
                                           (config['number of runs'] * config['number of evaluations'] ),4)
                    progress = min(progress,100)
                    sys.stdout.flush()
                    sys.stdout.write('\r')
                    _ = sys.stdout.write("[%-20s] %5.1f%%" % ('='*int(progress / 5),progress))
                
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
                    gameState = self.playGame(time,world,pacman,ghosts,config)
                    pacmanSeq.append(Pacman(world,config,x = pacman.x,y = pacman.y,score = pacman.score))
                    ghostsSeq.append([Ghost(world,config,x = ghost.x,y = ghost.y,score = ghost.score) for ghost in ghosts])
                if localBestFitness == 0 or localBestFitness < pacman.score:
                    localBestFitness = pacman.score
                    localHighestScoreGamePills = pills
                    localHighestScoreGamePacmanSeq = pacmanSeq
                    localHighestScoreGameGhostsSeq = ghostsSeq
                    log.writeResultsToLog(eval + 1,localBestFitness)
            if globalBestFitness == 0 or globalBestFitness < localBestFitness:
                globalBestFitness = localBestFitness
                globalHighestScoreGamePills = localHighestScoreGamePills
                globalHighestScoreGamePacmanSeq = localHighestScoreGamePacmanSeq
                globalHighestScoreGameGhostsSeq = localHighestScoreGameGhostsSeq
        log.writeToWorldFile(globalHighestScoreGamePacmanSeq,globalHighestScoreGameGhostsSeq,globalHighestScoreGamePills)
        

# Optional config file used by"-config" flag
parser = argparse.ArgumentParser(description='Simulates Pac-Man game where controllers decide movements randomly.')
parser.add_argument("-config",default=os.path.join('config','default.cfg'),help='config file path',metavar='Path')
args = parser.parse_args()

# Instantiate Config class to load and parse configurations (if no config is specified,default is used)
config = Config()
configDict = config.parseCfg(os.path.join(DIRECTORY,args.config))

if PRINT:
    print('\n')
    print('-----------------------------')
    print('|          Progress         |')
    print('-----------------------------')

#Instantiate GPac class to simulate Pac-Man game
gpac = GPac()
gpac.runExperiment(configDict)

if PRINT:
    print('\n')
