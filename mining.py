#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:56:47 2021

@author: frederic

    
class problem with     

An open-pit mine is a grid represented with a 2D or 3D numpy array. 

The first coordinates are surface locations.

In the 2D case, the coordinates are (x,z).
In the 3D case, the coordinates are (x,y,z).
The last coordinate 'z' points down.

    
A state indicates for each surface location  how many cells 
have been dug in this pit column.

For a 3D mine, a surface location is represented with a tuple (x,y).

For a 2D mine, a surface location is represented with a tuple (x,).


Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.


An action is represented by the surface location where the dig takes place.


"""
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import itertools

import functools # @lru_cache(maxsize=32)

from numbers import Number

import search

import time



def my_team():
    '''    Return the list of the team members of this assignment submission
    as a list    of triplet of the form (student_number, first_name, last_name)        '''
    return [(9960392,'Liam','Hulsman-Benson'), (1234568, 'Grace', 'Hopper')]
#end

    
def convert_to_tuple(a):
    '''
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.
    
    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    '''
    if isinstance(a, Number):
        return a
    #end
    if len(a)==0:
        return ()
    #end
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)
    #end
    
    
def convert_to_list(a):
    '''
    Convert the array-like parameter 'a' into a nested list of the same 
    shape as 'a'.

    Parameters
    ----------
    a : flat array or array of arrays

    Returns
    -------
    the conversion of 'a' into a list or a list of lists

    '''
    if isinstance(a, Number):
        return a
    #end
    if len(a)==0:
        return []
    #end
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]
    #end




class Mine(search.Problem):
    '''
    
    Mine represent an open mine problem defined by a grid of cells 
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.
    
    The z direction is pointing down, the x and y directions are surface
    directions.
    
    An instance of a Mine is characterized by 
    - self.underground : the ndarray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between 
                           adjacent columns 
    
    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the 
                                         mine
    
    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.
    
    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.
    
    States must be tuple-based.
    
    '''    
    
    def __init__(self, underground, dig_tolerance = 1):
        '''
        Constructor
        
        Initialize the attributes
        self.underground, self.dig_tolerance, self.len_x, self.len_y, self.len_z,
        self.cumsum_mine, and self.initial
        
        The state self.initial is a filled with zeros.

        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains 
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.

        '''
        # super().__init__() # call to parent class constructor not needed
        
        self.underground = underground 
        # self.underground  should be considered as a 'read-only' variable!
        self.dig_tolerance = dig_tolerance

        #Convert 2D to 3D and set 2D conversion flag, to convert back at end
        self.flag2D = False
        if self.underground.ndim == 2:
            self.flag2D = True
            self.underground = np.expand_dims(self.underground,1)
        #end

        self.initial = np.zeros_like(self.underground)
        self.cumsum_mine = 0.0

        self.len_x, self.len_y, self.len_z = self.underground.shape
    #end


    def surface_neigbhours(self, loc):
        '''
        Return the list of neighbours of loc

        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x,) in case of a 2D mine
            a pair (x,y) in case of a 3D mine

        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.

        '''
        L=[]
        assert len(loc) in (1,2)
        if len(loc)==1:
            if loc[0]-1>=0:
                L.append((loc[0]-1,))
            #end
            if loc[0]+1<self.len_x:
                L.append((loc[0]+1,))
            #end
        else:
            # len(loc) == 2
            for dx,dy in ((-1,-1),(-1,0),(-1,+1),
                          (0,-1),(0,+1),
                          (+1,-1),(+1,0),(+1,+1)):
                if  (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy < self.len_y):
                    L.append((loc[0]+dx, loc[1]+dy))
                #end
            #end
        #end
        return L
    #end

    def getDepth(self, state, loc):
        '''
                Returns an int of the depth of the current surface location.

                Parameters
                ----------
                state :
                    represented with nested lists, tuples or a ndarray
                    state of the partially dug mine

                loc : surface coordinates of a cell.
                    a singleton (x,) in case of a 2D mine
                    a pair (x,y) in case of a 3D mine


                Returns
                -------
                an int of the number of cells dug of the surface location

                '''

        return sum(state[loc])
    #end


    def actions(self, state):
        '''
        Return a generator of valid actions in the give state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't  break the dig_tolerance constraint.

        Parameters
        ----------
        state : 
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions

        '''        
        state = np.array(state)

        ####################
        #   Really bad need to find a better way.
        ####################

        for x in range(0, self.len_x, 1):
            for y in range(0, self.len_y, 1):
                validCheck = False
                for neighbour in self.surface_neigbhours((x,y)):
                    g = self.getDepth(state, (x,y))
                    h = self.getDepth(state, neighbour)
                    if abs(self.getDepth(state, (x,y)) - self.getDepth(state, neighbour)) <= self.dig_tolerance and self.getDepth(state, (x,y)) < self.len_z:
                        validCheck = True
                    else:
                        validCheck = False
                        break
                    #end
                #end

                if validCheck:
                    yield (x,y)
                #end
            #end

    #end
                
  
    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state)."""
        action = tuple(action)
        new_state = np.array(state) # Make a copy
        new_state[action] += 1
        return convert_to_tuple(new_state)
    #end
                
    
    def console_display(self):
        '''
        Display the mine on the console

        Returns
        -------
        None.

        '''
        print('Mine of depth {}'.format(self.len_z))
        if self.underground.ndim == 2:
            # 2D mine
            print('Plane x,z view')
        else:
            # 3D mine
            print('Level by level x,y slices')
        #end
        print(self.__str__())
    #end
        
    def __str__(self):
        if self.underground.ndim == 2:
            # 2D mine
            return str(self.underground.T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                   +str(self.underground[...,z]) for z in range(self.len_z))
                    
                        
                
            return self.underground[loc[0], loc[1],:]
        #end
    #end
    
    @staticmethod   
    def plot_state(state):
        if state.ndim==1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]) ,
                    state
                    )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim==2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x) # cols, rows
            x, y = _xx.ravel(), _yy.ravel()            
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3,3))
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #end
        plt.show()
    #end

    def payoff(self, state):
        '''
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the digged cells.
        
        No loops needed in the implementation!        
        '''
        # convert to np.array in order to use tuple addressing
        # state[loc]   where loc is a tuple

        return sum(state & self.underground)
    #end

    def is_dangerous(self, state):
        '''
        Return True if the given state breaches the dig_tolerance constraints.
        
        No loops needed in the implementation!
        '''
        # convert to np.array in order to use numpy operators
        state = np.array(state)         

        raise NotImplementedError
    #end

    def back2D(self, actions, state):
        """
        Converts 3D locations and arrays, as converted in the Mine.init, back to 2D.

        Parameters
        ----------
        actions
        state

        Returns
        -------
        actions, state
        """

        if self.flag2D:
            for i, a in enumerate(np.array(actions)):
                actions[i] = tuple(a[[0, 2]])
            # end
            state = np.squeeze(state)
        # end

        return actions, state
    #end

    # ========================  Class Mine  ==================================

def getParentsSum(mine, state, loc, seenLocs):
    '''
      Gets the specified cell value, specified by loc, incorporating the cells values required to get to "loc".

      Function works with a pattern that was noticed, further details in report. Briefly, To calculate any cell, the
       cell directly above loc, with the simplified mine value, can be added with cell values dig tolerance above the loc.
       Which produces an upside down pyramid shape, with the point being on the loc location.

       E.g. dig_tolerence = 1, len_x = 3, len_y = 1, len_z = 3 , loc = (1,0,2)

       getParentsSum(loc=(1,0,1)) + mine.underground[(0,0,0)] + mine.underground[(0,0,2)]

       Therefore if loc(1,0,1) is saved in dictionary with its values and path, the recursive search for
       its values can be skipped.


      Parameters
      ----------
      mine : a Mine instance
      state : a partially dug mine
      loc : a 3d tuple (x,y,z), to a location in the mine.
      seenLocs : a dictionary with keys as loc coordinates. Stores the sum and path for locs previosuly explored. Is
      defined outside function, so sucessive recursive calls can access entire dict.

      Returns
      -------
      output

      '''

    path = []  #Cell path to get to speicifed cell.
    output = mine.underground[loc]  #Add the speicied location cell value to output.
    z = loc[2]-1    #Decrease z by 1, to move verticall up the mine.

    #Check if cell has been mined previosuly, in this case can return 0 as it would sum ot 0 anyway.
    if state[loc] == 1:
        return 0
    #end

    prevCalcCoords = (loc[0], loc[1], loc[2] - 1)   #The cell directly above specifed loc.

    #Check if prevCalcCoords has been calculated previosuly, if it has, combine path and sums.
    if prevCalcCoords in seenLocs:
        output += seenLocs[prevCalcCoords]['sum'].copy()
        path = seenLocs[prevCalcCoords]['path'].copy()
        path.append(prevCalcCoords)
    #end

    #Calculate the cells dig_tolerence above the specified loc, in a donut shape.
    for x in range(-z,z+1):
        for y in range(-z, z+1):
            if 0 <= loc[0]-x < mine.len_x and 0 <= loc[1]-y < mine.len_y and (x,y) != (0,0):

                #Prevent values already used being used again
                if state[(loc[0]-x, loc[1]-y,z-mine.dig_tolerance)] == 0:
                    output += mine.underground[(loc[0]-x, loc[1]-y,z-mine.dig_tolerance)]
                    path.append((loc[0]-x, loc[1]-y,z-mine.dig_tolerance))
                #end
            #end
        #end
    #end

    #Save loc just calculated into dictionary with path required to get there and output.
    seenLocs[loc] = dict({"sum":output, "path":path})

    #Return the sum value of the specified cell.
    return output
#end

def ringCoords(mine, loc):
    z = loc[2] - 1  # Decrease z by 1, to move vertical up the mine.

    outputCoords = []

    #Calculate the cells dig_tolerence above the specified loc, in a donut shape.
    for x in range(-z,z+1):
        for y in range(-z, z+1):
            if 0 <= loc[0]-x < mine.len_x and 0 <= loc[1]-y < mine.len_y and (x,y) != (0,0):
                outputCoords.append((loc[0]-x, loc[1]-y,z-mine.dig_tolerance))
            #end
        #end
    #end

    return outputCoords
#end

def test(mine, state, currentLoc, pathTaken, visitedLocs):

    outputSum = 0
    outputPath = []


    #Will only trigger if has already been mined, Therefore cell will always return 0
    #and a blank path.
    if currentLoc in pathTaken:
        return 0, []
    elif currentLoc in visitedLocs:
        return visitedLocs[currentLoc]['sum'],visitedLocs[currentLoc]['path']
    #end

    #Get value of loc directly above currentLoc
    aboveLoc = (currentLoc[0], currentLoc[1], currentLoc[2]-1)

    #If cell has been previosuly calculated, get value from dict
    if aboveLoc in visitedLocs:
        outputSum += visitedLocs[aboveLoc]["sum"]
        outputPath += visitedLocs[aboveLoc]["path"]
    elif 0<=aboveLoc[2]<mine.len_z:
        outputSum, outputPath = test(mine,state,aboveLoc,pathTaken, visitedLocs)
    #end

    #Gets the ring sum above currentLoc
    for loc in ringCoords(mine, currentLoc):
        s,p = test(mine, state, loc, pathTaken, visitedLocs)

        outputSum += s
        outputPath += (p)
    #end

    #Add currentLoc value and path
    outputSum += mine.underground[currentLoc]
    outputPath += [(currentLoc)]

    #Add to dict to use values for locs below currentLoc.
    visitedLocs[currentLoc] = {"sum": outputSum, "path": outputPath}

    # print(len(visitedLocs))
    #
    # if len(visitedLocs) == 512:
    #     print("")

    return outputSum, outputPath
#end

def search_rec(mine, state, visitedLocs={}, bestSum=0, bestPath=[]):
    '''
      Recursive search function for search_dp_dig_plan,

      Each function call returns the best sum and path for the remaining values in the mine.

      Simplifies the mine by summing the cells above which are required to be mined to collect the specific cell.
      Max value is then selected from mine and each corresponding cell is removed. Mine requirement is then recalculated
      to update values in mine.


      Parameters
      ----------
      mine : a Mine instance
      state : a partially dug mine

      Returns
      -------
      best_action_list, best_payoff, best_final_state

      '''


    # g,h = test(mine,state,(0,0,0), pathTaken, visitedLocs)
    # pathTaken += h
    #
    # g, h = test(mine, state, (0, 0, 1), pathTaken, visitedLocs)
    #
    # pathTaken = h
    # print(g)
    # print(h)

    #Loop Through each cell in the mine.

    locs = np.where(state == 0)
    print(len(locs[0]))
    if len(locs) == 3:
        x,y,z = locs
        for loc in zip(x,y,z):
            s,p = test(mine,state, loc, bestPath, visitedLocs)

            if s > bestSum:
                bestSum = s
                bestPath = p
            #end
        #end

        if bestSum > 0:
            #Dig specified location
            for action in bestPath:
                state = mine.result(state,action)
            #end
            state = np.array(state)

            tempSum, tempPath, state = search_rec(mine, state, visitedLocs, bestPath=bestPath)

            #Remove any locations already dug, from previous recursions
            for loc in list(set(bestPath)&set(tempPath)):
                tempSum -= mine.underground[loc]
                tempPath.remove(loc)
            #end

            bestSum += tempSum
            bestPath += tempPath
        else:
            return 0, [], state
    return bestSum, bestPath, state
#end

def search_dp_dig_plan(mine):
    '''
    Search using Dynamic Programming the most profitable sequence of 
    digging actions from the initial state of the mine.
    
    Return the sequence of actions, the final state and the payoff
    

    Parameters
    ----------
    mine : a Mine instance

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''

    best_payoff, best_action_list, best_final_state  = search_rec(mine, mine.initial)


    return best_action_list, best_payoff, best_final_state
#end


def search_bb_dig_plan(mine):
    '''
    Compute, using Branch and Bound, the most profitable sequence of 
    digging actions from the initial state of the mine.
        

    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    
    raise NotImplementedError
#end

def find_action_sequence(s0, s1):
    '''
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.
    
    Preconditions: 
        s0 and s1 are legal states, s0<=s1 and 
    
    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state 

    Returns
    -------
    A sequence of actions to go from state s0 to state s1

    '''    
    # approach: among all columns for which s0 < s1, pick the column loc
    # with the smallest s0[loc]
    raise NotImplementedError
#end




def main():
    print(my_team())

    v = np.array([[-1, -1, 10], [-1, 20, 4], [-1, -1, -1]])
    w = np.array([[1, 4], [2, 5], [3, 6]])
    x = np.array([[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, -1]])
    y = np.array([[1, -6, 1, 1], [2, 5, 1, 1], [3, 6, 1, 1], [3, 6, 1, -10]])
    z = np.array([[[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, 1]], x - 1])



    mine = Mine(underground=v)

    best_action_list, best_payoff, best_final_state = search_dp_dig_plan(mine)

    print(best_action_list)
    print(best_final_state)
    print(best_payoff)

#end
        
if __name__ == "__main__":
    main()
#end
        
        
    
    
    