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

from functools import lru_cache

from numbers import Number

import search

import operator
import time




def my_team():
    '''    Return the list of the team members of this assignment submission
    as a list    of triplet of the form (student_number, first_name, last_name)        '''
    return [(9960392,'Liam','Hulsman-Benson'), (10077413, 'Alexander', 'Farrall')]
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

class NonMatchingShapes(Exception):
    """
    Expection for "checkShiftedArrays" function, raised if input shapes are not same dimensions.
    """
    pass


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
                    diff = self.getDepth(state, (x,y)) - self.getDepth(state, neighbour)
                    if abs(diff) <= self.dig_tolerance and self.getDepth(state, (x,y)) < self.len_z and diff <= 0:
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

    def results(self, state, actions, prevSeenLocs=None):
        """Return the state that results from executing the given
        actions in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state)."""

        new_state = np.array(state)  # Make a copy
        for action in actions:
            new_state[action] += 1

            if prevSeenLocs != None:
                #Remove it from dict as it will never be used again
                if action in prevSeenLocs:
                    del prevSeenLocs[action]
                #end
            #end
        #end
        return new_state
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

        return sum(state * self.underground)
    #end

    def checkShiftedArrays(self, data1, data2):
        """
        Both input parameters must be same shape.
        Find abs difference between each input and compare to dig tolerance.
        Parameters
        ----------
        data1 - x*y array of depth values
        data2 - x*y array of depth values

        Returns
        -------
        Boolean, True if neighbours are valid.
        """

        if data1.shape == data2.shape:
            if False in np.where(abs(data1 - data2) <= self.dig_tolerance, True, False):
                return False
            return True
        else:
            raise NonMatchingShapes("Data1 and Data2 are not same shape.")
    #end

    def is_dangerous(self, state):
        '''
        Return True if the given state breaches the dig_tolerance constraints.

        No loops needed in the implementation!
        '''
        # convert to np.array in order to use numpy operators
        state = np.array(state)

        # Convert 2D mines to 3D mines and set flag to convert back.
        if state.ndim == 2:
            state = np.expand_dims(state, 1)
        # end

        #Get depth values of entire array.
        summedState = np.sum(state, axis=2)

        #XAxis, Left and Right in array
        if not self.checkShiftedArrays(summedState[:-1, :], summedState[1:, :]):
            return True
        #end

        #YAxis, Up and Down in array
        if not self.checkShiftedArrays(summedState[:, :-1], summedState[:, 1:]):
            return True
        #end

        # TopLeft 2 BottomRight, top left and bottom right of any given cell
        if not self.checkShiftedArrays(summedState[:-1, :-1], summedState[1:, 1:]):
            return True
        # end

        # TopRight 2 BottomLeft, top right and bottom left of any given cell
        if not self.checkShiftedArrays(summedState[:-1, 1:], summedState[1:, :-1]):
            return True
        # end

        #If here, array is valid.
        return False
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

        #Check if the 2D flag has been set. If it has remove the dimension that was added initially
        #and return actions and state
        if self.flag2D:
            for i, a in enumerate(np.array(actions)):
                actions[i] = tuple(a[[0, 2]])
            # end
            state = np.squeeze(state)
        # end

        return actions, state
    #end

    def validCoords(self, loc):
        """
        Checks if a given loc is a valid location within the mines limits.
        Parameters
        ----------
        loc - tuple, (x,y,z) of mine coordinates

        Returns
        -------
            Boolean if loc is within the mine
        """

        if 0<=loc[0]<self.len_x and 0<=loc[1]<self.len_y and 0<=loc[2]<self.len_z:
            return True
        #end
        return False
    #end

    # ========================  Class Mine  ==================================

#DP Arroach
def getRingCoords(mine, loc):
    """
    For a given loc, returns the coordinates "dig_tolereance" above the given loc. Used to find the cells required to
    dig a certain cell.
    Parameters
    ----------
    mine - A Mine instance
    loc - location in a mine, (x,y,z)

    Returns
    -------
    list of coords, which are needed to dig the given loc.
    """
    outputCoords = []

    #Check the rig coords are within the mine.
    if (loc[2]-mine.dig_tolerance >= 0):

        #Loop though a 3x3 grid of x and y values
        for x_loc in range(-1,2):
            for y_loc in range(-1,2):

                #Exclude middle of 3x3 grid.
                if not (x_loc == 0 and y_loc == 0):

                    #Check each location is valid within mine.
                    if (0<= loc[0] - x_loc < mine.len_x and 0<= loc[1] - y_loc < mine.len_y):

                        #Add a valid location to output coords to be checked later.
                        outputCoords.append((loc[0] - x_loc, loc[1] - y_loc, loc[2] - mine.dig_tolerance))

    return outputCoords
#end

def getParentsSum(mine, state, loc, prevSeenLocs):
    """
    When given a loc it returns the sum and path for the given loc. Takes into account cells in mine which are required
    to be dug for given loc, to ensure loc does not put mine into an invalid state.
    Parameters
    ----------
    mine - A Mine instance
    state - the current state of a mine
    loc - coordinates in a mine, (x,y,z)
    prevSeenLocs - Dictionary with keys of locs, (x,y,z), and values of sum of loc and other cells required to get to loc.

    Returns
    -------
    The sum and path to dig a cartain loc in current mine state
    """
    aboveCellCoords = (loc[0], loc[1], loc[2] - 1)  # The cell directly above specifed loc.
    ringCoords = getRingCoords(mine, loc)

    outputSum = mine.underground[loc]
    outputPath = [loc]

    #Cell has been previosuly calculated, return that instead
    if loc in prevSeenLocs:
        return prevSeenLocs[loc]["Sum"], prevSeenLocs[loc]["Path"]
    #end

    #To get loc cell value, need to add cell directly above and cells in ring shape around it. more specific in report
    for coords in [aboveCellCoords] + ringCoords:
        if mine.validCoords(coords) and state[coords] == 0:
            tempSum, tempPath = getParentsSum(mine,state,coords,prevSeenLocs)
            outputSum += tempSum
            outputPath = tempPath + outputPath
        #end
    #end

    #Add to Dict to Use later, instead of calcing again
    prevSeenLocs[loc] = {"Sum":outputSum, "Path":outputPath}

    return outputSum, outputPath
#end

def searchRec(mine, state, prevSeenLocs = {}, minePath=[], mineSum=[]):
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

    #Loop Through each cell in the mine.
    try:
        x,y,z = np.where((state==0) & (mine.underground>0))

        tuples = sorted(zip(x,y,z), key=lambda x: x[-1])
        #Go through positions in mine
        for loc in tuples:

            #If loc has been previously calculated use it, otherwise calculate for first time..
            if loc in prevSeenLocs:
                s,p = prevSeenLocs[loc]['Sum'], prevSeenLocs[loc]['Path']

                """
                Need to test this further, it works with test data
                """
                for minedLoc in set(p) & set(minePath):
                    s -= mine.underground[minedLoc]
                #end
                # for pathLoc in p:
                #     if pathLoc in minePath:
                #         s -= mine.underground[pathLoc]
                #     #end
                # #end

            else:
                s, p = getParentsSum(mine, state, loc, prevSeenLocs)
            #end

            #If the sum of a loc is greater than 0, it will be worth to dig.
            if s > 0:
                mineSum += [s]

                # Find the difference between the sets, so a cell is not dug multiple times. Only add the diff to
                # the final path
                p = list(set(p) - (set(p) & set(minePath)))
                minePath += p



                # Perform digging action
                state = mine.results(state, p, prevSeenLocs)

                #Recursivally call function to optimize mine, Can ignore first 2 outputs as they are only returing
                #values to top function
                _,_,state = searchRec(mine, np.array(state), minePath=minePath, mineSum=mineSum)
                break
            #end
        #end

    #Saftey Check
    except Exception as e:
        print(e)
        print("No avaliable locations to dig remaining")
    #end

    #Return the values.
    return np.sum(mine.underground * state), minePath, state
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

    best_payoff, best_action_list, best_final_state = searchRec(mine, mine.initial.copy())

    return best_payoff, find_action_sequence(np.zeros_like(best_final_state), best_final_state), best_final_state
#end


#BB Approach
def getMaxSumOfColoumn(col):
    maxSum = col[0]
    maxIndex = 0

    for i in range(1,len(col)+1):
        tempSum = np.sum(col[0:i])
        if tempSum > maxSum:
            maxSum = tempSum
            maxIndex = i
        #end
    #end

    return maxSum, maxIndex
#end

def getSummedCols(mine):
    summedCols = mine.underground[:,:,0].copy()

    #For each coloumn find the max possible value from it.
    for x in range(mine.len_x):
        for y in range(mine.len_y):
            maxSum, maxIndex = getMaxSumOfColoumn(mine.underground[x,y])
            summedCols[x,y] = maxSum
        #end
    #end

    return summedCols
#end

def getMaxNode(summedCols, validCoords):
    #Get max Node that is in valid coords
    sortedSummedCols = np.dstack(np.unravel_index(np.flip(np.argsort(summedCols.ravel())), summedCols.shape))[0]
    validCoords = [l[1] for l in validCoords]

    #While the max value is not in valid list, go to 2nd max value until true, ect.
    i = 0
    while tuple(sortedSummedCols[i]) not in validCoords:
        i+=1
    #end

    return tuple(sortedSummedCols[i]), summedCols[tuple(sortedSummedCols[i])]
#end


def search_bb_dig_plan(mine, state=None):
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

    rollingSum = []
    rollingPath = []

    maxSum = 0
    maxPath = []

    if state == None:
        state = mine.initial.copy()
    #end

    summedCols = getSummedCols(mine)
    while 0 in state:
        diggableLocs = np.array([(mine.underground[(l[0],l[1],int(np.sum(state[l])))],l) for l in mine.actions(state)])

        #Take any +ve value showing on surface
        truth = list(map(lambda l: l[0] > 0, diggableLocs))
        if True in truth:
            digLoc = tuple(diggableLocs[truth.index(True)][1])
        else:
            #If there is not positive values showing gotta branch.
            digLoc, maxSum = getMaxNode(summedCols, diggableLocs)
        #end

        digLoc3D = (digLoc[0], digLoc[1], int(np.sum(state[digLoc])))

        #Add to ouput list
        rollingSum.append(mine.underground[digLoc3D])
        rollingPath.append(digLoc3D)

        #Mine Location and update summedCols
        state = np.array(mine.result(state, digLoc3D))
        summedCols[digLoc] -= mine.underground[digLoc3D]

        tempSum, _ = getMaxSumOfColoumn(rollingSum)
        if tempSum > maxSum:
            maxSum = tempSum
            maxPath = rollingPath
        #end
    #end



    return maxSum, maxPath
#end


#Extra Function
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

    flag = False

    #Convert both states to numpy array.
    s0 = np.array(s0)
    s1 = np.array(s1)

    outputActionList = []

    #Convert 2D mines to 3D mines and set flag to convert back.
    if s0.ndim == 2:
        s0 = np.expand_dims(s0, 1)
        s1 = np.expand_dims(s1, 1)
        flag = True
    #end

    while not np.array_equal(s0, s1):
        #s0 < s1
        x,y = np.where(np.sum(s0,2) < np.sum(s1,2))
        locs = list(zip(x,y))

        #Get smallest s0[loc
        vals = list(map(lambda loc: sum(s0[loc]),locs))

        loc = locs[vals.index(min(vals))]
        loc = (loc[0], loc[1], int(sum(s0[loc])))

        outputActionList.append(loc)
        s0[loc] = 1
    #end

    return outputActionList
#end

def back2D(actions, state):
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

    #Check if the 2D flag has been set. If it has remove the dimension that was added initially
    #and return actions and state
    for i, a in enumerate(np.array(actions)):
        actions[i] = tuple(a[[0, 2]])
    # end
    state = np.squeeze(state)

    return actions, state
#end



def main():
    # print(my_team())
    #
    b = np.array([[-1, -200, 1], [5, 8, 5]])
    v = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -10, 11]])
    vDash = np.array([[-1, -1, 10, 5], [-1, -20, -4, -7], [-1, -1, -1, -21]])
    w = np.array([[1, 4], [2, 5], [3, 6]])
    x = np.array([[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, -1]])
    y = np.array([[1, -6, 1, 1], [2, 5, 1, 1], [3, 6, 1, 1], [3, 6, 1, -10]])
    z = np.array([[[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, 1]], x - 1])

    mine = Mine(underground=v, dig_tolerance=1)
    #
    # best_action_list, best_payoff, best_final_state = search_dp_dig_plan(mine)
    #
    # print(best_action_list)
    # print(best_final_state)
    # print(best_payoff)

    print(search_bb_dig_plan(mine))


    #
    s0 = [[1,0,0], [1,0,0], [0,0,0]]
    s1 = [[1, 1, 0, 0], [1, 1, 0,0], [1, 1, 1,0], [1,1,1,1]]

    s2_1 = [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
    s2_2 = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
    s2_3 = [[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]]
    s2 = [s2_1, s2_2, s2_3]

    # mine.plot_state(np.array(s1))

    # print(mine.is_dangerous(s0))

    # print(find_action_sequence(s0,s1))

#end

if __name__ == "__main__":
    main()

#end




