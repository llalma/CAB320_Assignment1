import mining
import random as rand
import numpy as np
import time

from mining import getRingCoords, getParentsSum, searchRec, search_dp_dig_plan ,findOptimalColHeight, search_bb_dig_plan, find_action_sequence


def test1():
    # Test for runtime, can handle 15x15x15 in less than a minute
    np.random.seed(0)
    f = 8

    array = np.random.randint(-10, 10, (f, f, f))

    start = time.time()
    mine = mining.Mine(underground=array)
    best_action_list, best_payoff, best_final_state = mining.search_dp_dig_plan(mine)
    # best_action_list, best_final_state = mine.back2D(best_action_list, best_final_state)

    print(best_action_list)
    print(best_final_state)

    print("RunTime = " + str(time.time() - start))


# end

def test2():
    input = np.array([[-1, -1, 10], [-1, 20, 4], [-1, -1, -1]])
    mine = mining.Mine(input)

    actualPath, actualSum, actualState = mining.search_dp_dig_plan(mine)

    actualPath, actualState = mine.back2D(actualPath, actualState)

    expectedSum = 30
    expectedPath = [(1, 0), (1, 1), (2, 0), (0, 0), (1, 2), (0, 1), (0, 2)]
    expectedState = [[1, 1, 1], [1, 1, 1], [1, 0, 0]]

    assert expectedSum == actualSum
    assert np.array_equal(expectedPath, actualPath)
    assert np.array_equal(expectedState, actualState)


# end

def surface_neigbhours2DTest():

    input = np.array([[-1, -1, 10], [-1, 20, 4], [-1, -1, -1]])
    mine = mining.Mine(input)

    loc = (1,0)
    actualNeighbours  = mining.Mine.surface_neigbhours(mine, loc)

    expectedNeighbours = [(0, 0), (2, 0)]
    assert expectedNeighbours == actualNeighbours
# end

def surface_neigbhours3DTest():

    x = np.array([[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, -1]])
    input = np.array([[[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, 1]], x - 1])
    mine = mining.Mine(input)

    loc = (1,1)
    actualNeighbours  = mining.Mine.surface_neigbhours(mine, loc)
    expectedNeighbours = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)]
    assert expectedNeighbours == actualNeighbours
# end

def actions2DTest():
    input = np.array([[-1, -1, 10], [-1, 20, 4], [-1, -1, -1]])
    mine = mining.Mine(input)

    state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    state = np.expand_dims(state, 1)
    expectedActions = [(0, 0),(1, 0),(2, 0)]
    actualActions = mine.actions(state)

    for i,child in enumerate(actualActions):
        print(child)
        assert child == expectedActions[i]
    # end

# end

def actions3DTest():
        x = np.array([[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, -1]])
        input = np.array([[[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, 1]], x - 1])
        mine = mining.Mine(input)

        y = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        state = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], y ])
        # state = np.expand_dims(state, 1)
        actualActions = mine.actions(state)
        expectedActions = [(0, 0),(0, 1),(0, 2),(1, 0),(1, 1),(1, 2)]

        for i, child in enumerate(actualActions):
            print(child)
            assert child == expectedActions[i]

def emptyTupleTest():
    input = np.array([[]])
    mine = mining.Mine(input)

    state = np.array([[]])
    state = np.expand_dims(state, 1)
    expectedActions = None
    actualActions = mine.actions(state)

    for i,child in enumerate(actualActions):
        print(child)
        assert child == expectedActions

    # end

# end

def oneTupleValueTest():
    input = np.array([[33]])
    mine = mining.Mine(input)
    state = np.array([[0]])
    state = np.expand_dims(state, 1)
    expectedActions = [(0,0)]
    actualActions = mine.actions(state)

    for i,child in enumerate(actualActions):

        print(child)
        assert child == expectedActions[i]
    # end

# end


def resultTest():
    input = np.array([[-1, -1, 10], [-1, 20, 4], [-1, -1, -1]])
    mine = mining.Mine(input)

    state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    state = np.expand_dims(state, 1)
    expectedActions = ((np.array([1, 1, 1]),), (np.array([0, 0, 0]),), (np.array([0, 0, 0]),))

    actualActions = mine.result(state,(0, 0))

    assert np.array(actualActions).all() == np.array(expectedActions).all()

# end

def resultsTest():
    input = np.array([[-1, -1, 10], [-1, 20, 4], [-1, -1, -1]])
    mine = mining.Mine(input)

    state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    state = np.expand_dims(state, 1)
    expectedActions = [[[2, 2, 2]],[[0, 0, 0]],[[0, 0, 0]]]

    actualActions = mine.results(state,(0, 0))

    assert np.array(actualActions).all() == np.array(expectedActions).all()

# end


def payoffTest():
    input = np.array([[-2, -1, 10], [-1, 20, 7], [-1, -1, -1]])
    mine = mining.Mine(input)

    state = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    state = np.expand_dims(state, 1)


    expectedValue = 33
    actualValue = mine.payoff(state)

    assert actualValue == expectedValue

# end


def is_dangerousTest():
    input = np.array([[-2, -1, 10], [-1, 20, 7], [-1, -1, -1]])
    mine = mining.Mine(underground = input, dig_tolerance=1)

    state = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    state = np.expand_dims(state, 1)


    expected = False
    actual = mine.is_dangerous(state)
    print(actual)

    assert actual == expected

# end


def is_dangerousTest():
    input = np.array([[-2, -1, 10], [-1, 20, 7], [-1, -1, -1]])
    mine = mining.Mine(underground = input, dig_tolerance=1)

    state = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    state = np.expand_dims(state, 1)


    expected = True
    actual = mine.is_dangerous(state)
    print(actual)

    assert actual == expected

# end

def back2DTest():
    input = np.array([[-2, -1, 10], [-1, 20, 7], [-1, -1, -1]])
    mine = mining.Mine(input)
    action = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    state = np.expand_dims(state, 1)


    # expected = False
    [actualaction,actualstate]  = mine.back2D(action,state)
    print(actualaction)
    print(actualstate)
    # assert actual == expected

# end


def validCoordsTest():
    input = np.array([[-2, -1, 10], [-1, 20, 7], [-1, -1, -1]])
    mine = mining.Mine(input)

    expected = False
    actual = mine.validCoords((-1,-1,-1))
    assert actual == expected

    expected = True
    actual = mine.validCoords((0,0,0))
    assert actual == expected

    expected = True
    actual = mine.validCoords((2,0,2))
    assert actual == expected

    expected = False
    actual = mine.validCoords((2,1,2))
    assert actual == expected

# end

def validCoordsTest3D():

    x = np.array([[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, -1]])
    input = np.array([[[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, 1]], x - 1])
    mine = mining.Mine(input)

    expected = False
    actual = mine.validCoords((-1,-1,-1))
    assert actual == expected

    expected = True
    actual = mine.validCoords((0,0,0))
    assert actual == expected

    expected = False
    actual = mine.validCoords((2,0,2))
    assert actual == expected

    expected = False
    actual = mine.validCoords((2,1,2))
    assert actual == expected

    expected = False
    actual = mine.validCoords((2,10,2))
    assert actual == expected

# end

def getRingCoords2DTest():


    input = np.array([[-2, -1, 10], [-1, 20, 7], [-1, -1, -1]])
    mine = mining.Mine(input)

    expected = [(2, 0, 0), (0, 0, 0)]
    actual = getRingCoords(mine,(1,0,1))
    assert actual == expected


    expected = [(2, 0, 0), (1, 0, 0),(0, 0, 0)]
    actual = getRingCoords(mine,(1,1,1))
    assert actual == expected


    expected = []
    actual = getRingCoords(mine,(-1,0,0))
    assert actual == expected


# end

def compareTest():
    numTries = rand.randint(5,10)
    outputList = []

    for _ in range(numTries):
        if rand.choice([True, False]):
            #2D mine test
            size = tuple(np.random.randint(1, 3, (2)))
        else:
            #3D mine test
            size = tuple(np.random.randint(1, 3, (3)))
        #end

        digTolerance = rand.randint(1,10)
        underground = np.random.randint(-10, 10, size)

        mine = mining.Mine(underground, digTolerance)

        dpResult = mining.search_dp_dig_plan(mine)
        bbResults = mining.search_bb_dig_plan(mine)

        outputList.append(str(dpResult) == str(bbResults))

        if outputList[-1] != True:
            print("")
    #end

    assert(False not in outputList)
#end


def getRingCoords3DTest():


    x = np.array([[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, -1]])
    input = np.array([[[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, 1]], x - 1])
    mine = mining.Mine(input)

    expected = [(1, 1, 0), (0, 2, 0), (0, 1, 0)]
    actual = getRingCoords(mine,(1,2,1))
    assert actual == expected


    expected = [(1, 2, 0), (1, 0, 0), (0, 2, 0), (0, 1, 0), (0, 0, 0)]
    actual = getRingCoords(mine,(1,1,1))
    assert actual == expected


    expected = []
    actual = getRingCoords(mine,(-1,0,0))
    assert actual == expected


# end



def getParentsSumTest2D():


    input = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -10, 11]])
    mine = mining.Mine(input)

    state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    state = np.expand_dims(state, 1)
    coords = (1,0,0)
    prevSeenLocs = {}

    expectedsum = -1
    expectedPath = [(1, 0, 0)]
    Sum, Path = getParentsSum(mine,state, coords, prevSeenLocs)



    assert expectedsum == Sum
    assert expectedPath == Path


# end


def getParentsSumTest2D():


    input = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -10, 11]])
    mine = mining.Mine(input)

    state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    state = np.expand_dims(state, 1)
    coords = (1,0,0)
    prevSeenLocs = {}

    expectedsum = -1
    expectedPath = [(1, 0, 0)]
    Sum, Path = getParentsSum(mine,state, coords, prevSeenLocs)



    assert expectedsum == Sum
    assert expectedPath == Path


# end


def main():
    # test1()
    # test2()
    # surface_neigbhours2DTest()
    # surface_neigbhours3DTest()
    # actions2DTest()
    # actions3DTest()
    # emptyTupleTest()
    # oneTupleValueTest()
    # resultTest()
    # resultsTest()
    # payoffTest()
    # is_dangerousTest()
    compareTest()
    # is_dangerousTest()
    # back2DTest()
    # validCoordsTest()
    # validCoordsTest3D()
    # getRingCoords2DTest()
    # getRingCoords3DTest()
    getParentsSumTest2D()
# end

if __name__ == "__main__":
    main()
# end
