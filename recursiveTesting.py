import mining
import random as rand
import numpy as np
import time


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

    actualActions = mine.actions( state)
    for child in actualActions:
        print(next(child))
    # expectedActions = [(0, 0), (2, 0)]
    # assert actualActions == expectedActions

def actions3DTest():
        x = np.array([[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, -1]])
        input = np.array([[[1, 4, 1, 1], [2, 5, 1, 1], [3, 6, 1, 1]], x - 1])
        mine = mining.Mine(input)

        state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        actualActions = mine.actions(mine, state)
        for child in actualActions:
            print(next(child))
        # expectedActions = [(0, 0), (2, 0)]
        # assert actualActions == expectedActions



# end

def main():
    # test1()
    # test2()
    # surface_neigbhours2DTest()
    # surface_neigbhours3DTest()
    actions2DTest()
# end

if __name__ == "__main__":
    main()
# end
