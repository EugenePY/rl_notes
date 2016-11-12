import gym
import numpy as np
from gym import spaces
from skimage.draw import bezier_curve
import copy

class GridMap(gym.Env):
    """
    Simplest Env :
               ________
              | X |  O | end
              |___|____|
              | O |  O |
        begin |___|____|

    The grid is (grid_length, grid_width)
    """

    actionmapping = {0: (0, 1),  # right
                     1: (1, 0),  # down
                     2: (0, -1),  # left
                     3: (-1, 0),  # up
                     4: (0, 0)}  # do not move
    action = {0: 'right',
              1: 'down',
              2: 'left',
              3: 'up',
              4: 'stay'}

    def __init__(self, grid_length, grid_width):
        self.grid_length = grid_length
        self.grid_width = grid_width
        self.grid_map_loc2score = np.random.normal(
            size=(grid_length, grid_width))

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(self.grid_width *
                                                 self.grid_length)
        # prepare the grid map
        mid_pnt = (grid_length // 2, grid_width // 2)
        goal = (grid_length-1, grid_width-1)
        self.grid_map_loc2score[goal] = 2
        rr, cc = bezier_curve(0, 0, mid_pnt[0], mid_pnt[1],
                              goal[0], goal[1], -0.9)

        # create the map for visualiztion
        MAP = ''
        MAP += ('+---' * self.grid_width + '+\n' +
                '| X ' * self.grid_width + '|\n') * self.grid_length
        MAP += '+---' * self.grid_width + '+\n'
        MAP = bytearray(MAP)
        self._ROW_TXNUM = (4 * self.grid_width + 2) * 2
        self._COL_TXNUM = 4
        for row, col in zip(rr, cc):
            MAP[row * self._ROW_TXNUM + self._ROW_TXNUM//2 +
                self._COL_TXNUM * col + 3 - 1] = 'O'
        self._BEST_ROUTE = zip(rr, cc)
        self.map = MAP
        self.grid_map_loc2score[rr, cc] = 1

        self.steps_before_done = 0
        self.reset()

    def _random_trial(self):
        pass

    def _draw(self, state):
        row, col = state
        current_map = copy.copy(self.map)
        current_map[(row) * self._ROW_TXNUM + (col) * self._COL_TXNUM + 3 +
                    self._ROW_TXNUM//2 - 1] = '@'
        return current_map

    def _reset(self):
        self.state = (0, 0)  # the start point
        self.steps_before_done = 0
        return self.state

    def _step(self, action):
        """
        agent have given an action and the env respond a reward to the agent.
        """
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        loc_x, loc_y = np.array(self.state) + \
            np.array(self.actionmapping[action])
        done = loc_x == self.grid_width - 1 and loc_y == self.grid_length - 1
        done = bool(done)

        # out of the map dead!
        outofmap = loc_x < 0 or loc_y < 0 or \
            loc_x >= self.grid_width or loc_y >= self.grid_length
        if outofmap:
            reward = -100
            self.steps_before_done += 1
            self.state = self.state
        else:
            reward = self.grid_map_loc2score[loc_x, loc_y]
            self.state = (loc_x, loc_y)
            self.steps_before_done += 1
        if done:
            self.reward = 10
            self.state = (loc_x, loc_y)
        return self.state, reward, done, {}

    def _render(self, mode='human', close=False):
        return NotImplementedError

if __name__ == '__main__':
    env = GridMap(grid_width=10, grid_length=19)
    print '----------- MAP 19x10 ----------'
    print env.map
