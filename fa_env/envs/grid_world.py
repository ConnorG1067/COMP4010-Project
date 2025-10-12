from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random

MAP = [
        "gggggwwwggggg",    #1
        "gggggwwwggggg",
        "gggggpppggggg",
        "gggsswwwssggg",
        "gggsswwwssggg",    #5
        "wwpwwwwwwwpww",
        "wwpwwwwwwwpww",
        "wwpwwwwwwwpww",
        "ggpsswwwsspgg",
        "gggsswwwssggg",
        "gggggpppggggg",
        "gggggwwwggggg",
        "gggggwwwggggg"   #13

]

TERRAIN_TYPES = {
    's': (252, 186, 3),  # Sand - yellow
    'p': (100, 100, 100),  # Pavement - gray
    'g': (0, 255, 0),      # Grass - green
    'r': (255, 255, 0),    # Respawn - yellow
    't': (255, 0, 0),       # Target - red
    'b': (17, 54, 4),    # Bush - dark green)
    'w': (65, 90, 217)        # Bush - dark green)
}

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    # 
    #


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):

        self.base_map = np.array([list(row) for row in MAP])
        self.map = self.base_map.copy()
        self.size = len(MAP)    
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the Map
        self.map = self.base_map.copy()
        
        #Place respawn
        pavment_spaces = np.argwhere(self.map == 'p')
        space = random.choice(pavment_spaces)
        self.map[space[0], space[1]] = 'r'

        #Place target
        sand_spaces = np.argwhere(self.map == 's')
        space = random.choice(pavment_spaces)
        self.map[space[0], space[1]] = 't'
        
        #Place bushes
        for i in range(9):
            grass_spaces = np.argwhere(self.map == 'g')
            space = random.choice(grass_spaces)
            self.map[space[0], space[1]] = 'b'


        # Respawn
        respawn = np.argwhere(self.map == 'r')[0]
        self._agent_location = np.array(respawn)

        # Target
        target = np.argwhere(self.map == 't')[0]
        self._target_location = np.array(target)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels


        # Draw map
        for i in range(self.size):
            for j in range(self.size):
                color = TERRAIN_TYPES[self.map[i, j]]
                pygame.draw.rect(canvas, color, pygame.Rect(j * pix_square_size, i * pix_square_size, pix_square_size, pix_square_size))


        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

