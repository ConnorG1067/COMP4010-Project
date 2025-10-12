from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    stay = 4

### GridMap Envrionment
# Sand
# Trees
# Sidewalks
# Charging station
# Garbage
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self._map_matrix = np.array([
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
        ])

        self.size = len(self._map_matrix)  # The size of the square grid
        self.window_size = 664  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "garbage_disposal": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "stay"
        self.action_space = spaces.Discrete(5)

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
            Actions.stay.value: np.array([0, 0]),
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
        return {"agent": self._agent_location, "garbage_disposal": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        non_zero_coords = np.argwhere(self._map_matrix != 0)
        self._agent_location = non_zero_coords[np.random.randint(0, len(non_zero_coords))]

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = non_zero_coords[np.random.randint(0, len(non_zero_coords))]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3,4}) to the direction we walk in
        direction = self._action_to_direction[action]

        new_coords = self._agent_location + direction
        if((new_coords[0] < 13 and new_coords[1] < 13) and (new_coords[0] >= 0 and new_coords[1] >= 0)):
            self._agent_location = new_coords if self._map_matrix[new_coords[0]][new_coords[1]] == 1 else self._agent_location

        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
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
        
    def _map_initialization(self, canvas, square_size):
        for row in range(len(self._map_matrix)):
            for col in range(len(self._map_matrix[row])):
                rect = pygame.Rect(col*square_size, row*square_size, square_size, square_size)
                pygame.draw.rect(
                    surface=canvas,
                    color=(0, 0, 0),
                    rect=rect,
                    width=1 if self._map_matrix[row][col] == 1 else 0,
                )
        
    # Load the agent
    def _render_agent(self, canvas, pix_square_size):
        agent_img = pygame.image.load("./fa_env/env_assets/roomba.png")
        agent_img = pygame.transform.scale(
            agent_img,
            (int(pix_square_size * 0.8), int(pix_square_size * 0.8))
        )

        agent_pos = (self._agent_location + 0.5) * pix_square_size
        rect = agent_img.get_rect(center=agent_pos)
        
        canvas.blit(agent_img, rect)
    
    def _render_garbage_disposal(self, canvas, pix_square_size):
        garbage_disposal_img = pygame.image.load("./fa_env/env_assets/trashcan.png")
        garbage_disposal_img = pygame.transform.scale(
            garbage_disposal_img,
            (int(pix_square_size * 0.8), int(pix_square_size * 0.8))
        )

        garbage_pos = (self._target_location + 0.5) * pix_square_size

        rect = garbage_disposal_img.get_rect(center=garbage_pos)
        canvas.blit(garbage_disposal_img, rect)

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
            self.window_size / len(self._map_matrix)
        )  # The size of a single grid square in pixels

        self._render_garbage_disposal(canvas, pix_square_size)

        self._render_agent(canvas, pix_square_size)

        self._map_initialization(canvas, pix_square_size)
        # # Finally, add some gridlines
        # for x in range(self.size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )

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

