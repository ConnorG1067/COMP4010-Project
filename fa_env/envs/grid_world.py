from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random

# Subject to change based on future requirements
MAX_ITERATIONS = 20000

MAP = np.array([
    ['g', 'g', 'g', 's', 's', 'w', 'w', 'w', 's', 's', 'g', 'g', 'g'],
    ['g', 'b', 'g', 's', 's', 'w', 'w', 'w', 's', 's', 'b', 'g', 'g'],
    ['g', 'g', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'g', 'g'],
    ['g', 'g', 'p', 's', 's', 'w', 'w', 'w', 's', 's', 'p', 'g', 'g'],
    ['g', 'g', 'p', 's', 's', 'w', 'w', 'w', 's', 's', 'p', 'g', 'g'],
    ['w', 'w', 'p', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'p', 'w', 'w'],
    ['w', 'w', 'p', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'p', 'w', 'w'],
    ['w', 'w', 'p', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'p', 'w', 'w'],
    ['g', 'g', 'p', 's', 's', 'w', 'w', 'w', 's', 's', 'p', 'g', 'g'],
    ['g', 'b', 'p', 's', 'b', 'w', 'w', 'w', 's', 's', 'p', 'g', 'g'],
    ['g', 'g', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'g', 'g'],
    ['b', 'g', 'g', 'g', 'g', 'w', 'w', 'w', 'g', 'g', 'g', 'b', 'g'],
    ['g', 'g', 'g', 'g', 'g', 'w', 'w', 'w', 'g', 'g', 'g', 'g', 'g']
])

### GridMap Envrionment
#Terrain Types
#p - pavement no chance of getting stuck, movment cost from pavement is normal
#g - Grass 0.01 chance of getting stuck, movment cost from grass is double
#s - Sand 0.05 chance of getting stuck, movment cost from sand is triple
#w - Water a phsical barrier that cant be interacted with similer to borders

# Object types
#c - charging station charges robots battery by x% for each step that ends on it
#b - bush/obstacles a phsical barrier that cant be interacted with similer to borders
#t - target/trash bin the dropoff location for litter
#l - litter/garbage to be collected by robot
class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, terrain_map=MAP, render_mode=None, is_training=True):
        self.size = len(MAP)
        self.window_size = 416  # The size of the PyGame window

        self.terrain_map = MAP
        self.grid_rows, self.grid_cols = terrain_map.shape

        self._agent_max_battery = 100.00  
        self._agent_battery = self._agent_max_battery 
        self._charging_station_position = (2, 2) 
        
        self._max_iteration_count = 1000000
        self._iteration_count = 0

        # Status: uncollected, collected,and deposited
        self._trash_bin_position = (10, 10) 
        self._garbage = [{"location": (12, 0), "status": "uncollected"},
                        {"location": (1, 3), "status": "uncollected"},
                        {"location": (8, 11), "status": "uncollected"}]
        
        # Used in rendering
        self._agent_held_garbage = 0
        self._agent_max_held_garbage = 1
        self.env_base_garbage_count = 3
        self._env_garbage_count = 3
    
        # We have 7 actions, corresponding to 'right', 'up', 'left', 'down', 'recharge','pick up',and 'drop off'
        self.action_space = spaces.Discrete(7)

        
        # Observation space: Discrete encoding, Map size * battery states * (garbage status * # of garbage)
        self.observation_space = spaces.Discrete(
            self.grid_rows * self.grid_cols * 4 * (3 ** self.env_base_garbage_count)
        )

        # Assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        '''
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        '''
        self.window = None
        self.clock = None

    def battery_state(self):
        # Convert battery to discrete state
        if self._agent_battery <= 25: return 0  # Low
        elif self._agent_battery <= 50: return 1  # Medium
        elif self._agent_battery <= 85: return 2  # High
        else: return 3  # Max

    def reset(self):
        while True:
            r = random.randint(0, self.grid_rows - 1)
            c = random.randint(0, self.grid_cols - 1)
            if self.terrain_map[r][c] not in ['w', 'b']:
                self.agent_row, self.agent_col = r, c
                break
        
        self._iteration_count = 0
        self._agent_battery = self._agent_max_battery
        self._env_garbage_count = 3
        self._agent_held_garbage = 0

        for g in self._garbage:
            g["status"] = "uncollected"
        return self.encode_state()

    def encode_state(self):
        status_map = {'uncollected': 0, 'collected': 1, 'deposited': 2}
        garbage_code = 0
        base = 3
        for i, p in enumerate(self._garbage):
            garbage_code += status_map[p["status"]] * (base ** i)
        
        state_idx = (
            self.agent_row * self.grid_cols * 4 * (base ** self.env_base_garbage_count) +
            self.agent_col * 4 * (base ** self.env_base_garbage_count) +
            self.battery_state() * (base ** self.env_base_garbage_count) +
            garbage_code
        )
        return state_idx

    def step(self, action):
        # Base Variables
        terminated = False
        truncated = False
        reward = -0.01  
        movement_cost = 1
        action_cost = 0.5

        self._iteration_count += 1
        

        # Movement
        #if action in [0, 1, 2, 3]:
        if action < 4:
            new_row, new_col = self.agent_row, self.agent_col
            if action == 0: 
                new_row = self.agent_row - 1
                if self.render_mode == 'human': print("Up")
            elif action == 1: 
                new_row =  self.agent_row + 1
                if self.render_mode == 'human': print("Down")
            elif action == 2: 
                new_col =  self.agent_col + 1
                if self.render_mode == 'human': print("Right")
            elif action == 3: 
                new_col = self.agent_col - 1
                if self.render_mode == 'human': print("Left")

            # Check walls
            if((new_row < 0 or new_col < 0) or (new_row > 12 or new_col > 12) or self.terrain_map[new_row][new_col] in ['w', 'b']):
                reward -= 0.1  # penalty for trying to move into wall
            else:
                # Update position
                self.agent_row, self.agent_col = new_row, new_col
                
                # Apply terrain effects
                # terrain = self.terrain_map[self.agent_row][self.agent_col]
                # if terrain == 'g' :
                #     self._agent_battery -= movement_cost * 2
                # elif terrain == 's' :
                #     self._agent_battery -= movement_cost * 4
                # else: # pavment
                self._agent_battery -= movement_cost
        
        # Recharge
        elif action == 4:
            if self.render_mode == 'human': print("Recharge")
            if (self.agent_row, self.agent_col) == self._charging_station_position:
                self._agent_battery = min(self._agent_battery+5, self._agent_max_battery)
                # if self.battery_state() == 0:
                #     reward += 0.1
                # elif self.battery_state() == 1:
                #     reward += 0.05
                # elif self.battery_state() == 2:
                #     reward += 0.01
                # else:
                #     reward -= 0.1
            else:
                self._agent_battery -= action_cost


        # Pickup
        elif action == 5:
            if self.render_mode == 'human': print("Pick up")
            success = False
            for g in self._garbage:
                if g["status"] == "uncollected" and g["location"] == (self.agent_row, self.agent_col) and self._agent_held_garbage < self._agent_max_held_garbage:
                    g["status"] = "collected"
                    reward += 0.2
                    success = True
                    self._agent_held_garbage += 1
                    self._env_garbage_count -= 1

            self._agent_battery -= action_cost
            if not success:
                reward -= 0.1
        
        # Dropoff
        elif action == 6:
            if self.render_mode == 'human': print("Drop off")
            success = False
            if (self.agent_row, self.agent_col) == self._trash_bin_position:
                for g in self._garbage:
                    if g["status"] == "collected":
                        g["status"] = "deposited"
                        reward += 0.5
                        success = True
                        self._agent_held_garbage = 0
            
            self._agent_battery -= action_cost
            if not success:
                reward = -0.1
        
        # Map cleared
        if all(g["status"] == "deposited" for g in self._garbage) and not terminated:
            terminated = True
            reward += 1

        # Battery died
        elif self._agent_battery <= 0 and not terminated:
            terminated = True
            reward -= 1

        if(self._iteration_count > self._max_iteration_count):
            truncated = True
        
        # Render pygame
        if self.render_mode == 'human':
            self._render_frame()

        return self.encode_state(), reward, terminated, truncated, {}


    ###########################################################################################################
    # Redering Methods
    ###########################################################################################################
    def _render_agent(self, canvas, pix_square_size):
        agent_img = pygame.image.load('./fa_env/env_assets/roomba.png')
        agent_img = pygame.transform.scale(
            agent_img,
            (int(pix_square_size * 0.8), int(pix_square_size * 0.8))
        )

        # Compute agent position using row and column
        agent_x = (self.agent_col + 0.5) * pix_square_size
        agent_y = (self.agent_row + 0.5) * pix_square_size
        agent_pos = (agent_x, agent_y)

        rect = agent_img.get_rect(center=agent_pos)
        canvas.blit(agent_img, rect)


    def _render_objects(self, canvas, square_size):
        # Objects
        recharge = pygame.image.load('./fa_env/env_assets/charge_1.png')
        trashbin = pygame.image.load('./fa_env/env_assets/trashcan_1.png')
        bush = pygame.image.load('./fa_env/env_assets/obstacle_10.png') 
        garbage = pygame.image.load('./fa_env/env_assets/garbage.png')

        # Recharge
        canvas.blit(
            pygame.transform.scale(recharge,(square_size, square_size)),
            (square_size * self._charging_station_position[1], square_size * self._charging_station_position[0])
            )
        # Bin
        canvas.blit(
            pygame.transform.scale(trashbin,(square_size, square_size)),
            (square_size * self._trash_bin_position[1], square_size * self._trash_bin_position[0])
            )
        
        # Bushes
        # bush_spaces = np.argwhere(np.isin(self.terrain_map, ['b']))
        # for row, col in bush_spaces:
        #     canvas.blit(
        #         pygame.transform.scale(bush, (square_size, square_size)),
        #         (square_size * col, square_size * row)
        #         )

        # Garbage
        for g in self._garbage:
            if g["status"] == "uncollected":
                canvas.blit(
                    pygame.transform.scale(garbage,(square_size, square_size)),
                    (square_size * g["location"][1], square_size * g["location"][0])
                    )

    def _render_battery_text(self, canvas, text_color):
        font = pygame.font.SysFont(None, 24)
        battery_text = font.render(f"Battery: {self._agent_battery:.2f}%", True, text_color)
        canvas.blit(battery_text, (10,5))
    
    def _render_trash_text(self, canvas, text_color):
        font = pygame.font.SysFont(None, 24)
        agent_text = font.render(f"Agent Trash: {self._agent_held_garbage}/{self._agent_max_held_garbage}", True, text_color)
        env_text = font.render(f"Env Trash: {self._env_garbage_count}/{self.env_base_garbage_count}", True, text_color)
        canvas.blit(agent_text, (10,25))
        canvas.blit(env_text , (10,45))

    def _render_frame(self):
        # We can remove self.render_mode == 'human',render_frame is only called when render_mode=='human'
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('Cleaning Robot - Demo')
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        # Setup Clock
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Setup Pygame window
        canvas = pygame.Surface((self.window_size, self.window_size))
        background = pygame.image.load('./fa_env/env_assets/map_1.png')
        canvas.blit(background, (0, 0))

        # The size of a single grid square in pixels
        pix_square_size = ( self.window_size / len(self.terrain_map))  

        # Reder Assets
        self._render_objects(canvas, pix_square_size)
        self._render_agent(canvas, pix_square_size)

        # Render Text
        text_color = (165, 0, 168)
        self._render_battery_text(canvas, text_color)
        self._render_trash_text(canvas, text_color)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to
        # keep the framerate stable.
        self.clock.tick(self.metadata['render_fps'])

        # Does not end run nicely, the results will not be saved
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()