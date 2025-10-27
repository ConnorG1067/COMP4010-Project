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
    ['g', 'g', 'g', 's', 's', 'w', 'w', 'w', 's', 's', 'g', 'g', 'g'],
    ['g', 'g', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'g', 'g'],
    ['g', 'g', 'p', 's', 's', 'w', 'w', 'w', 's', 's', 'p', 'g', 'g'],
    ['g', 'g', 'p', 's', 's', 'w', 'w', 'w', 's', 's', 'p', 'g', 'g'],
    ['w', 'w', 'p', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'p', 'w', 'w'],
    ['w', 'w', 'p', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'p', 'w', 'w'],
    ['w', 'w', 'p', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'p', 'w', 'w'],
    ['g', 'g', 'p', 's', 's', 'w', 'w', 'w', 's', 's', 'p', 'g', 'g'],
    ['g', 'g', 'p', 's', 's', 'w', 'w', 'w', 's', 's', 'p', 'g', 'g'],
    ['g', 'g', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'g', 'g'],
    ['g', 'g', 'g', 'g', 'g', 'w', 'w', 'w', 'g', 'g', 'g', 'g', 'g'],
    ['g', 'g', 'g', 'g', 'g', 'w', 'w', 'w', 'g', 'g', 'g', 'g', 'g']
])

OBJECT_ASSETS = {  
    'c': pygame.image.load('./fa_env/env_assets/charge_1.png'),
    'b': pygame.image.load('./fa_env/env_assets/obstacle_10.png'),  
    't': pygame.image.load('./fa_env/env_assets/trashcan_1.png'),
    'l': pygame.image.load('./fa_env/env_assets/garbage.png')
}

MISC_ASSETS = {
    'garbage' : pygame.image.load("./fa_env/env_assets/garbage.png"),
}

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    stay = 4  # wait/recharge
    pick_up = 5
    drop_off = 6

### GridMap Envrionment
#Terrain Types
#p - pavement no chance of getting stuck, movment cost from pavement is normal
#g - Grass 0.01 chance of getting stuck, movment cost from grass is double
#s - Sand 0.05 chance of getting stuck, movment cost from sand is triple
#w - Water a phsical barrier that cant be interacted with similer to borders

#Object types
#c - charging station charges robots battery by x% for each step that ends on it
#b - bush/obstacles a phsical barrier that cant be interacted with similer to borders
#t - target/trash bin the dropoff location for litter
#l - litter/garbage to be collected by robot
class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        self.base_map = np.array([list(row) for row in MAP])
        self.map = self.base_map.copy()
        self.size = len(MAP)
        self.window_size = 416  # The size of the PyGame window
        self.iteration_count = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                'agent': spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                'target': spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 7 actions, corresponding to 'right', 'up', 'left', 'down', 'stay','pick up',and 'drop off'
        self.action_space = spaces.Discrete(7)

        '''
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to 'right', 1 to 'up' etc.
        '''
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
            Actions.stay.value: np.array([0, 0]),
            Actions.pick_up.value: np.array([0, 0]),
            Actions.drop_off.value: np.array([0, 0])
        }

        assert render_mode is None or render_mode in self.metadata['render_modes']
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

    def _get_obs(self):
        return {'agent': self._agent_location, 'garbage_disposal': self._target_location}

    def _get_info(self):
        return {
            'distance': np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        #Reset the Map
        self.map = self.base_map.copy()

        #Reset the Battery
        self._agent_Battery = 100.00

        #Reset held garbage
        self._agent_held_garbage = 0
        self._agent_max_held_garbage = 3

        self.iteration_count = 0

        self.env_base_garbage_count = random.randint(3, 6)
        self.env_garbage_count = self.env_base_garbage_count

        #Place charging station
        charging_space = random.choice(np.argwhere(np.isin(self.map, ['p'])))

        self.map[charging_space[1]][charging_space[0]] = 'c'
        self.map[charging_space[0]][charging_space[1]] = 't'
        self._target_location = np.array([charging_space[0], charging_space[1]]) 

        #Place bushes
        for i in range(9):
            grass_spaces = np.argwhere(self.map == 'g')
            space = random.choice(grass_spaces)
            self.map[space[1]][space[0]] = 'b'

        #Idea for Connor, to prevent getting trapped do as i did with bushes, but place agent only on pavement, target only on sand

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        accessible_coords = accessible_coords = np.argwhere(np.isin(self.map, ['g','s','p']))
        
        # Find garbage location
        random_garb_coords = [accessible_coords[index] for index in (np.random.choice(len(accessible_coords), self.env_base_garbage_count, replace=False))]
        for coord in random_garb_coords: self.map[coord[1]][coord[0]] = 'l'
        accessible_coords = accessible_coords = np.argwhere(np.isin(self.map, ['p']))
        

        # Choose the agent's location uniformly at random (Roomba)
        self._agent_location = accessible_coords[np.random.randint(0, len(accessible_coords))]
        # Explaination 
        accessible_coords = accessible_coords[~np.all(accessible_coords == self._agent_location, axis=1)]

        if self.render_mode == 'human':
            self._render_frame()
        

        return self._get_obs(), self._get_info()

    def step(self, action):
        #base variables
        terminated = False
        reward = 0  
        self.iteration_count += 1
        
        # Map the action (element of {0,1,2,3,4}) to the direction we walk in
        direction = self._action_to_direction[action]
        old_coords = self._agent_location
        new_coords = self._agent_location + direction

        if((new_coords[0] < 13 and new_coords[1] < 13) and (new_coords[0] >= 0 and new_coords[1] >= 0) and (str(self.map[new_coords[1]][new_coords[0]]).strip() not in ['w', 'b'])):
            self._agent_location = new_coords

        #Terrain effcts and battery decay, robot is effected by predvous tile
        tile_type = self.map[old_coords[1]][old_coords[0]]
        random_number = random.random() #random float between 0 and 1

        #Base battery useage
        self._agent_Battery-= 0.05
        if action < 4:      #Movement
            movement_cost = 0.2 
            if tile_type == 'g':
                if(random_number < 0.01):
                    terminated = True
                    reward -= 0.5
                    print("Stuck in grass")
                self._agent_Battery-=movement_cost*2
            elif tile_type == 's':
                if(random_number < 0.05):
                    terminated = True
                    reward -= 0.5
                    print("Stuck in sand")
                self._agent_Battery-=movement_cost*3
            else:
                self._agent_Battery-=movement_cost

        elif action == 4:       #Recharge battery
            if tile_type == 'c':
                if(self._agent_Battery + 5 > 100):
                    self._agent_Battery = 100
                else:
                    self._agent_Battery+=5

                if self._agent_Battery < 80:
                    reward += 0.05
                elif(self._agent_Battery >= 99):
                    reward -= 0.01
                else:                    
                    reward += 0.01

                
                print("Charging")

        elif action == 5:       #Pick up litter
            if tile_type == 'l' and (self._agent_held_garbage<self._agent_max_held_garbage):
                reward += 0.2
                self._agent_held_garbage+=1
                self.env_garbage_count-=1
                self.map[old_coords[1]][old_coords[0]] = self.base_map[old_coords[1]][old_coords[0]]
                print("Picked up litter")
            else:
                reward -= 0.05
                print("Nothing to Picked up")
            self._agent_Battery-=0.1

        elif action == 6:       #Drop off litter
            if tile_type == 't' and self._agent_held_garbage>0:
                reward += 0.5*self._agent_held_garbage
                self._agent_held_garbage=0
                print("drop off litter")
            else:
                reward -= 0.05
                print("Cant drop off not at trashcan") if tile_type == 't' else None
                print("nothing to drop off") if self._agent_held_garbage == 0 else None
                
            self._agent_Battery-=0.1
        


        #Map cleared
        if(self.env_garbage_count==0 & self._agent_held_garbage == 0):
            terminated = True
            reward += 1
            print("Map cleared")
        
        #Battery died
        elif(self._agent_Battery<=0):
            terminated = True
            reward -= 1
            print("battery died")
        
        #Render pygame
        if self.render_mode == 'human':
            self._render_frame()

        return self._get_obs(), reward, terminated, self.iteration_count >= MAX_ITERATIONS, self._get_info()


    #Doesnt do anything?
    #def render(self):
    #    if self.render_mode == 'rgb_array':
    #       return self._render_frame()
        
    #Render Agent
    def _render_agent(self, canvas, pix_square_size):
        agent_img = pygame.image.load('./fa_env/env_assets/roomba.png')
        agent_img = pygame.transform.scale(
            agent_img,
            (int(pix_square_size * 0.8), int(pix_square_size * 0.8))
        )

        agent_pos = (self._agent_location + 0.5) * pix_square_size
        rect = agent_img.get_rect(center=agent_pos)
        
        canvas.blit(agent_img, rect)
    
    #Render Objects
    def _render_objects(self, canvas, square_size):
        object_spaces = np.argwhere(np.isin(self.map, ['c', 'b','t','l']))

        for row, col in object_spaces:
            object_type = self.map[row][col]
            if object_type in OBJECT_ASSETS:
                landscape = pygame.transform.scale(
                    OBJECT_ASSETS[object_type],
                    (square_size, square_size)
                )
                canvas.blit(landscape, (square_size * col, square_size * row))

    #Render battery display
    def _render_battery(self, canvas):
        font = pygame.font.SysFont(None, 24)
        battery_text = font.render(f"Battery: {self._agent_Battery:.2f}%", True, (255,0,0))
        canvas.blit(battery_text, (5,5))

    def _render_frame(self):
        #we can remove self.render_mode == 'human',render_frame is only called when render_mode=='human'
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('Cleaning Robot - Demo')
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        #Setup Clock
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        #Setup Pygame window
        canvas = pygame.Surface((self.window_size, self.window_size))
        background = pygame.image.load('./fa_env/env_assets/map_1.png')
        canvas.blit(background, (0, 0))

        # The size of a single grid square in pixels
        pix_square_size = ( self.window_size / len(self.map))  

        self._render_objects(canvas, pix_square_size)

        self._render_agent(canvas, pix_square_size)

        self._render_battery(canvas)

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    # def _render_garbage(self, canvas, pix_square_size):
        # garbage_img = pygame.transform.scale(
        #     MISC_ASSETS['garbage'],
        #     (int(pix_square_size * 0.8), int(pix_square_size * 0.8))
        # )

        # for coord in random_garb_coords:
        #     rect = garbage_img.get_rect(center=coord)
        
            # canvas.blit(garbage_img, rect)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()