from agent import *
import numpy as np
import scipy
import copy
import time
from vector import Vector2D
import pygame

class Scene:

    def __init__(self, c):
        self.c = c  # save the config

        # generate agent data at the positions in mask_grid
        # generate the positions at which we got agents using the specified density
        mask_grid = np.random.uniform(
            size=(c.height, c.width)) < c.initial_population_density

        self.agent_grid = np.full(mask_grid.shape, None, dtype=object)
        for y in range(len(mask_grid)):
            for x in range(len(mask_grid[y])):
                if mask_grid[y, x]:
                    self.agent_grid[y, x] = Agent()

        self.trail_grid = np.zeros((c.height, c.width))  # contains trail data

        # Contains chemo-nutrient/food data. Also need a binary mask of the initial
        # chemo grid, to keep food source values constant across iterations.
        self.chemo_grid = np.zeros((c.height, c.width))
        for food in c.foods:
            self.chemo_grid[
                food[0]:food[0] + c.food_size, food[1]:food[1] + c.food_size
            ] = c.food_deposit

        self.food_grid = self.chemo_grid > 0  # food sources mask grid

        # boolean grid, used to account for possibility that agent is moved to a
        # grid position that has not yet been iterated over, leading to an agent
        # moving multiple times.
        self.to_update = np.full_like(self.agent_grid, True)

        # generate the walls in the grid
        self.wall_mask = np.full_like(self.agent_grid, False)
        for y in range(c.wall_num_height):
            for x in range(c.wall_num_width):
                y_start = c.wall_height + y * 2 * c.wall_height
                x_start = c.wall_width + x * 2 * c.wall_width
                self.wall_mask[
                    y_start:y_start + c.wall_height,
                    x_start:x_start + c.wall_width
                ] = True

        # kill all agents in walls
        self.agent_grid[self.wall_mask == True] = None


    def out_of_bounds(self, pos):
        return pos.x < 0 or pos.y < 0 or \
               pos.y >= self.c.height or pos.x >= self.c.width or \
               self.wall_mask[pos.y, pos.x]


    def rotate_sense(self, agent, coordinate):
        '''Rotates agent towards the sensor with the highest calculated value
           as per Wu et al.'''
        lsensor, rsensor = agent.sensor_positions(self.c)
        lcoord = coordinate + lsensor
        rcoord = coordinate + rsensor

        def sensor_value(coord):
            value = -np.inf
            if not self.out_of_bounds(coord):
                value = self.c.chemo_weight * self.chemo_grid[coord.y, coord.x] + \
                        self.c.trail_weight * self.trail_grid[coord.y, coord.x]
            return value

        # compute values of sensors
        lvalue = sensor_value(lcoord)
        rvalue = sensor_value(rcoord)

        # check if the sensors are out of bounds
        if lvalue == -np.inf and rvalue == -np.inf:
            agent.rotate_180()
            agent.counter -= 1
            return

        if lvalue == -np.inf:
            agent.rotate_right()
            agent.counter -= 1
            return

        if rvalue == -np.inf:
            agent.rotate_left()
            agent.counter -= 1
            return

        # update direction based on which is larger
        if lvalue > rvalue:
            agent.rotate_left()
            return

        if lvalue < rvalue:
            agent.rotate_right()
            return


    def move_agent(self, agent, coordinate):
        '''Moves an agent forward along its direction vector, by one grid position.'''
        new_pos = coordinate + agent.direction()
        agent.counter += 1

        # rotate agent towards sensor with highest value
        self.rotate_sense(agent, new_pos)

        self.agent_grid[new_pos.y, new_pos.x] = copy.deepcopy(agent)
        # flag this position, so that the agent cannot be moved more than once
        # (happens when agent is moved to a position not yet iterated over)
        self.to_update[new_pos.y, new_pos.x] = False

        # remove the agent from the old position
        self.agent_grid[coordinate.y, coordinate.x] = None

        # deposit trail at the new position
        self.trail_grid[new_pos.y, new_pos.x] += self.c.trail_deposit


    def reproduce(self, coordinate):
        '''Randomly initializes a new agent in the position of its parent agent, if
        the parent has exceeded the reproduction trigger threshold.'''
        self.agent_grid[coordinate.y, coordinate.x] = Agent()
        self.to_update[coordinate.y, coordinate.x] = False


    def agent_present(self, coordinate):
        '''Computes a position update for the given agent.'''
        agent = self.agent_grid[coordinate.y, coordinate.x]

        # penalty for being far from food, die earlier
        if self.chemo_grid[coordinate.y, coordinate.x] < self.c.starvation_threshold:
            agent.counter -= self.c.starvation_penalty

        # pickup food when we can
        if self.chemo_grid[coordinate.y, coordinate.x] > self.c.food_pickup_threshold:
            agent.food += self.c.food_pickup_amount
            agent.food = max(agent.food, self.c.food_pickup_limit)
        # drop food
        if agent.food > self.c.food_drop_amount:
            self.chemo_grid[coordinate.y, coordinate.x] += self.c.food_drop_amount
            agent.food -= self.c.food_drop_amount

        # If the elimination trigger is met, remove agent.
        # Otherwise, attempt to move forward.
        if agent.counter < self.c.elimination_trigger:
            self.agent_grid[coordinate.y, coordinate.x] = None
        else:
            new_pos = coordinate + agent.direction()
            if (
                not self.out_of_bounds(new_pos) and \
                self.agent_grid[new_pos.y, new_pos.x] is None
            ):
                self.move_agent(agent, coordinate)

                # if the reproduction trigger is met, generate new agent in the current agent's old position
                if self.agent_grid[new_pos.y, new_pos.x].counter > self.c.reproduction_trigger:
                    self.reproduce(coordinate)
            else:
                agent.random_direction()
                agent.counter -= 1


    def step(self):
        """Perform one update step on the scene by updating each agent in a random order."""
        # generate a shuffled list of coordinates which determines the agent update order.
        # coordinates are only updated if an agent is on them.
        X, Y = np.meshgrid(np.arange(self.c.width), np.arange(self.c.height))
        grid_coordinates = np.vstack((Y.flatten(), X.flatten())).T
        coordinates = np.random.permutation(grid_coordinates)  # [(y, x)]

        self.to_update = np.full_like(self.agent_grid, True)

        # step through all the coordinates and update the agents on those positions
        for coordinate in coordinates:
            coordinate = Vector2D(*coordinate)
            # update agent position and trail grid
            if (
                self.agent_grid[coordinate.y, coordinate.x] is not None and \
                self.to_update[coordinate.y, coordinate.x]
            ):
                self.agent_present(coordinate)

        self.diffuse()


    def diffuse(self):
        """Convolve both the chemo and trail with an average filter after all
        agents have been updated + rotated."""

        # chemo grid
        chemo_kernel = np.ones(
            (self.c.chemo_filter_size, self.c.chemo_filter_size)
        ) * (1 / self.c.chemo_filter_size**2)

        self.chemo_grid = scipy.signal.convolve2d(self.chemo_grid, chemo_kernel, mode='same')
        self.chemo_grid = self.chemo_grid * (1 - self.c.chemo_damping)

        self.chemo_grid = self.chemo_grid * (1 - self.wall_mask.astype(int))  # clip out diffusion into walls

        # reset the values in the food sources to the default
        not_food_grid = self.food_grid == 0
        self.chemo_grid = np.multiply(not_food_grid, self.chemo_grid) + \
            self.food_grid * self.c.chemo_deposit

        # trail grid
        trail_kernel = np.ones(
            (self.c.trail_filter_size, self.c.trail_filter_size)
        ) * (1 / self.c.trail_filter_size**2)

        self.trail_grid = scipy.signal.convolve2d(self.trail_grid, trail_kernel, mode='same')
        self.trail_grid = self.trail_grid * (1 - self.c.trail_damping)

        self.trail_grid = self.trail_grid * (1 - self.wall_mask.astype(int))  # clip out diffusion into walls

    def get_all_adjacents(self, scaled_coord):
        '''Compute all possible nodes adjacent to the current one'''

        # get all possible 4-connected coordinates, and 
        # filter them out if invalid.
        adj_coords = np.zeros((4,2))
        adj_coords[0] = scaled_coord + (-1,0) # up
        adj_coords[1] = scaled_coord + (0,1) # right
        adj_coords[2] = scaled_coord + (1, 0) # down
        adj_coords[3] = scaled_coord + (0, -1) #left

        # compute if coordinates are in walls
        is_odd = adj_coords % 2 == 1
        in_walls = np.all(is_odd, axis=1)

        # compute if coordinates are out of bounds
        max_idx = np.max(self.c.all_coordinates_unscaled, axis=0) 
        too_small = np.any(adj_coords < 0, axis=1)
        too_large = np.any(
            np.concatenate(
                (np.expand_dims(adj_coords[:,0]>max_idx[0], axis=-1), 
                 np.expand_dims(adj_coords[:,1]>max_idx[1], axis=1)), axis=1), axis=1)

        # compute which coordinates have neither of the problems
        is_valid = ~(in_walls | too_small | too_large)
        valid_coords = adj_coords[is_valid]

        # use the mask to get all corresponding node IDs
        adj_coord_idx = [np.squeeze(np.where(
            np.all(self.c.all_coordinates_unscaled == coord, axis=1))).item()
                for coord in valid_coords]

        # return these in the order up, right, down, left. If invalid,
        # set to -1.
        valid_nodes = np.ones(4)*-1
        valid_nodes[is_valid] = adj_coord_idx

        return valid_nodes

    def verify_adjacents(self, trail_patch):
        """Determines which of the edges of the patch are 
           significantly covered by the trail. Returns a boolean
           array, one index for each edge."""

        def get_triangle_masks(n):

            center = (n // 2, n // 2)
            # generate all possible indices
            indices = np.indices((n, n)).reshape(2, -1).T

            # generate one mask (the top)
            top = indices[(indices[:, 0] - indices[:, 1] <= 0) & 
                          (indices[:, 0] + indices[:, 1] <= n - 1)]

            top_mask = np.zeros((n,n))
            top_mask[top[:,0], top[:,1]] = 1

            # rotate it 90 degrees to get the rest
            return (top_mask,
                    np.rot90(top_mask, k=3),
                    np.rot90(top_mask, k=2),
                    np.rot90(top_mask, k=1))

        # divide the square into 4 triangles. The amount of trail in
        # each triangle determines whether an "edge" of slime mold
        # is really present there. 
        triangle_masks = get_triangle_masks(
            trail_patch.shape[0])

        sufficient_trail = np.zeros(4)
        for i, mask in enumerate(triangle_masks):
            sufficient_trail[i] = \
            np.sum(np.multiply(mask, trail_patch)) > self.c.triangle_cutoff

        # only accept this as adjacent direction if the edge of the square
        # in that direction also has trail (otherwise we have a disconnected
        # blob of slime mold)
        trail_on_edge = np.zeros(4)
        trail_on_edge[0] = np.sum(trail_patch[0,:]) > self.c.patch_edge_cutoff # top
        trail_on_edge[1] = np.sum(trail_patch[:,-1]) > self.c.patch_edge_cutoff # right
        trail_on_edge[2] = np.sum(trail_patch[-1,:]) > self.c.patch_edge_cutoff # bottom
        trail_on_edge[3] = np.sum(trail_patch[:,0]) > self.c.patch_edge_cutoff # left

        return np.logical_and(sufficient_trail, trail_on_edge)

    def get_graph(self):
        """Extract graph representation from a scene. Each empty patch
           in the scene represents a node """

        adjacency_list = np.ones((len(self.c.all_coordinates_unscaled), 4))*-1

        # only used for visualization
        patches_filled = np.full(len(self.c.all_coordinates_unscaled), False)


        # for each of the coordinates, determine adjacencies
        for i, unscaled_coord in enumerate(self.c.all_coordinates_unscaled):

            scaled_coord = self.c.all_coordinates_scaled[i]

            mask_grid = 1-(self.agent_grid == None) 

            agent_patch = mask_grid[
                scaled_coord[0]-self.c.wall_height//2 : 
                    scaled_coord[0]+self.c.wall_height//2 + 1,
                scaled_coord[1]-self.c.wall_width//2 : 
                    scaled_coord[1]+self.c.wall_width//2 + 1]

            trail_patch = self.trail_grid[
                scaled_coord[0]-self.c.wall_height//2 : 
                    scaled_coord[0]+self.c.wall_height//2 + 1,
                scaled_coord[1]-self.c.wall_width//2 : 
                    scaled_coord[1]+self.c.wall_width//2 + 1]

            # patch is "filled" when a certain # of agents is 
            # exceeded or if there is a lot of trail deposited.
            # the 'or' allows for more robust formation of 
            # edges around corners (agents may not be present all
            # the time)
            patch_filled = False
            if np.sum(agent_patch) >= self.c.agent_cutoff or \
                    np.sum(trail_patch) > self.c.patch_cutoff:
                        patch_filled = True

            if patch_filled:
                # compute all possible adjacencies. Nodes
                # represented in order up, right, down, left.  
                adjacent_nodes = self.get_all_adjacents(unscaled_coord)
                # determine which adjacencies are actually covered
                # by the patch, and filter the rest out
                covered_by_patch = self.verify_adjacents(trail_patch)
                adjacent_nodes[~covered_by_patch] = -1

                adjacency_list[i] = adjacent_nodes
                patches_filled[i] = True
            # if patch not filled, row in adjacency list remains empty

        return adjacency_list, patches_filled

    def pixelmap(self):
        """Create a pixelmap of the scene on the gpu that can be drawn directly."""
        # create a black and white colormap based on
        # creating a colormap for the walls

        # diagnostics, shows a green pixel in patches considered "filled", and shows
        # green "arms" in directions of valid edges
        if self.c.display_objective_results:
            adjacency_list, patches_filled = self.get_graph()
            idx_filled_patches = np.argwhere(patches_filled)
            filled_patch_coords = np.squeeze(self.c.all_coordinates_scaled[idx_filled_patches])


        agent_colormap = ((1 - ((self.agent_grid != None) | self.wall_mask)) * 255)
        # agent_colormap = ((1 - (self.agent_grid != None)) * 255)
        # agent_colormap = ((1 - self.wall_mask) * 255)

        # create colormap for trails and food source, blue and red respectively
        # upscale trail and chemo maps
        trail_colormap = np.copy(self.trail_grid)
        chemo_colormap = np.copy(self.chemo_grid)
        food_colormap  = np.copy(self.food_grid)

        # To achieve the desired color,the target color channel is set to 255,
        # and the other two are *decreased* in proportion to the value in the
        # trail/chemo map. This makes low values close to white, and high
        # values a dark color.
        red_channel = np.full_like(agent_colormap, 255)
        green_channel = np.full_like(agent_colormap, 255)
        blue_channel = np.full_like(agent_colormap, 255)

        if self.c.display_chemo:
            # TODO make chemo spreading visual when trails are also visible
            # intensity transformation, strictly for visual purposes
            # clipping the map back to [0, 255]
            intensity = 100
            chemo_colormap = np.minimum(intensity * chemo_colormap, 255)
            chemo_colormap = np.full_like(chemo_colormap, 255) - chemo_colormap # inverting the map

            red_channel = np.full_like(chemo_colormap, 255)
            green_channel = chemo_colormap
            blue_channel = np.copy(chemo_colormap)

        if self.c.display_trail:
            # intensity transformation, strictly for visual purposes
            # clipping the map back to [0, 255]
            intensity = 20
            trail_colormap = np.minimum(intensity * trail_colormap, 255)
            trail_colormap = np.full_like(trail_colormap, 255) - trail_colormap # inverting the map

            trail_pixels = trail_colormap < 255
            not_trail_pixels = trail_colormap == 255

            red_channel = red_channel * not_trail_pixels + trail_colormap * trail_pixels
            green_channel = green_channel * not_trail_pixels + trail_colormap * trail_pixels
            blue_channel = blue_channel * not_trail_pixels + np.full_like(blue_channel, 255) * trail_pixels

        if self.c.display_agents:
            agent_pixels = agent_colormap == 0
            not_agent_pixels = agent_colormap == 255

            red_channel = red_channel * not_agent_pixels + agent_colormap * agent_pixels
            green_channel = green_channel * not_agent_pixels + agent_colormap * agent_pixels
            blue_channel = blue_channel * not_agent_pixels + agent_colormap * agent_pixels

        if self.c.display_food:
            # placing food sources on top of everything
            food_pixels = food_colormap > 0
            not_food_pixels = food_colormap == 0

            red_channel = red_channel * not_food_pixels + np.full_like(red_channel, 255) * food_pixels
            green_channel = green_channel * not_food_pixels + np.zeros_like(green_channel) * food_pixels
            blue_channel = blue_channel * not_food_pixels + np.zeros_like(blue_channel) * food_pixels

        if self.c.display_objective_results:
            # place a green pixel at the center of each filled patch
            red_channel[filled_patch_coords[:,0], filled_patch_coords[:,1]] = 0
            blue_channel[filled_patch_coords[:,0], filled_patch_coords[:,1]] = 0
            green_channel[filled_patch_coords[:,0], filled_patch_coords[:,1]] = 255

            # create green "arms" in direction of objective function edges
            for i, coord in enumerate(self.c.all_coordinates_scaled):
                for direction, adj in enumerate(adjacency_list[i]):
                    # up
                    if adj != -1 and direction == 0:
                        red_channel[coord[0]-self.c.wall_height//2:coord[0], coord[1]] = 0
                        blue_channel[coord[0]-self.c.wall_height//2:coord[0], coord[1]] = 0
                        green_channel[coord[0]-self.c.wall_height//2:coord[0], coord[1]] = 255
                    # right
                    elif adj != -1 and direction == 1:
                        red_channel[coord[0], coord[1]+1:coord[1]+1+self.c.wall_width//2] = 0
                        blue_channel[coord[0], coord[1]+1:coord[1]+1+self.c.wall_width//2] = 0
                        green_channel[coord[0], coord[1]+1:coord[1]+1+self.c.wall_width//2] = 255
                    # down
                    elif adj != -1 and direction == 2:
                        red_channel[coord[0]+1:coord[0]+1+self.c.wall_height//2, coord[1]] = 0
                        blue_channel[coord[0]+1:coord[0]+1+self.c.wall_height//2, coord[1]] = 0
                        green_channel[coord[0]+1:coord[0]+1+self.c.wall_height//2, coord[1]] = 255
                    # left
                    elif adj != -1 and direction == 3:
                        red_channel[coord[0], coord[1]-1-self.c.wall_width//2:coord[1]] = 0
                        blue_channel[coord[0], coord[1]-1-self.c.wall_width//2:coord[1]] = 0
                        green_channel[coord[0], coord[1]-1-self.c.wall_width//2:coord[1]] = 255                        

        pixelmap = np.stack(
            (red_channel.astype(np.uint8), green_channel.astype(np.uint8), blue_channel.astype(np.uint8)),
            axis=-1
        )

        # transpose from shape (height, width, 3) to (width, height, 3) for pygame
        transposed_pixelmap = np.transpose(pixelmap, (1, 0, 2))
        scaled_pixelmap = transposed_pixelmap.repeat(self.c.upscale, axis=0).repeat(self.c.upscale, axis=1)

        return scaled_pixelmap