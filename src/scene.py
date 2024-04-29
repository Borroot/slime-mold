from agent import *
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import collections
import numpy as np



def scene_init(config_scene, config_chemo, key):
    height, width, _, initial_population_density, food_sources, _ = config_scene
    chemo_deposit, _, _, _ = config_chemo

    # generate the positions at which we got agents using the specified density
    key, subkey = jr.split(key, 2)
    mask_grid = jr.uniform(subkey, (height, width)) < initial_population_density

    # generate agent data at the positions in mask_grid
    key, *subkeys = jr.split(key, height * width + 1)
    subkeys = jnp.array(subkeys).reshape((height, width, len(key)))
    agent_grid = jax.vmap(jax.vmap(
        lambda mask, key: jax.lax.cond(
            mask, lambda k: agent_init(k), lambda _k: jnp.zeros((3,)).astype(int), key
        )
    ))(mask_grid, subkeys)

    trail_grid = jnp.zeros((height, width))  # contains trail data

    # contains chemo-nutrient/food data. Food sources are initialized as 3x3
    # arrays, the extra dimensions here allow food source placement at the very
    # edges (though the 3x3 size will get clipped). Also need a binary mask of 
    # the initial chemo grid, to keep food source values constant across iterations.
    chemo_grid = jnp.zeros((height+2, width+2)) 
    for food_source in food_sources:
        chemo_grid = \
            chemo_grid.at[
                food_source[0]:food_source[0]+3, 
                food_source[1]:food_source[1]+3].set(chemo_deposit)

    chemo_grid = chemo_grid[1:-1, 1:-1]
    food_source_grid = chemo_grid > 0

    return agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid


@partial(jax.jit, static_argnames=['height', 'width'])
def _grid_coordinates(height, width):
    """Get all coordinates in a 2d grid."""
    X, Y = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    return jnp.vstack((Y.flatten(), X.flatten())).T



@jax.jit
def _step(carry_in, coordinate):
    ''' Performs an update to a single grid coordinate position'''

    def out_of_bounds(ls_coord, rs_coord, height, width):
        ''' Returns whether the left or right sensor(s) are out
        of bounds of the field '''

        # Goes through all possible out-of-bounds cases individually,
        # jax doesn't allow them to all be evaluated at once.
        l_oob = jnp.select(condlist=[ls_coord[0] < 0, 
                                     ls_coord[0] >= height,
                                     ls_coord[1] < 0,
                                     ls_coord[1] >= width],
                    choicelist=[True,True,True,True],
                    default=False)

        r_oob = jnp.select(condlist=[rs_coord[0] < 0, 
                                     rs_coord[0] >= height,
                                     rs_coord[1] < 0,
                                     rs_coord[1] >= width],
                    choicelist=[True,True,True,True],
                    default=False)

        return l_oob, r_oob

    def continue_rotation(carry_in):
        ''' Returns whether any of the sensors are out of bounds, or
        if agent has reached the elimination trigger via rotation
        Used as the condition in the while loop within rotate_agent'''
        ls_coord, rs_coord, agent, sensor_length, \
        elimination_trigger, height, width = carry_in
        l_oob, r_oob = out_of_bounds(ls_coord, rs_coord, height, width)
        oob = l_oob | r_oob
        not_stuck_agent = agent[2] > elimination_trigger

        return oob & not_stuck_agent

    def rotate_agent(carry_in):
        ''' Rotates agent once, according to rules in Wu et al. Part of 
        loop in agent_present() function. Used when agent cannot move 
        forward. '''
        ls_coord, rs_coord, agent, sensor_length, \
        elimination_trigger, height, width = carry_in

        l_oob, r_oob = out_of_bounds(ls_coord, rs_coord, height, width)
        # use l_oob, r_oob, and current agent direction as indices to
        # determine how to rotate the agent. 
        # indexing order: (l_oob, r_oob, dx+1, dy+1), all values for
        # l_oob = 0 and r_oob = 0 should never be accessed. 
        rotation_lut = jnp.array(

                         [[[[(-5,-5),(-5,-5),(-5,-5)],
                            [(-5,-5),(-5,-5),(-5,-5)],
                            [(-5,-5),(-5,-5),(-5,-5)]],

                           [[( 0,-1),(-1,-1),(-1, 0)],
                            [( 1,-1),(-5,-5),(-1, 1)],
                            [( 1, 0),( 1, 1),( 0, 1)]]],

                          [[[(-1, 0),(-1, 1),( 0, 1)],
                            [(-1,-1),(-5,-5),( 1, 1)],
                            [( 0,-1),( 1,-1),( 1, 0)]],

                           [[( 1, 1),( 1, 0),( 1,-1)],
                            [( 0, 1),(-5,-5),( 0,-1)],
                            [(-1, 1),(-1, 0),(-1,-1)]]]])

        # convert l_oob and r_oob (booleans) to numerical indices
        l_oob_idx = jax.lax.cond(l_oob, lambda:1 , lambda:0)
        r_oob_idx = jax.lax.cond(r_oob, lambda:1 , lambda:0)

        new_direction = rotation_lut[l_oob_idx, r_oob_idx, agent[0]+1, agent[1]+1]
        # update direction and decrement movement counter
        agent = agent.at[:2].set(new_direction)
        agent = agent.at[-1].set(agent[-1]-1)

        new_left_sensor, new_right_sensor = agent_sensor_positions(agent, sensor_length)
        new_ls_coord = coordinate + new_left_sensor
        new_rs_coord = coordinate + new_right_sensor    
        # jax.debug.print('Stuck, rotating: {x}', x=agent)

        return (new_ls_coord, new_rs_coord, agent, sensor_length, 
            elimination_trigger, height, width)    

    def remove_agent(agent, coordinate, 
        scene, config_agent, config_trail, config_chemo, to_update, key):
        ''' Clears the grid positions at a specific coordinate. Used to
        destroy agents in the field. '''
        # jax.debug.print("Removed agent at position")
        agent_grid, mask_grid, trail_grid, _, _ = scene
        agent_grid = agent_grid.at[*coordinate].set(jnp.zeros(3, dtype=jnp.int32))
        # jax.debug.print("New value at position: {x}", x=agent_grid[*coordinate])
        mask_grid = mask_grid.at[*coordinate].set(False)

        return agent_grid, mask_grid, trail_grid, to_update

    def rotate_sense(agent, coordinate, scene, 
        config_agent, config_trail, config_chemo, key):
        ''' Rotates agent towards the sensor with the highest calculated
        value as per Wu et al. '''
        _, _, trail_grid, chemo_grid, _ = scene
        sensor_length, _, _ = config_agent
        _, _, _, trail_weight = config_trail
        _, _, _, chemo_weight = config_chemo

        left_sensor, right_sensor = agent_sensor_positions(agent, sensor_length)
        ls_coord = coordinate + left_sensor
        rs_coord = coordinate + right_sensor

        # jax.debug.print("Comparing sensors at positions: ")
        # jax.debug.print('{x}', x=ls_coord)
        # jax.debug.print('{x}', x=rs_coord)

        # compute values of sensors
        l_sv = chemo_weight*chemo_grid[*ls_coord] + trail_weight*trail_grid[*ls_coord]
        r_sv = chemo_weight*chemo_grid[*rs_coord] + trail_weight*trail_grid[*rs_coord]

        # jax.debug.print("Sensor values : ")
        # jax.debug.print('Left: {x}', x=l_sv)
        # jax.debug.print('Right: {x}', x=r_sv)

        # update direction based on which is larger
        direction_idx = jnp.select(condlist=[l_sv > r_sv, 
                                     l_sv < r_sv],
                    choicelist=[0,1],
                    default=jr.choice(key, 2))

        new_directions = agent_sensor_directions(agent)
        agent = agent.at[:2].set(new_directions[direction_idx])

        # jax.debug.print('New agent properties: {x}', x=agent)

        return agent

    def move_agent(agent, coordinate, 
        scene, config_agent, config_trail, config_chemo, to_update, key):
        ''' Moves an agent forward along its velocity vector, by
        one grid position'''
        agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid = scene
        trail_deposit, _, _, _ = config_trail

        new_pos = coordinate + agent[:2]    
        agent = agent.at[-1].set(agent[-1]+1)

        agent_grid, mask_grid, trail_grid, to_update = remove_agent(
            agent, coordinate, scene, config_agent, config_trail, config_chemo,
            to_update, key)

        scene = agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid

        # rotate agent towards sensor with highest value
        key, subkey = jr.split(key, 2)
        agent = rotate_sense(agent, new_pos, scene, 
            config_agent, config_trail, config_chemo, subkey)


        agent_grid = agent_grid.at[*new_pos].set(agent)
        mask_grid = mask_grid.at[*new_pos].set(True)
        # flag this position, so that the agent cannot be moved more than once
        # (happens when agent is moved to a position not yet iterated over)
        to_update = to_update.at[*new_pos].set(False)

        # jax.debug.print("Moved agent to position: {x}", x=new_pos)
        # jax.debug.print("New agent parameters: {x}", x=agent)

        # deposit trail at new position
        # jax.debug.print("Trail value at new position: {x}", x=trail_grid[*new_pos])
        trail_grid = trail_grid.at[*new_pos].set(trail_grid[*new_pos] + trail_deposit)
        # jax.debug.print("Deposited trail at position: {x}", x=new_pos)
        # jax.debug.print("New trail value value at position: {x}", x=trail_grid[*new_pos])



        return agent_grid, mask_grid, trail_grid, to_update

    def random_orientation(agent, coordinate, 
        scene, config_agent, config_trail, config_chemo, to_update, key):
        ''' Randomly selects new velocity for the current agent '''
        # jax.debug.print("Could not move, rotating agent instead")
        agent_grid, mask_grid, trail_grid, _, _ = scene
         
        temp_agent = agent_init(key)
        agent = agent.at[:2].set(temp_agent[:2])
        agent = agent.at[-1].set(agent[-1]-1)
        # jax.debug.print("Rotated agent: {x}", x=agent)

        agent_grid = agent_grid.at[*coordinate].set(agent)

        return agent_grid, mask_grid, trail_grid, to_update


    def attempt_move(agent, coordinate, 
        scene, config_agent, config_trail, config_chemo, to_update, key):
        ''' Attempts to move an agent one square in the direction of 
        their velocity vector. If the square is occupied, randomly
        chooses a new velocity vector and assigns it to the agent.'''
        agent_grid, mask_grid, trail_grid, _, _ = scene

        new_pos = coordinate + agent[:2]
        agent_grid, mask_grid, trail_grid, to_update = jax.lax.cond(
            mask_grid[*new_pos]==False, 
            move_agent,
            random_orientation,
            agent, coordinate, 
            scene, config_agent, config_trail, config_chemo, to_update, key)

        return agent_grid, mask_grid, trail_grid, to_update


    def reproduce(agent_grid, mask_grid, to_update, old_coordinate, key):
        ''' Randomly initializes a new agent in the position of its parent
        agent, if the parent has exceeded the reproduction trigger threshold '''
        new_agent = agent_init(key)
        agent_grid = agent_grid.at[*old_coordinate].set(new_agent)
        mask_grid = mask_grid.at[*old_coordinate].set(True)
        # don't want to update the children in the current iteration
        to_update = to_update.at[*old_coordinate].set(False)
        # jax.debug.print("Reproducing, placed an agent {x} at position {y}", x=new_agent, y=old_coordinate)

        return agent_grid, mask_grid, to_update


    def dont_reproduce(agent_grid, mask_grid, to_update, old_coordinate, key):
        ''' Does nothing, needed for jax.lax.cond in reproduction step'''
        # jax.debug.print("Not reproducing")
        return agent_grid, mask_grid, to_update


    def agent_present(agent, coordinate, scene, 
        config_agent, config_trail, config_chemo, to_update, key):
        ''' Computes a position update for the given agent'''
        # compute coordinates of each sensor
        # jax.debug.print("Agent coordinate: {x}", x=coordinate)
        # jax.debug.print("Original agent: {x}", x=agent)

        agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid = scene
        sensor_length, reproduction_trigger, elimination_trigger = config_agent

        left_sensor, right_sensor = agent_sensor_positions(agent, sensor_length)
        ls_coord = coordinate + left_sensor
        rs_coord = coordinate + right_sensor
        height, width = mask_grid.shape

        # rotate the agent until its sensors are not out of bounds
        carry_in = (ls_coord, rs_coord, agent, sensor_length, elimination_trigger, 
            height, width)

        carry_out = jax.lax.while_loop(continue_rotation, rotate_agent, carry_in)
        _, _, agent, _, _, _, _ = carry_out

        # change agent to its rotated form, and repackage the scene
        agent_grid = agent_grid.at[*coordinate].set(agent)
        scene = agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid

        # if the elimination trigger is met, remove agent. Otherwise, attempt
        # to move forward.
        key, subkey = jr.split(key, 2)

        agent_grid, mask_grid, trail_grid, to_update = jax.lax.cond(
            agent[-1] < elimination_trigger,
            remove_agent, attempt_move, 
            agent, coordinate, scene, 
            config_agent, config_trail, config_chemo, to_update, subkey)

        # if the reproduction trigger is met, generate new agent in
        # the current agent's old position 
        agent_grid, mask_grid, to_update = jax.lax.cond(
            agent[-1] > reproduction_trigger,
            reproduce, dont_reproduce, 
            agent_grid, mask_grid, to_update, coordinate, key)

        # jax.debug.print("Agent post initial rotation: {x}", x=agent)

        return agent_grid, mask_grid, trail_grid, to_update

    def agent_absent(agent, coordinate, scene, 
        config_agent, config_trail, config_chemo, to_update, key):
        ''' Does nothing, needed for jax conditionals when agent is not present in
        the grid position or should be ignored. '''
        # jax.debug.print("No agent")
        return agent_grid, mask_grid, trail_grid, to_update

    scene, config_agent, config_trail, config_chemo, to_update, key = carry_in

    agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid = scene
    sensor_length, reproduction_trigger, elimination_trigger = config_agent
    trail_deposit, trail_damping, trail_filter_size, trail_weight = config_trail
    chemo_deposit, chemo_damping, chemo_filter_size, chemo_weight = config_chemo

    key, subkey = jr.split(key, 2)

    agent = agent_grid[*coordinate]  
    # if grid has an agent there and to_update is False in this position,
    # the agent was placed at that position during an update of a previous
    # coordinate and should be ignored.
    is_agent = mask_grid[*coordinate]
    is_agent = is_agent & (to_update[*coordinate])

    # update agent position and trail grid
    agent_grid, mask_grid, trail_grid, to_update = jax.lax.cond(
        is_agent, agent_present, agent_absent, 
        agent, coordinate, scene, config_agent, config_trail, config_chemo, 
        to_update, subkey)   

    # package arguments for next iteration
    scene = agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid
    config_agent = sensor_length, reproduction_trigger, elimination_trigger
    config_trail = trail_deposit, trail_damping, trail_filter_size, trail_weight
    config_chemo = chemo_deposit, chemo_damping, chemo_filter_size, chemo_weight

    carry_out = scene, config_agent, config_trail, config_chemo, to_update, key

    return carry_out, None


@partial(jax.jit, static_argnames=['config_trail', 'config_chemo', 'config_agent'])
def scene_step(scene, config_trail, config_chemo, config_agent, key):
    """Perform one update step on the scene by updating each agent in a random order."""

    agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid = scene
    _, trail_damping, trail_filter_size, _ = config_trail
    chemo_deposit, chemo_damping, chemo_filter_size, _ = config_chemo

    # generate a shuffled list of coordinates which determines the agent update order.
    # coordinates are only updated if an agent is on them. 
    coordinates = jr.permutation(key, _grid_coordinates(*mask_grid.shape))
    # boolean grid, used to account for possibility that agent is moved to a 
    # grid position that has not yet been iterated over, leading to an agent
    # moving multiple times. 
    to_update = jnp.full_like(mask_grid, True)

    results, _ = jax.lax.scan(_step, 
        (scene, config_agent, config_trail, config_chemo, to_update, key), coordinates)
    scene, _, _, _, _, _ = results

    # convolve both the chemo and trail with an average filter after 
    # all agents have been updated + rotated. 
    agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid = scene

    # jax.debug.print('{x}', x=chemo_grid)
    # chemo grid
    chemo_kernel = jnp.ones((chemo_filter_size, chemo_filter_size))*\
        (1/chemo_filter_size**2)
    chemo_grid = jax.scipy.signal.convolve2d(chemo_grid, chemo_kernel, mode='same')

    chemo_grid = chemo_grid*(1-chemo_damping)

    # reset the values in the food sources to the default
    not_food_source_grid = food_source_grid == 0
    chemo_grid = jnp.multiply(not_food_source_grid, chemo_grid) + \
        food_source_grid*chemo_deposit

    # trail grid
    trail_kernel = jnp.ones((trail_filter_size, trail_filter_size))*\
        (1/trail_filter_size**2)
    trail_grid = jax.scipy.signal.convolve2d(trail_grid, trail_kernel, mode='same')
    trail_grid = trail_grid*(1-trail_damping)

    scene = agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid


    return scene


# @partial(jax.jit, static_argnames=['upscaled_shape'])
def scene_pixelmap(scene, upscaled_shape, display_config):
    """Create a pixelmap of the scene on the gpu that can be drawn directly."""
    agent_grid, mask_grid, trail_grid, chemo_grid, food_source_grid = scene

    display_chemo, display_trail, display_agents, display_food = display_config

    # create a black and white colormap based on where there are agents
    agent_colormap = ((1 - mask_grid) * 255)
    # upscale the black and white colormap
    agent_colormap = jax.image.resize(agent_colormap, upscaled_shape, method='nearest')

    # create colormap for trails and food source, blue and red respectively.
    # upscale trail and chemo maps
    trail_colormap = jax.image.resize(trail_grid, upscaled_shape, method='nearest')
    chemo_colormap = jax.image.resize(chemo_grid, upscaled_shape, method='nearest')
    food_source_colormap = \
        jax.image.resize(
            food_source_grid, upscaled_shape, method='nearest')

    # To achieve the desired color,the target color channel is set to 255, 
    # and the other two are *decreased* in proportion to the value in the 
    # trail/chemo map. This makes low values close to white, and high 
    # values a dark color.

    red_channel = jnp.full_like(agent_colormap, 255)
    green_channel = jnp.full_like(agent_colormap, 255) 
    blue_channel = jnp.full_like(agent_colormap, 255) 

    if display_chemo:  
        # intensity transformation, strictly for visual purposes
        # clipping the map back to [0, 255]
        intensity = 30
        chemo_colormap = jnp.minimum(intensity*chemo_colormap, 255)
        # inverting the map
        chemo_colormap = jnp.full_like(chemo_colormap, 255)-chemo_colormap
        
        red_channel = jnp.full_like(chemo_colormap, 255)
        green_channel = chemo_colormap
        blue_channel = jnp.copy(chemo_colormap)

    if display_trail:
        # intensity transformation, strictly for visual purposes
        # clipping the map back to [0, 255]
        intensity = 10
        trail_colormap = jnp.minimum(intensity*trail_colormap, 255)
        # inverting the map
        trail_colormap = jnp.full_like(trail_colormap, 255)-trail_colormap

        trail_pixels = trail_colormap < 255
        not_trail_pixels = trail_colormap == 255
        
        red_channel = jnp.multiply(red_channel, not_trail_pixels) + \
            jnp.multiply(trail_colormap, trail_pixels)

        green_channel = jnp.multiply(green_channel, not_trail_pixels) + \
            jnp.multiply(trail_colormap, trail_pixels)

        blue_channel = jnp.multiply(blue_channel, not_trail_pixels) + \
            jnp.multiply(jnp.full_like(blue_channel, 255), trail_pixels)

    if display_agents:
        agent_pixels = agent_colormap == 0
        not_agent_pixels = agent_colormap == 255

        red_channel = jnp.multiply(red_channel, not_agent_pixels) + \
            jnp.multiply(agent_colormap, agent_pixels)

        green_channel = jnp.multiply(green_channel, not_agent_pixels) + \
            jnp.multiply(agent_colormap, agent_pixels)

        blue_channel = jnp.multiply(blue_channel, not_agent_pixels) + \
            jnp.multiply(agent_colormap, agent_pixels)

    if display_food:
        # placing food sources on top of everything
        food_source_pixels = food_source_colormap > 0
        not_food_source_pixels = food_source_colormap == 0

        red_channel = jnp.multiply(red_channel, not_food_source_pixels) + \
            jnp.multiply(jnp.full_like(red_channel, 255), food_source_pixels)

        green_channel = jnp.multiply(green_channel, not_food_source_pixels) + \
            jnp.multiply(jnp.zeros_like(green_channel), food_source_pixels)

        blue_channel = jnp.multiply(blue_channel, not_food_source_pixels) + \
            jnp.multiply(jnp.zeros_like(blue_channel), food_source_pixels)    


    pixelmap = jnp.stack((red_channel.astype(jnp.uint8), green_channel.astype(jnp.uint8), blue_channel.astype(jnp.uint8)), axis=-1)

    # transpose from shape (height, width, 3) to (width, height, 3) for pygame
    transposed_pixelmap = jnp.transpose(pixelmap, (1, 0, 2))

    # move the data from the gpu to the cpu so pygame can draw it
    return jax.device_put(transposed_pixelmap, device=jax.devices('cpu')[0])





