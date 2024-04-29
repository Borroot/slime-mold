from agent import *
from scene import *
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import matplotlib.pyplot as plt
import collections
import pygame


def initialize_config():
    # define all the hyperparameters for the simulation

    # original parameters
    # height, width = 200, 200
    # upscale = 4
    # initial_population_density = 0.5

    # trail_deposit = 5
    # trail_damping = 0.1
    # trail_filter_size = 3
    # trail_weight = 0.4

    # chemo_deposit = 10
    # chemo_damping = 0.2
    # chemo_filter_size = 5
    # chemo_weight = 1 - trail_weight

    # sensor_length = 7
    # reproduction_trigger = 15
    # elimination_trigger = -10

    # food_sources = jr.choice(jr.PRNGKey(21), jnp.arange(10,190), shape=(10,2))

    # downscaled
    height, width = 100, 100
    upscale = 4
    initial_population_density = 0.5

    trail_deposit = 5
    trail_damping = 0.2 # DOUBLED, less spreading
    trail_filter_size = 3
    trail_weight = 0.4

    chemo_deposit = 10
    chemo_damping = 0.4 # DOUBLED, less spreading
    chemo_filter_size = 3 # DECREASED, keep odd
    chemo_weight = 1 - trail_weight

    sensor_length = 4 # DECREASED
    reproduction_trigger = 15 
    elimination_trigger = -10

    food_sources = jr.choice(jr.PRNGKey(21), jnp.arange(10,90), shape=(10,2))

    # visualization settings
    display_chemo=True
    display_trail=False
    display_agents=True
    display_food=True

    # pack the config info into smaller variables for convenience
    config_trail = (trail_deposit, trail_damping, trail_filter_size, trail_weight)
    config_chemo = (chemo_deposit, chemo_damping, chemo_filter_size, chemo_weight)
    config_agent = (sensor_length, reproduction_trigger, elimination_trigger)
    display_config = (display_chemo, display_trail, display_agents, display_food)
    config_scene = (height, width, upscale, initial_population_density, 
        food_sources, display_config)

    return config_scene, config_agent, config_trail, config_chemo


def check_interrupt():
    """Check if the user tries to close the program."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True
    return False


def draw(scene, config_scene, screen, font, i):
    """Draw the scene to the screen."""
    height, width, upscale, _, _, display_config = config_scene

    # draw the scene

    upscaled_shape = upscale * jnp.array([height, width])

    surface = pygame.pixelcopy.make_surface(scene_pixelmap(
        scene, upscaled_shape, display_config))
    screen.blit(surface, (0, 0))

    # draw iteration counter
    text = font.render(f'{i}', True, (88, 207, 57))
    screen.blit(text, (5, 5))

    pygame.display.update()


def run_with_gui(key, num_iter=20000):
    """Run a simulation with gui, this is useful for debugging and getting images."""
    config_scene, config_agent, config_trail, config_chemo = initialize_config()
    height, width, upscale, _, _, _ = config_scene

    pygame.init()

    font = pygame.font.Font(None, 48)
    screen = pygame.display.set_mode(upscale * jnp.array([width, height]))

    key, subkey = jr.split(key, 2)
    scene = scene_init(config_scene, config_chemo, subkey)

    for i in range(num_iter):
        key, subkey = jr.split(key, 2)
        scene = scene_step(scene, config_trail, config_chemo, config_agent, subkey)

        draw(scene, config_scene, screen, font, i)

        if check_interrupt():
            break

    pygame.quit()


def run_headless(key):
    """Run simulations headless on the gpu, this is super efficient."""
    pass


if __name__ == '__main__':
    key = jr.PRNGKey(37)
    run_with_gui(key)

