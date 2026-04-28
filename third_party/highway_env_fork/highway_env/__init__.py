import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from . import envs  # triggers gym.register() for hetero envs
name = 'highway_env'
