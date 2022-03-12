'''
Use this file to configure your DDPG Agent. Upon initialization 
of the DDPG agent these parameters will be loaded and saved to a log 
directory. This file may be edited upon each run to intialize the 
Agent differently.

for each new run the 'desc' field should be updated with a brief string to 
be used as the directory name under which this new run will be saved.
'''

CONFIG = {
    'BUFFER_SIZE' : int(1e6),
    'BATCH_SIZE' : 512,
    'GAMMA' : 0.99,
    'TAU' : 1e-3,
    'LR_ACTOR' : 1e-5,
    'LR_CRITIC' : 1e-4,
    'WEIGHT_DECAY' : 0,
    'UPDATE_EVERY' : 40,
    'UPDATES_PER_STEP' : 10,
    'fc1_units' : 300,
    'fc2_units' : 400,
    'desc' : 'less_updates'
}