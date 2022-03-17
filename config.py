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
    'BATCH_SIZE' : 1024,
    'GAMMA' : 0.99,
    'TAU' : 1e-3,
    'LR_ACTOR' : 1e-4,
    'LR_CRITIC' : 1e-3,
    'WEIGHT_DECAY' : 0,
    'UPDATE_EVERY' : 50,
    'UPDATES_PER_STEP' : 40,
    'fc1_units' : 256,
    'fc2_units' : 128,
    'desc' : 'fewer_units'
}