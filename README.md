[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

## Introduction

This project was completed as part of the Udacity Deep Reinforcement Learning Nanodegree program. The goal of this project is to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment (described below) which was created with Unity and provided by Udacity. 

## Description of Environment
Below is a sample animation (provided by Udacity) of 10 agents that have learned how to play this environment. Each agent controls a double jointed arm and the goal is to keep the end point of the arm in the rotating green region (i.e. the target location). A reward of +0.1 is provided to the agent for every timestep that the arm's endpoint falls within the green highlighted region. 

![Trained Agent (image provided by Udacity)][image1]


Each agent observes a 33-dimensional state vector that contains information about position, rotation, velocity, and angular velocities of the arm. At everytimestep each agent takes a cointinous valued 4-dimensional action that describes the torque to apply to each of the two joints. Each action value should be between -1 and 1. For this project we use an environment with 20 agents. This environment is considered solved when the average score over 100 episodes exceeds +30 over all 20 agents. 


## How to run the code
First download the Unity environment using the instructions provided by Udacity:
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
2. Place the downloaded file in this repository and unzip or decompress it. 
3. Set the file_name='' argument in the first line of cell [2] in D4PG.ipynb to point towards the Reacher.exe file you downloaded in the above two steps. Depending on how you extracted the downloaded files it may already point in the correct location. 
4. Run cells 1,2,3, and 4 in the D4PG notebook to start training an agent from scratch. The final cell will visualize the performance of your trained agent(s).
5. You can also load a pretrained agent by running cells 1, 2, and 5
6. You can visualize the performance of your trained agent(s) by running cell 6.