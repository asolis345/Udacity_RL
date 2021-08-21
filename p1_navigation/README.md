# Project 1 - Navigation with Curious George

## Project Details

The objective of this project was to develop a DQN agent (monkey) that would be able to solve unity's [Banana Collector Env.](https://github.com/handria-ntoanina/unity-ml-banana).<br>

The monkey has 4 available actions:
* move forward
* move backward
* turn left
* turn right

The state space has 37 dimensions, this state space contains the monkey's velocity, along with ray-based perception of objects around monkey's forward direction.

The basic objective of the monkey is to navigate the the yellow bananas while avoiding the blue/purple bananas. The monkey gets a reward of +1 for each yellow banana and -1 for year blue/purple banana. The monkey is also limited to only 300 moves per episode.

The environment is considered as solved if the monkey is winning an average of +13 points for 100 consecutive episodes.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the monkey on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the approriate repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

## Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own monkey!  
