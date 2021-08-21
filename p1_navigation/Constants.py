EPISODES = 1801             # maximum number of episodes to run
ANN = [128, 64, 64, 128]    # Ann architecture (does not include in and out layers, 
                            # pulled auto from state/action space
BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 512            # minibatch size
GAMMA = 0.95                # discount factor
TAU = 1e-2                  # for soft update of target parameters
LR = 1e-2                   # learning rate 
UPDATE_EVERY = 2            # how often to update the network
EXPLORATION = 0.9999        # epsilon
EXPLORATION_DEC = 0.002999666 * 0.75    # epsilon decay rate
EXPLORATION_MIN = 0.01               # min epsilon to decay to 
DROPUT = 0.0            # droput to be used in nn
USE_L_RELU = False      # use LeakyRelu as activation function in network