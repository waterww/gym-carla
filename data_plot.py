import numpy as np
from matplotlib import pyplot as plt

def main(episode):

    data = np.load("/home/miao/gym-carla/action_list_test_%d.npz" % episode)
    #laterals = data['laterals']
    #yaws = data['yaws']
    actions = data['actions']
    #rewards = data['rewards']

    steps = np.arange(actions.size)
    
    '''
    plt.title("ep%d" % episode)
    plt.plot(steps,laterals)
    plt.xlabel('step')
    plt.ylabel('laterals')
    plt.show()

    plt.title("ep%d" % episode)
    plt.plot(steps,yaws)
    plt.xlabel('step')
    plt.ylabel('yaws')
    plt.show()
    '''

    plt.title("ep%d" % episode)
    plt.plot(steps,actions)
    plt.xlabel('step')
    plt.ylabel('actions')
    plt.show()

    print(actions)

    '''
    plt.title("ep%d" % episode)
    plt.plot(steps,rewards)
    plt.xlabel('step')
    plt.ylabel('rewards')
    plt.show()
    '''

if __name__ == '__main__':
    main(-3)
    