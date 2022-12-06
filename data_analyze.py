import numpy as np
from matplotlib import pyplot as plt

def main(episode):

    data = np.load("/home/miao/hpc_output_files/laterals_yaws_actions_rewards_ep%d.npz" % episode)
    laterals = data['laterals']
    yaws = data['yaws']
    actions = data['actions']
    rewards = data['rewards']

    steps = np.arange(laterals.size)

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

    plt.title("ep%d" % episode)
    plt.plot(steps,actions,'.')
    plt.xlabel('step')
    plt.ylabel('actions')
    plt.show()

    plt.title("ep%d" % episode)
    plt.plot(steps,rewards)
    plt.xlabel('step')
    plt.ylabel('rewards')
    plt.show()

if __name__ == '__main__':
    main(1658)
    