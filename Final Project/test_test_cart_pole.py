from test_cart_pole import run_cart_pole
import matplotlib.pyplot as plt

if __name__ == '__main__':
    NUM_TRIALS = 50

    num_episodes_list = []
    best_avg_reward_list = []

    for i in range(NUM_TRIALS):
        episodes, best_avg_reward = run_cart_pole()
        num_episodes_list.append(episodes)
        best_avg_reward_list.append(best_avg_reward)

    plt.plot(num_episodes_list)
    plt.grid()
    plt.title("# Episodes to Solve")
    plt.xlabel("Iteration")
    plt.ylabel("# Episodes")
    plt.savefig("num_episodes_condition_23.png", dpi=300)
    plt.close()

    plt.plot(best_avg_reward_list)
    plt.grid()
    plt.title("Best Average Rewards")
    plt.xlabel("Iteration")
    plt.ylabel("Best Average Reward")
    plt.savefig("best_avg_reward_condition_23.png", dpi=300)
    plt.close()
