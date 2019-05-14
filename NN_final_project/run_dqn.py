import numpy as np
import matplotlib.pyplot as plt
import os

from env.maze import Maze
from DQN_brain import DeepQNetwork


def exist_or_create_folder(path_name):
    flag = False
    pure_path = os.path.dirname(path_name)
    if not os.path.exists(pure_path):
        try:
            os.makedirs(pure_path)
            flag = True
        except OSError:
            pass
    return flag


def write_to_file(file_path, content, overwrite=False):
    exist_or_create_folder(file_path)
    if overwrite is True:
        with open(file_path, 'w') as f:
            f.write(str(content))
    else:
        with open(file_path, 'a') as f:
            f.write(str(content))


def plot_cost(data, path):
    data_average = []
    size = len(data)
    for i in range(50, size):
        data_average.append(sum(data[(i-50):i])/50)

    np.save('./logs/data_average_rate.out', np.array(data_average))
    np.save('./logs/data_rate.out', np.array(data))

    plt.plot(np.arange(len(data_average)), data_average)
    plt.ylabel('success rate')
    plt.xlabel('episode')
    # plt.show()
    plt.savefig(path)
    plt.close()


def plot_rate(data, path):
    data_average = []
    size = len(data)
    for i in range(50, size):
        data_average.append(sum(data[(i-50):i])/50)

    np.save('./logs/data_average.out', np.array(data_average))
    np.save('./logs/data.out', np.array(data))

    plt.plot(np.arange(len(data_average)), data_average)
    plt.ylabel('episode steps')
    plt.xlabel('episode')
    # plt.show()
    plt.savefig(path)
    plt.close()


def run_maze():
    step = 0
    render_time = 0
    episode_step_holder = []
    success_holder = []

    for episode in range(400):
        episode_step = 0
        observation = env.reset().ravel()

        while True:
            env.render(render_time)
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = observation_.ravel()
            # print('action:{0} | reward:{1} | done: {2}'.format(action, reward, done))
            RL.store_transition(observation, action, reward, observation_)

            if step > 200:
                RL.learn()

            observation = observation_
            step += 1
            episode_step += 1

            if episode_step > 500:
                done = True

            if done:
                print('{0} -- {1} -- {2}'.format(episode, info, episode_step))
                if info != 'success':
                    episode_step = 500
                    reward = -1
                    success_holder.append(0)
                else:
                    success_holder.append(1)
                env.render(render_time)
                episode_step_holder.append(episode_step)
                break

    # end of game
    print('game over')
    save_path = RL.saver.save(RL.sess, './logs/model_dqn.ckpt')
    print("Model saved in path: {}".format(save_path))
    RL.sess.close()
    env.destroy()
    plot_cost(episode_step_holder, './logs/episode_steps.png')
    plot_rate(success_holder, './logs/success_rate.png')


def main():
    global env, RL
    env = Maze('./env/maps/map2.json', full_observation=True)
    RL = DeepQNetwork(
        n_actions=4,
        n_features=25,
        restore_path=None,
        # restore_path='./logs/model_dqn.ckpt',
        learning_rate=0.005,
        reward_decay=0.9,
        e_greedy=0.95,
        replace_target_iter=800,
        batch_size=64,
        # e_greedy_increment=None,
        e_greedy_increment=1e-3,
        output_graph=False,
    )
    env.after(100, run_maze)
    env.mainloop()


if __name__ == "__main__":
    main()
