from env.square_maze import Maze
from DQN_brain import DeepQNetwork

from utils import plot_cost, plot_rate


def run_maze():
    step = 0
    render_time = 0
    episode_step_holder = []
    success_holder = []
    base_path = './logs/double_dqn/'

    for i_episode in range(400):
        episode_step = 0
        s = env.reset().ravel()

        while True:
            env.render(render_time)
            action = RL.choose_action(s)
            s_, reward, done, info = env.step(action)
            s_ = s_.ravel()
            # print('action:{0} | reward:{1} | done: {2}'.format(action, reward, done))
            RL.store_transition(s, action, reward, s_)

            if step > 200:
                RL.learn()

            s = s_
            step += 1
            episode_step += 1

            if episode_step > 500:
                done = True

            if done:
                print('{0} -- {1} -- {2}'.format(i_episode, info, episode_step))
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
    save_path = RL.saver.save(RL.sess, base_path + 'model_dqn.ckpt')
    print("Model saved in path: {}".format(save_path))
    RL.sess.close()
    env.destroy()
    # plot_cost(episode_step_holder, base_path + 'episode_steps.png')
    plot_rate(success_holder, base_path, index=15)


def main():
    global env, RL
    env = Maze('./env/maps/map2.json', full_observation=True)
    RL = DeepQNetwork(
        n_actions=4,
        n_features=25,
        restore_path=None,
        # restore_path=base_path + 'model_dqn.ckpt',
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
