import tensorflow as tf

from env.square_maze import Maze
from utils import plot_cost, plot_rate
from A2C_brain import Actor, Critic


OUTPUT_GRAPH = False
RESTORE = False
MAX_EPISODE = 100
LR_A = 0.000001  # learning rate for actor
LR_C = 0.000002  # learning rate for critic
N_F = 25
N_A = 4


def run_maze():
    step = 0
    render_time = 0
    episode_step_holder = []
    success_holder = []
    base_path = './logs/a2c/'

    sess = tf.Session()

    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F, lr=LR_C)

    sess.run(tf.global_variables_initializer())
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    if RESTORE:
        saver.restore(sess, base_path + 'model_a2c.ckpt')
        print("Model restore in path: {}".format(base_path + 'model_a2c.ckpt'))

    if OUTPUT_GRAPH:
        tf.summary.FileWriter('.logs/', sess.graph)

    for i_episode in range(MAX_EPISODE):
        episode_step = 0
        s = env.reset().ravel()

        while True:
            env.render(render_time)

            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)
            s_ = s_.ravel()

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[log(Pi(s,a)) * td_error]

            s = s_
            step += 1
            episode_step += 1

            if episode_step > 100:
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
    if RESTORE:
        save_path = saver.save(sess, base_path + 'model_a2c.ckpt')
        print("Model saved in path: {}".format(save_path))
    sess.close()
    env.destroy()
    # plot_cost(episode_step_holder, base_path + 'episode_steps.png')
    plot_rate(success_holder, base_path + 'success_rate.png')


def main():
    global env
    env = Maze('./env/maps/map2.json', full_observation=True)
    env.after(100, run_maze)
    env.mainloop()


if __name__ == '__main__':
    main()
