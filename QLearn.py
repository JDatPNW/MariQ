import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 200000

SHOW_EVERY = 500

DISCRETE_OS_SIZE = [14,14]
q_table = np.random.uniform(low=-1, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

epsilon = 0.5
START_EPSIOLON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSIOLON_DECAYING)


def get_discrete_state(state):
    ang = 0
    vel = 0

    velbins = np.array([ 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1])
    angbins = np.array([ 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21])


    if(state[2] > 0):
        ang = np.digitize(state[2], angbins, right=True)
    else:
        ang = -np.digitize(abs(state[2]), angbins, right=True)

    if(state[3] > 0):
        vel = np.digitize(state[3], velbins, right=True)
    else:
        vel = -np.digitize(abs(state[3]), velbins, right=True)

    discrete_state = np.array([ang, vel])

    #print(state, discrete_state)

    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False


    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q

        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSIOLON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()
