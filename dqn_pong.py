import sys
#sys.path.insert(0, "C:/Users/John/Google Drive/Danmarks Tekniske Universitet/Year II/02456 - Deep Learning/project/dqn-pong/dqn-pong/lib")
sys.path.insert(0, "D:/Work/GoogleDrive/Danmarks Tekniske Universitet/Year II/02456 - Deep Learning/project/dqn-pong/dqn-pong/lib")

from lib import wrappers
from lib import dqn_model

import argparse, time, collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

############### HYPERPARAMETERS ###############
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5 # reward boundary for the last 100 episodes to stop training
GAMMA = 0.9 # Bellman equation
BATCH_SIZE = 32 # sampled from the replay buffer
REPLAY_SIZE = 10000 # maximum capacity of the buffer
REPLAY_START_SIZE = 10000 # every this frames before training to populate the replay buffer
LEARNING_RATE = 1e-4 # for ADAM
SYNC_TARGET_FRAMES = 1000 # how often sync model weights from training model to target model
                          # used to get the value of the next state in Bellman approx
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0  # epsilon-decay start
EPSILON_FINAL = 0.02 # epsilon-decay finish
##############################################

# tuples; holds (state, action,reward,done, next_state)
Experience = collections.namedtuple("Experience", field_names=["state", "action", "reward", "done", "new_state"])

class ExperienceBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        '''
        Create a list of random idices and repack the sampled entries into np arrays
        '''
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
        
class Agent():
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
    
    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0
    
    def play_step(self, net, epsilon=0.0, device = "cpu"):
        '''
        Main agent method. Performs step in the environment, stores result to buffer
        '''
        done_reward = None
    
        # case: exploration
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else: 
            # case: exploitation; the NN is used to get the Q function values
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            # net.forward
            q_vals_v = net.forward(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        
        # the agent takes a step based on the NN best action
        # returns next observation(state) reward store them in exp buffer
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state # what?

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    # wrap numpy arras to torch tensors
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device) # check this
    rewards_v = torch.tensor(rewards).to(device)
    #done_mask = torch.ByteTensor(dones).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    # pass the observations to first model; extract Q-values using gather()
    # gather(): first_arg = dim index to perform gathering; 1 - actions
    # second_arg = tensor of indices of elements to be chosen
    # unsqueeze() and squeeze() - required to get rid of extra dimensions we created
    # useful ilustration in the book
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # apply target net to next state obs and calculate max Q-value along the action dim=1
    # max returns both max val and indices (calculates max and argmax)
    next_state_values = tgt_net(next_states_v).max(1)[0]

    # IMPORTANT for convergence!
    # if transition from last step in episode, action value = no discounted reward of next state
    # there is no next state to gather reward from
    next_state_values[done_mask] = 0.0

    # detach value from computation graph -> prevent gradients from flowing into the NN
    # used to calculate Q approx for next states
    next_state_values = next_state_values.detach()

    # calculate Bellman approximation value
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    # calculate mean squared error loss
    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            temp_str = str(frame_idx) + ": done " +
                       str(len(total_rewards)) + " games, mean reward " +
                       str(mean_reward) + ", eps " +
                       str(epsilon) + ", speed " +
                       str(speed) + " f/s"

            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    with open("output.txt", "a") as f:
                    f.write("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_rew$
                    f.write(temp_str)
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
