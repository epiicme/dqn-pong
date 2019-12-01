import sys
sys.path.insert(0, "C:/Users/John/Google Drive/Danmarks Tekniske Universitet/Year II/02456 - Deep Learning/project/dqn-pong/dqn-pong/lib")
#sys.path.insert(0, "D:/Work/GoogleDrive/Danmarks Tekniske Universitet/Year II/02456 - Deep Learning/project/dqn-pong/dqn-pong/lib")

from lib import wrappers
from lib import dqn_model

import gym, time, argparse, torch
import numpy as np

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

#DEFAULT_MODEL = "PongNoFrameskip-v4-best-16.69.dat"
DEFAULT_MODEL = "PongNoFrameskip-v4-best-16.18.dat"
DEFAULT_RECORD_PATH = "./records/"
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # #parser.add_argument("-m", "—model", required=True, help="Model file to load")
    # parser.add_argument("-m", "-—model", default=DEFAULT_MODEL, help="Model file to load")
    # parser.add_argument("-e", "-—env", default=DEFAULT_ENV_NAME,
    #                     help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    # parser.add_argument("-r", "-—record", default = "./records/", help="Directory to store video recording")
    # args = parser.parse_args()

    #env = wrappers.make_env(args.env)
    env = wrappers.make_env(DEFAULT_ENV_NAME)

    env = gym.wrappers.Monitor(env, DEFAULT_RECORD_PATH)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(DEFAULT_MODEL))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)

        c[action] += 1

        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)

    print("Total reward: %.2f % total_reward")
    print("Action counts:", c)
    
    #if args.record:
    env.env.close()
    #env.clos()


    

