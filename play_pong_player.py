from gym.utils import play
import gym

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"

if __name__ == "__main__":

    play.play(gym.make(DEFAULT_ENV_NAME), zoom=4, fps=30)

    

