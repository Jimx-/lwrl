# LWRL

Lightweight deep reinforcement learning library written with PyTorch

| Breakout(DQN)                                                                                         | Cartpole(Vanilla Policy Gradient)                                                  |
|:-----------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|
| ![breakout-dqn](https://thumbs.gfycat.com/AnchoredScornfulAustraliansilkyterrier-size_restricted.gif) | ![cartpole-vpg](https://thumbs.gfycat.com/WelllitFluffyBadger-size_restricted.gif) |

LWRL aims to provide a configurable and modular reinforcement learning library. With LWRL, enviroments, agents and models are encapsulated in individual modules that can be combined together. It is possible to experiment with new models and network architectures quickly by replacing parts of the application with the new modules. Also, everything can be easily configurable by providing a config file in JSON format. The goal of LWRL is to provide a toolbox that intergrates deep reinforcement learning algorithms and can be used to create reinforment learning applications quickly.

LWRL is built on top of [PyTorch](https://pytorch.org/) deep learning framework. It uses  [Visdom](https://github.com/facebookresearch/visdom.git) to visualize the training process. Currently LWRL is only compatible with Python 3.



## Features

LWRL currently supports OpenAI Gym style environments. These algorithms are provided by the library:

- Deep Q-learning(DQN) - `dqn_agent`  [paper](https://www.cs.toronto.edu/%7Evmnih/docs/dqn.pdf)
- Dueling DQN - `duel_dqn_agent`  [paper](https://arxiv.org/pdf/1511.06581.pdf)
- Double DQN - `dqn_agent` with `double_q_learning=True`  [paper](https://arxiv.org/abs/1509.06461.pdf)
- Vanilla Policy Gradient(REINFORCE algorithm) - `vpg_agent`  [paper](http://www-anw.cs.umass.edu/%7Ebarto/courses/cs687/williams92simple.pdf)
- Actor-critic models - use `baseline` model in `vpg_agent`
- Deep deterministic policy gradient(DDPG) - `ddpg_agent`  [paper](https://arxiv.org/pdf/1509.02971.pdf)


## Getting started

### Installation

```
git clone git@github.com:Jimx-/lwrl.git
cd lwrl && pip install -r requirements.txt
```

This will install all dependencies except PyTorch. For instructions on installing PyTorch, see [pytorch.org](https://pytorch.org/).

### Running the examples

LWRL contains some examples scripts with configurations that you may run.  For example, to train a DQN agent on Breakout, run:

```sh
python3 -m lwrl.experiments.atari-dqn.main --env_id=BreakoutNoFrameskip-v4 --agent lwrl/configs/atari-dqn.json --network lwrl/configs/networks/nature-dqn.json --save_dir=/path/to/save/model --is_train
```

Before training, make sure that Visdom server is started by running:

```sh
python visdom.server &
```

During the training, you can navigate to `http://localhost:8097` for the training process visualization. After the agent is trained, you can test the agent by:

```sh
python3 -m lwrl.experiments.atari-dqn.main --env_id=BreakoutNoFrameskip-v4 --agent lwrl/configs/atari-dqn.json --network lwrl/configs/networks/nature-dqn.json --save_dir=/path/to/save/model --visualize
```

More example scripts and configurations can be found in `experiments` and `config` folders.



## License

This project is licensed under the MIT License.



## Acknowledgments

- This project is inspired by [TensorForce](https://github.com/reinforceio/tensorforce)




