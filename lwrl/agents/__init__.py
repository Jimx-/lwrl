from lwrl.agents.agent import Agent
from lwrl.agents.learning_agent import LearningAgent
from lwrl.agents.memory_agent import MemoryAgent
from lwrl.agents.batch_agent import BatchAgent
from lwrl.agents.ql_agent import QLearningAgent, DuelingQLearningAgent
from lwrl.agents.nstep_ql_agent import NStepQLearningAgent
from lwrl.agents.vpg_agent import VPGAgent
from lwrl.agents.ddpg_agent import DDPGAgent

agent_dict = dict(
    dqn_agent=QLearningAgent,
    duel_dqn_agent=DuelingQLearningAgent,
    nstep_dqn_agent=NStepQLearningAgent,
    vpg_agent=VPGAgent,
    ddpg_agent=DDPGAgent,
)


def agent_factory(type, *args, **kwargs):
    return agent_dict[type](*args, **kwargs)
