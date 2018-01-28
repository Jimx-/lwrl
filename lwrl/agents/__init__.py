from lwrl.agents.agent import Agent
from lwrl.agents.learning_agent import LearningAgent
from lwrl.agents.ql_agent import QLearningAgent, DuelingQLearningAgent

agent_dict = dict(
    dqn_agent=QLearningAgent,
    duel_dqn_agent=DuelingQLearningAgent,
)

def agent_factory(type, *args, **kwargs):
    return agent_dict[type](*args, **kwargs)
