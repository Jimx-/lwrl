from lwrl.memories.memory import Memory
from lwrl.memories.sequential import SequentialMemory

def get_replay_memory(config):
    type_dict = {
        "sequential": SequentialMemory,
    }

    return type_dict[config['type']](**config['args'])
