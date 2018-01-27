
class Memory:
    def __init__(self):
        pass

    def add(self, obs, action, reward, done):
        """
        Add a transition to the memory
        :param transition: transition to add
        :return: None
        """
        raise NotImplementedError()

    def sample(self, size):
        """
        Sample a batch of transitions from the memory
        :param size: size of the batch
        :return: the batch of transitions
        """
        raise NotImplementedError()

    def size(self):
        """
        :return: the size of memory
        """
        raise NotImplementedError()
