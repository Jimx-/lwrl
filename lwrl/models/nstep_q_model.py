import random
import torch

from lwrl.models import QModel
import lwrl.utils.th_helper as H


class NStepQModel(QModel):
    def calculate_td_error(self, q_values, next_q_values, rewards, neg_done_mask):
        final_q = next_q_values[-1]
        batch_size = rewards.shape[0]
        for i in range(batch_size - 1, -1, -1):
            rewards[i] = rewards[i] + (rewards[i + 1] if i != batch_size - 1 else final_q) * neg_done_mask[i] * self.discount_factor

        return rewards.unsqueeze(1) - q_values
