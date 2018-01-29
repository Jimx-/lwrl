import os
import json
import torch


class Saver:
    def __init__(self, save_dir, max_to_keep=5):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.last_checkpoints = []
        self.max_to_keep = max_to_keep

        self._restore_meta()

    def _get_meta_path(self):
        return os.path.join(self.save_dir, 'checkpoint.json')

    def _save_meta(self):
        meta_file = self._get_meta_path()

        with open(meta_file, 'w') as fout:
            fout.write(json.dumps({
                'last_checkpoints': self.last_checkpoints
            }))

    def _restore_meta(self):
        meta_file = self._get_meta_path()

        if not os.path.exists(meta_file):
            self._save_meta()

        with open(meta_file, 'r') as fin:
            meta_dict = json.load(fin)
            self.last_checkpoints = meta_dict['last_checkpoints']

    def save(self, save_dict, global_steps):
        ckpt_name = 'model-{}.pth'.format(global_steps)
        self.last_checkpoints.append(ckpt_name)

        if len(self.last_checkpoints) > self.max_to_keep:
            del_ckpt = self.last_checkpoints.pop(0)
            os.remove(os.path.join(self.save_dir, del_ckpt))

        ckpt_path = os.path.join(self.save_dir, ckpt_name)
        torch.save(save_dict, ckpt_path)
        self._save_meta()

    def restore(self):
        self._restore_meta()

        if not self.last_checkpoints:
            return None

        ckpt_name = self.last_checkpoints[-1]
        ckpt_path = os.path.join(self.save_dir, ckpt_name)
        return torch.load(ckpt_path)
