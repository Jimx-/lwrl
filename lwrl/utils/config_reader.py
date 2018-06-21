import json


def read_config(config_file):
    with open(config_file, 'r') as fin:
        return json.load(fin)
