import json


def get_config():
    with open("config.json", 'r') as file:
        config = json.load(file)
    return config
    