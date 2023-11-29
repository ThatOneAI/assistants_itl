import os
from itllib import Itl
from .resources import ResourceSet

CLUSTER = "assistants"
CONFIG_PATH = os.path.join("config.yaml")
SECRETS_PATH = "./secrets"

itl = Itl()
itl.apply_config(CONFIG_PATH, SECRETS_PATH)
itl.start()

tools = ResourceSet()
prompts = ResourceSet()
tasklogs = ResourceSet()
assistants = ResourceSet()
