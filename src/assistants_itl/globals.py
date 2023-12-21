import os
from itllib import Itl, SyncedResources

CLUSTER = "assistants"
CONFIG_PATH = os.path.join("config.yaml")
SECRETS_PATH = "./secrets"

itl = Itl()
itl.apply_config(CONFIG_PATH, SECRETS_PATH)
itl.start()

tools = SyncedResources()
prompts = SyncedResources()
tasklogs = SyncedResources()
assistants = SyncedResources()
configs = SyncedResources()
loops = SyncedResources()
streams = SyncedResources()
