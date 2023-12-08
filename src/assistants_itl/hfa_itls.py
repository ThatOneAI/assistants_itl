from typing import Union
from pydantic import BaseModel
from itllib import ResourceController

from .globals import *
from .hfa_module import HFAssistantConfig, HFAssistant, TaskLog, Stream


tasklogs.register(itl, CLUSTER, "assistants.thatone.ai", "v1", "TaskLog")(TaskLog)


@assistants.register(itl, CLUSTER, "assistants.thatone.ai", "v1", "HFAssistant")
class HFAController(ResourceController):
    async def create_resource(self, config):
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")
        result = HFAssistant(HFAssistantConfig(**config["spec"]))
        await result.connect()
        return result

    async def update_resource(self, resource: HFAssistant, config):
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")
        resource.configure(HFAssistantConfig(**config["spec"]))

    async def delete_resource(self, resource):
        print("You'll need to restart the script to complete deletion of an assistant")
        return super().delete_resource(resource)


@prompts.register(itl, CLUSTER, "assistants.thatone.ai", "v1", "Prompt")
class Prompt(BaseModel):
    prompt: str


@configs.register(itl, CLUSTER, "assistants.thatone.ai", "v1", "Config")
class Config(ResourceController):
    async def create_resource(self, config):
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")
        return config["spec"]


streams.register(itl, CLUSTER, "assistants.thatone.ai", "v1", "Stream")(Stream)
