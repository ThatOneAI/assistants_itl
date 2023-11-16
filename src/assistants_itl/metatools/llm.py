from string import Template
import yaml
import re
import os
import openai
from transformers.tools import Tool
from itllib.clusters import ResourceController
from itllib import Itl

from . import Metatool, _check_config


class LanguageMetatool(Metatool):
    def __init__(self, itl: Itl, cluster):
        super().__init__(itl, cluster, "LanguageTool")

    def create_tool(self, config: dict):
        if "spec" not in config or "interface" not in config["spec"]:
            raise ValueError("Config is missing required key: spec.interface")

        api = config["spec"]["interface"]
        if api == "openai-chat":
            return ChatGPTTool.from_config(config)


class ChatGPTTool(Tool):
    def __init__(
        self,
        description,
        model,
        calls,
        interface="openai-chat",
    ):
        super().__init__()
        self.description = description
        self.model = model
        self.calls = calls
        self.interface = interface
        self.api_key = os.environ.get("OPENAI_API_KEY", None)

    @classmethod
    def from_config(self, config: dict):
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")

        required_keys = {"description", "model", "interface", "calls"}
        optional_keys = set()
        _check_config(config["spec"], required_keys, optional_keys)

        return ChatGPTTool(**config["spec"])

    def to_config(self):
        return yaml.dump(
            {
                "apiVersion": "metatools.thatone.ai/v1",
                "kind": "ChatGPTTool",
                "metadata": {"name": self.name},
                "spec": {
                    "interface": "openai-chat",
                    "model": self.model,
                    "description": self.description,
                    "calls": self.calls,
                },
            }
        )

    def __call__(self, **kwargs):
        params = {"result": None}
        params.update(kwargs)

        for call in self.calls:
            user_prompt = Template(call["userPrompt"]).substitute(params)
            system_prompt = Template(call["systemPrompt"]).substitute(params)

            orig_key = openai.api_key
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            client = openai.OpenAI(api_key=self.api_key)
            result = (
                client.chat.completions.create(model=self.model, messages=messages)
                .choices[0]
                .message.content
            )

            if "retain" in call:
                match_result = re.match(call["retain"], result)
                if match_result:
                    result = match_result.group(1)

            params["result"] = result

        return result
