from itllib import Itl
from transformers.tools import Tool

from . import Metatool, _check_config


class TemplateMetatool(Metatool):
    def __init__(self, itl: Itl, cluster):
        super().__init__(itl, cluster, "TemplateTool")

    def create_tool(self, config):
        print("creating TemplateTool")
        return TemplateTool.from_config(config)


class TemplateTool(Tool):
    def __init__(self, description, type, values):
        super().__init__()
        self.description = description
        self.type = type
        self.filters = values

    @classmethod
    def from_config(cls, config: dict):
        print("... from", config)
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")

        required_keys = {"description", "type", "values"}
        _check_config(config["spec"], required_keys, set())

        print("passed checks")

        return TemplateTool(**config["spec"])

    def to_config(self):
        return {
            "apiVersion": "metatools.thatone.ai/v1",
            "kind": "TemplateTool",
            "metadata": {"name": self.name},
            "spec": {
                "description": self.description,
                "type": self.type,
                "values": self.values,
            },
        }

    def __call__(self, **kwargs):
        for filter in self.filters:
            result = {}
            for key_template, value_template in filter.items():
                key = key_template.format(**kwargs)
                value = value_template.format(**kwargs)
                result[key] = value
