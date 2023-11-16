import requests
from string import Template
import yaml
from transformers.tools import Tool
from itllib import Itl
from itllib.clusters import ResourceController

from . import Metatool, _check_config


class RestApiMetatool(Metatool):
    def __init__(self, itl: Itl, cluster):
        super().__init__(itl, cluster, "RestApiTool")

    def create_tool(self, config):
        return RestApiTool.from_config(config)


class RestApiTool(Tool):
    def __init__(
        self,
        description,
        type,
        method: str = None,
        url: str = None,
        headers: dict = {},
        params: dict = {},
        data: str = None,
        backoff: float = None,
        attempts: int = None,
    ):
        super().__init__()
        self.description = description
        self.type = type

        self.method = method
        self.url = url
        self.headers = headers
        self.params = params
        self.data = data
        self.backoff = backoff
        self.attempts = attempts

    @classmethod
    def from_config(cls, config: dict):
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")

        required_keys = {"method", "url", "type", "description"}
        optional_keys = {"headers", "params", "data", "backoff", "attempts"}
        _check_config(config["spec"], required_keys, optional_keys)

        return RestApiTool(**config["spec"])

    def to_config(self):
        return {
            "apiVersion": "metatools.thatone.ai/v1",
            "kind": "RestApiTool",
            "metadata": {"name": self.name},
            "spec": {
                "method": self.method,
                "url": self.url,
                "type": self.type,
                "headers": self.headers,
                "params": self.params,
                "data": self.data,
                "backoff": self.backoff,
                "attempts": self.attempts,
            },
        }

    def __call__(self, **kwargs):
        method = Template(self.method).substitute(kwargs)
        url = Template(self.url).substitute(kwargs)

        if self.headers:
            headers = {}
            for key, value in self.headers.items():
                if isinstance(key, str):
                    key = Template(key).substitute(kwargs)
                if isinstance(value, str):
                    value = Template(value).substitute(kwargs)
                headers[key] = value
        else:
            headers = None

        if self.params:
            params = {}
            for key, value in self.params.items():
                if isinstance(key, str):
                    key = Template(key).substitute(kwargs)
                if isinstance(value, str):
                    value = Template(value).substitute(kwargs)
                params[key] = value
        else:
            params = None

        if isinstance(self.data, str):
            string_data = Template(self.data).substitute(kwargs)
            json_data = None
        elif isinstance(self.data, dict):
            string_data = None
            json_data = {}
            for key, value in self.data.items():
                if isinstance(key, str):
                    key = Template(key).substitute(kwargs)
                if isinstance(value, str):
                    value = Template(value).substitute(kwargs)
                json_data[key] = value
        else:
            string_data = None
            json_data = None

        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            data=string_data,
            json=json_data,
            stream=True,
        )

        return response.text
