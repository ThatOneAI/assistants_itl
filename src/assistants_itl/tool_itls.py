import os
import asyncio
from contextlib import contextmanager
import re
from string import Template
from typing import Union
import openai
from pydantic import BaseModel
import requests
import traceback

from itllib import ResourceController

from .globals import *
from .utils import ConfigTemplate


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


def format_message(format, kwargs):
    if isinstance(format, str):
        return ConfigTemplate(format).substitute(kwargs)
    elif isinstance(format, dict):
        return {k: format_message(v, kwargs) for k, v in format.items()}
    elif isinstance(format, list):
        return [format_message(v, kwargs) for v in format]
    else:
        return format


@tools.register(itl, CLUSTER, "tools.thatone.ai", "v1", "SendTool")
class SendTool(BaseModel):
    description: str
    loopSecret: str = None
    streamName: str = None
    sendUrl: str = None
    stream: str = None
    format: Union[str, dict, list] = None
    join: str = None
    print: bool = False
    synchronous: bool = False

    def get_send_url(self):
        if self.sendUrl:
            return self.sendUrl
        elif self.stream:
            stream = streams[self.stream]
            return stream.get_send_url()
        elif self.loopSecret and self.streamName:
            loop = loops[self.loopSecret]
            return f"https://{loop.get_endpoint()}/send/{self.streamName}"
        else:
            raise ValueError(
                "SendTool must have either: (*) 'sendUrl', (*) 'stream', or (*) 'loopSecret' and 'streamName'"
            )

    def __call__(self, *args, **kwargs):
        global itl

        try:
            if args and kwargs:
                raise ValueError(
                    "SendTool can only be called with args or kwargs, not both"
                )

            if self.format:
                if args:
                    raise ValueError("SendTool with 'format' must use kwargs")
                message = format_message(self.format, kwargs)

            elif self.join:
                if kwargs:
                    raise ValueError("SendTool with 'join' must use args, not kwargs")
                message = self.join.join(str(x) for x in args)
            else:
                message = args or kwargs

            if self.print:
                print(message)

            sendUrl = self.get_send_url()

            if self.synchronous:
                itl.stream_send_sync(sendUrl, message)
            else:
                itl.stream_send(sendUrl, message)

            return message
        except Exception as e:
            traceback.print_exc()


@tools.register(itl, CLUSTER, "tools.thatone.ai", "v1", "RestApiTool")
class RestApiTool(BaseModel):
    description: str
    method: str
    url: str
    headers: dict = {}
    params: dict = {}
    data: str = None
    backoff: float = None
    attempts: int = None

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


@tools.register(itl, CLUSTER, "tools.thatone.ai", "v1", "ChatGptTool")
class ChatGptTool(BaseModel):
    description: str
    model: str
    calls: list[dict]

    def __call__(self, **kwargs):
        params = {"result": None}
        params.update(kwargs)

        for call in self.calls:
            user_prompt = Template(call["userPrompt"]).substitute(params)
            system_prompt = Template(call["systemPrompt"]).substitute(params)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            client = openai.OpenAI(api_key=OPENAI_API_KEY)
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


@tools.register(itl, CLUSTER, "tools.thatone.ai", "v1", "EditConfigTool")
class EditConfigTool(BaseModel):
    description: str
    config: str

    def __call__(self, key, value):
        global itl, configs

        group, version, kind, name = self.config.split("/")

        try:
            current_config = configs[self.config][key]
        except KeyError:
            print(f"Must create the config {self.config} before editing it")

        config_piece = current_config
        key_pieces = key.split(".")
        for key_piece in key_pieces[:-1]:
            config_piece = config_piece.setdefault(key_piece, {})
        config_piece[key_pieces[-1]] = value

        looper = asyncio.get_event_loop()
        looper.create_task(
            itl.resource_apply(
                CLUSTER,
                {
                    "apiVersion": f"{group}/{version}",
                    "kind": kind,
                    "metadata": {"name": name},
                    "spec": current_config,
                },
                False,
            )
        )

        return current_config
