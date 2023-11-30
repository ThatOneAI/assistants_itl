import os
import asyncio
from contextlib import contextmanager
import re
from string import Template
import openai
from pydantic import BaseModel
import requests

from itllib import ResourceController

from .hfa_module import HFAssistantConfig, HFAssistant, TaskLog
from .globals import *


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


@prompts.register(itl, CLUSTER, "assistants.thatone.ai", "v1", "Prompt")
class Prompt(BaseModel):
    prompt: str


@tools.register(itl, CLUSTER, "tools.thatone.ai", "v1", "SendTool")
class SendTool(BaseModel):
    description: str
    sendUrl: str
    format: str = None
    join: str = None
    print: bool = False
    synchronous: bool = False

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

                if isinstance(self.format, str):
                    message = Template(self.format).substitute(**kwargs)
                elif isinstance(self.format, dict):
                    message = {
                        k: Template(v).substitute(**kwargs)
                        for k, v in self.format.items()
                    }
                elif isinstance(self.format, list):
                    message = [Template(v).substitute(**kwargs) for v in self.format]
            elif self.join:
                if kwargs:
                    raise ValueError("SendTool with 'join' must use args, not kwargs")
                message = self.join.join(str(x) for x in args)
            else:
                message = args or kwargs

            if self.print:
                print(message)

            if self.synchronous:
                itl.stream_send_sync(self.sendUrl, message)
            else:
                itl.stream_send(self.sendUrl, message)

            return message
        except Exception as e:
            print("Error in SendTool:", e)


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
