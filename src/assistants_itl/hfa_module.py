from contextlib import contextmanager
import os
import re
from string import Template
from typing import Generator, Optional, Union
from pydantic import BaseModel
import random
from uuid import uuid1
from time import time
import asyncio

from transformers.tools import Tool, OpenAiAgent, agents
from itllib import Itl, ResourceController
import yaml

from .globals import *
from .utils import ConfigTemplate

NODE_ID = f"{int(time()*1000)}-{random.randint(0, 1000000000000)}"
SEQUENCE = 0


def _create_task_id():
    global NODE_ID, SEQUENCE
    SEQUENCE += 1
    return f"tasklog-{NODE_ID}-{SEQUENCE}"


class ToolWrapper(Tool):
    def __init__(self, function_resource):
        self.description = function_resource.description
        self.function = function_resource

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class HFAssistantConfig(BaseModel):
    codeModel: str
    streams: list[str]
    header: str
    tools: dict[str, str]
    examples: list[str]


class TaskLog(BaseModel):
    prompt: str
    tools: dict[str, str] = {}
    steps: Optional[str] = None
    code: Optional[str] = None


class Stream(BaseModel):
    connectUrl: str
    incomingFormat: str = None
    incomingFilter: Union[str, list, dict] = None


class HFAssistant:
    def __init__(self, config: HFAssistantConfig):
        global tools, prompts, tasklogs

        self.header = ConfigTemplate(config.header)
        self.streams = config.streams
        self.tools = config.tools
        self.examples = config.examples

        self.agent = _create_agent(config.codeModel)
        self.available_tools = tools
        self.available_prompts = prompts
        self.available_examples = tasklogs

        # TODO: expose streams as tools

    async def connect(self):
        tasks = []

        for stream_config_path in self.streams:
            tasks.append(itl.resource_read(CLUSTER, *stream_config_path.split("/")))

        results = await asyncio.gather(*tasks)
        for i, stream_config_json in enumerate(results):
            if stream_config_json == None:
                raise ValueError("Missing stream config for", self.streams[i])
            if "spec" not in stream_config_json:
                raise ValueError("Missing spec in stream config for", self.streams[i])

            stream_config = Stream(**stream_config_json["spec"])
            self._handle_messages(stream_config)

    def configure(self, config: HFAssistantConfig):
        if self.streams != config.streams:
            raise ValueError("Cannot change streams after initialization")

        self.tools = config.tools
        self.examples = config.examples
        self.header = ConfigTemplate(config.header)
        self.agent = _create_agent(config.codeModel)

    def chat(self, message: str):
        # TODO: Tasks coming in from the stream should append to the history, tasks
        # coming in from the TaskLog controller should not

        self.agent.prepare_for_new_chat()

        if self.agent.cached_tools:
            self.agent.cached_tools.clear()
        for name, reference in self.tools.items():
            if reference in self.available_tools:
                self.agent.toolbox[name] = ToolWrapper(self.available_tools[reference])
            else:
                print(f"Missing tool: {reference}")
        self.agent.chat_history = self._assemble_history()

        tasklog = None

        def store_response(explanation, code):
            nonlocal tasklog
            tasklog = TaskLog(
                prompt=message, tools=self.tools, steps=explanation, code=code
            )

        with _capture_response(store_response) as response:
            self.agent.chat(message)

        return tasklog

    def _assemble_header(self):
        return self.header.substitute()

    def _assemble_tool_description(self, tools):
        tool_lines = []
        for name, description in tools.items():
            tool_lines.append(f"- {name}: {description}")
        tool_description = "\n".join(tool_lines)
        return f"Tools:\n{tool_description}"

    def _assemble_task_log(self, tasklog, previous_tools={}):
        parts = []

        if previous_tools != tasklog.tools:
            parts.append(self._assemble_tool_description(tasklog.tools))
            parts.append("=====")

        parts.append(f"{tasklog.prompt}")
        parts.append(f"Assistant: {tasklog.steps}")
        parts.append(f"```python\n{tasklog.code}\n```")

        return "\n\n".join(parts)

    def _assemble_history(self):
        current_tools = {
            name: tool.description for name, tool in self.agent.toolbox.items()
        }
        tasks = []

        header = self._assemble_header()
        if header:
            tasks.append(header)

        previous_tools = {}

        for task_name in self.examples:
            if task_name not in self.available_examples:
                print(f"Missing example task: {task_name}")
                continue

            task = self.available_examples[task_name]
            tasks.append(self._assemble_task_log(task, previous_tools))
            previous_tools = task.tools

        if current_tools != previous_tools:
            tasks.append(self._assemble_tool_description(current_tools))

        return "\n\n".join(tasks) + "\n"

    def _handle_messages(self, stream_config):
        incoming_template = ConfigTemplate(stream_config.incomingFormat or "${message}")
        incoming_filter = stream_config.incomingFilter

        @itl.ondata(stream_config.connectUrl)
        async def ondata(*args, **kwargs):
            if args and not kwargs:
                if len(args) != 1:
                    return
                message = args[0]
            elif kwargs and not args:
                message = kwargs
            else:
                return

            if not _check_filter(message, incoming_filter):
                return None

            if not isinstance(message, dict):
                message = {"message": message}

            incoming = incoming_template.substitute(**message)
            tasklog = self.chat(incoming)

            tasklog_config = {
                "apiVersion": "assistants.thatone.ai/v1",
                "kind": "TaskLog",
                "metadata": {"name": _create_task_id()},
                "spec": tasklog.model_dump(),
            }

            # Push the log to the cluster
            await itl.resource_create(CLUSTER, tasklog_config, attach_prefix=True)

            # Add the log to the list of known tasklogs
            tasklog_name = itl.attach_cluster_prefix(
                CLUSTER, tasklog_config["metadata"]["name"]
            )
            tasklog_id = f"assistants.thatone.ai/v1/TaskLog/{tasklog_name}"
            tasklogs[tasklog_id] = tasklog

            # Add the log to the history
            self.examples.append(tasklog_id)

        return ondata


def _check_filter(message, filter):
    if filter == None:
        return True

    if isinstance(message, str):
        if not isinstance(filter, str):
            return False
        result = re.match(filter, message) != None
        return result

    if not isinstance(message, dict):
        return False

    if isinstance(filter, list):
        for key in filter:
            if key not in message:
                return False
        return True

    if not isinstance(filter, dict):
        return False

    for key, value in filter.items():
        if key not in message:
            return False
        if not _check_filter(message[key], value):
            return False

    return True


@contextmanager
def _no_fetch():
    orig_fn = agents.download_prompt
    try:
        agents.download_prompt = lambda *args, **kwargs: ""
        yield
    finally:
        agents.download_prompt = orig_fn


def _create_agent(code_model):
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)

    with _no_fetch():
        result = OpenAiAgent(model=code_model, api_key=openai_api_key)
        result.toolbox.clear()
        return result


@contextmanager
def _capture_response(callback) -> Generator[None, None, None]:
    orig_fn = agents.clean_code_for_chat

    def _hijack_clean_code_for_chat(*args, **kwargs):
        nonlocal callback
        explanation, code = orig_fn(*args, **kwargs)
        callback(explanation, code)
        return explanation, code

    try:
        agents.clean_code_for_chat = _hijack_clean_code_for_chat
        yield
    finally:
        agents.clean_code_for_chat = orig_fn
