import asyncio
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
import os
import json
from pydantic import BaseModel

from transformers.tools import Agent, Tool, OpenAiAgent
from itllib import Itl

from .resources import ResourceController
from .globals import *


class ToolWrapper(Tool):
    def __init__(self, function_resource):
        self.description = function_resource.description
        self.function = function_resource

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class HFAssistantConfig(BaseModel):
    codeModel: str
    stream: str
    mission: str
    tools: dict[str, str]
    examples: list[str]


class HFAssistant:
    def __init__(self, config: HFAssistantConfig):
        global tools, prompts, tasklogs

        self.stream = config.stream
        self.mission = config.mission
        self.tools = config.tools
        self.examples = config.examples

        self.agent = _create_agent(config.codeModel)
        self.available_tools = tools.resources
        self.available_prompts = prompts.resources
        self.available_examples = tasklogs.resources

        itl.ondata(self.stream)(self._handle_messages)

    def configure(self, config: HFAssistantConfig):
        if self.stream != config.stream:
            raise ValueError("Cannot change stream after initialization")

        self.mission = config.mission
        self.tools = config.tools
        self.examples = config.examples

    def run(self, message: str):
        self.agent.prepare_for_new_chat()
        return self.chat(message)

    def chat(self, message: str):
        print("Incoming message:", message)
        if self.agent.cached_tools:
            self.agent.cached_tools.clear()
        for name, reference in self.tools.items():
            if reference in self.available_tools:
                self.agent.toolbox[name] = ToolWrapper(self.available_tools[reference])
        tools = {name: tool.description for name, tool in self.agent.toolbox.items()}
        self.agent.chat_history = self._assemble_prompt(tools)

        print("=== Result ===")
        result = self.agent.chat(message)
        print(result)

        return result

    def _assemble_tool_description(self, tools):
        tool_lines = []
        for name, description in tools.items():
            tool_lines.append(f"- {name}: {description}")
        tool_description = "\n".join(tool_lines)
        return f"Tools:\n{tool_description}"

    def _assemble_task_log(self, tasklog, tools={}):
        parts = []

        if tools:
            parts.append(self._assemble_tool_description(tools))
            parts.append("=====")

        parts.append(f"Human: {tasklog.prompt}")
        parts.append(f"Assistant: {tasklog.steps}")
        parts.append(f"```python\n{tasklog.code}\n```")

        return "\n\n".join(parts)

    def _assemble_prompt(self, current_tools):
        tasks = []

        mission = self.available_prompts.get(self.mission, None)
        if mission:
            tasks.append(mission.prompt)

        previous_tools = {}

        for task_name in self.examples:
            if task_name not in self.available_examples:
                print(f"Missing example task: {task_name}")
                continue

            task = self.available_examples[task_name]
            if previous_tools != task.tools:
                tools = task.tools
                previous_tools = tools
            else:
                tools = {}
            tasks.append(self._assemble_task_log(task, tools))

        if current_tools != previous_tools:
            tasks.append(self._assemble_tool_description(current_tools))

        return "\n\n".join(tasks) + "\n"

    async def _handle_messages(self, message):
        if not isinstance(message, str):
            print(f"Invalid message type: {type(message)}")
            return

        if message.startswith(">"):
            return None

        if message.startswith("+"):
            mode = "chat"
            message = message[1:]
        else:
            mode = "run"

        if not isinstance(message, str):
            print(f"Invalid message type: {type(message)}")
            return

        message = message.strip()
        if len(message) == 0:
            print("Rejecting empty message")
            return

        if mode == "run":
            self.run(message)
        elif mode == "chat":
            self.chat(message)
        else:
            print(f"Invalid mode: {mode}. Expected 'chat' or 'run'")
            return

        print("Done processing message")


@contextmanager
def _no_fetch():
    from transformers.tools import agents

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
