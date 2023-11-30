from contextlib import contextmanager
import os
from typing import Generator
from pydantic import BaseModel
import random
from uuid import uuid1
from time import time

from transformers.tools import Tool, OpenAiAgent, agents
from itllib import Itl, ResourceController

from .globals import *

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
    stream: str
    mission: str
    tools: dict[str, str]
    examples: list[str]


@tasklogs.register(itl, CLUSTER, "assistants.thatone.ai", "v1", "TaskLog")
class TaskLog(BaseModel):
    prompt: str
    tools: dict[str, str] = {}
    steps: str = ""
    code: str


class HFAssistant:
    def __init__(self, config: HFAssistantConfig):
        global tools, prompts, tasklogs

        self.stream = config.stream
        self.mission = config.mission
        self.tools = config.tools
        self.examples = config.examples

        self.agent = _create_agent(config.codeModel)
        self.available_tools = tools
        self.available_prompts = prompts
        self.available_examples = tasklogs

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
        with _save_response() as response:
            result = self.agent.chat(message)

        code = response.code
        explanation = response.explanation

        print("explanation:", explanation)
        print("code:", code)
        print("result:", result)

        tasklog = TaskLog(prompt=message, tools=tools, steps=explanation, code=code)

        return result, tasklog

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
            result, tasklog = self.run(message)
        elif mode == "chat":
            result, tasklog = self.chat(message)
        else:
            print(f"Invalid mode: {mode}. Expected 'chat' or 'run'")
            return

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

        print("Done processing message")


@assistants.register(itl, CLUSTER, "assistants.thatone.ai", "v1", "HFAssistant")
class HFAController(ResourceController):
    def create_resource(self, config):
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")
        return HFAssistant(HFAssistantConfig(**config["spec"]))

    def update_resource(self, resource: HFAssistant, config):
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")
        resource.configure(HFAssistantConfig(**config["spec"]))

    def delete_resource(self, resource):
        print("You'll need to restart the script to complete deletion of an assistant")
        return super().delete_resource(resource)


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


class _HFAgentResponse:
    explanation: str
    code: str


@contextmanager
def _save_response() -> Generator[_HFAgentResponse, None, None]:
    orig_fn = agents.clean_code_for_chat
    result = _HFAgentResponse()

    def _hijack_clean_code_for_chat(*args, **kwargs):
        nonlocal result
        explanation, code = orig_fn(*args, **kwargs)
        result.explanation = explanation or ""
        result.code = code or ""
        return result.explanation, result.code

    try:
        agents.clean_code_for_chat = _hijack_clean_code_for_chat
        yield result
    finally:
        agents.clean_code_for_chat = orig_fn
