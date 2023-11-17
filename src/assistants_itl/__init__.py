import os
import asyncio
from contextlib import contextmanager
import traceback

from transformers.tools import OpenAiAgent
from itllib import Itl, ResourceController

from .metatools import Metatool, MetaToolSet, _check_config
from .metatools.stringutils import StringMetatool
from .metatools.api import RestApiMetatool
from .metatools.llm import LanguageMetatool
from .metatools.constants import TemplateMetatool
from .hfassistant_module import HFMetaAssistantModule


CONFIG_PATH = os.path.join("config.yaml")
SECRETS_PATH = "./secrets"
CLUSTER = "assistants"

itl = Itl()
itl.apply_config(CONFIG_PATH, SECRETS_PATH)
itl.start()

metatools = MetaToolSet(itl, CLUSTER)

prompts = {}
tasklogs = {}
tools = metatools.tools  # MetaToolSet will manage this
assistants = {}

# Set up the controllers


@itl.controller(CLUSTER, "assistants.thatone.ai", "v1", "Prompt")
async def prompt_controller(pending: ResourceController):
    async for op in pending:
        print("Prompt update:", pending.name)
        config = await op.new_config()
        if config == None:
            # Delete the prompt
            if pending.name in prompts:
                del prompts[pending.name]
            await op.accept()
            print("Deleted", f"Prompt/{pending.name}")
            continue

        if "prompt" not in config:
            print("Skipping prompt", pending.name, "because prompt is not in config")
            await op.reject()
            continue

        prompts[pending.name] = config["prompt"]
        await op.accept()
        print("Loaded", f"Prompt/{pending.name}")


@itl.controller(CLUSTER, "assistants.thatone.ai", "v1", "TaskLog")
async def tasklog_controller(pending: ResourceController):
    async for op in pending:
        print("Tasklog update:", pending.name)
        config = await op.new_config()
        if config == None:
            # Delete the prompt
            if pending.name in tasklogs:
                del tasklogs[pending.name]
            await op.accept()
            print("Deleted", f"TaskLog/{pending.name}")
            continue

        if "spec" not in config:
            print("Skipping TaskLog", pending.name, "because spec is not in config")
            await op.reject()
            continue

        spec = config["spec"]
        required_keys = {"prompt", "tools", "steps", "code"}
        optional_keys = set()
        try:
            _check_config(spec, required_keys, optional_keys)
        except ValueError as e:
            print("Error in", f"TaskLog/{pending.name}", e)
            await op.reject()
            continue

        if not isinstance(config["spec"]["tools"], dict):
            print("Skipping TaskLog", pending.name, "because tools is not a dict")
            await op.reject()
            continue

        # Set the prompt
        tasklogs[pending.name] = config["spec"]
        await op.accept()
        print("Loaded", f"TaskLog/{pending.name}")


@itl.controller(CLUSTER, "assistants.thatone.ai", "v1", "HFAssistant")
async def hfassistant_controller(pending: ResourceController):
    async for op in pending:
        print("assistant update:", pending.name)
        config = await op.new_config()
        if config == None:
            # Delete the assistant
            if pending.name in assistants:
                del assistants[pending.name]
            await op.accept()
            print("Deleted", f"HFAssistant/{pending.name}")
            continue

        if "spec" not in config:
            await op.reject()
            print("Skipping HFAssistant", pending.name, "because spec is not in config")
            continue
        spec = config["spec"]

        if "stream" not in spec:
            await op.reject()
            print("Skipping HFAssistant", pending.name, "because stream is not in spec")
            continue
        stream = spec["stream"]

        if await op.old_config() == None:
            # Create the assistant
            assistants[pending.name] = HFMetaAssistantModule(
                itl,
                CLUSTER,
                stream,
                _create_agent(),
                prompts,
                tasklogs,
                tools,
            )

        try:
            assistants[pending.name].configure(spec)
        except ValueError as e:
            print("Error in", f"HFAssistant/{pending.name}", e)
            await op.reject()
            continue

        await op.accept()
        print("Loaded", f"HFAssistant/{pending.name}")


@contextmanager
def _no_fetch():
    from transformers.tools import agents

    orig_fn = agents.download_prompt
    try:
        agents.download_prompt = lambda *args, **kwargs: ""
        yield
    finally:
        agents.download_prompt = orig_fn


def _create_agent():
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)

    with _no_fetch():
        result = OpenAiAgent(model="gpt-3.5-turbo", api_key=openai_api_key)
        result.toolbox.clear()
        return result


# Load the existing prompts, tasks, tools, and assistants


async def load_existing():
    retrieved_prompts = (
        await itl.resource_read_all(CLUSTER, "assistants.thatone.ai", "v1", "Prompt")
        or []
    )
    for data in retrieved_prompts:
        try:
            config = data["config"]
            name = data["name"]
            prompts[name] = config["prompt"]
        except KeyError:
            print("Failed to load prompt", name)

    retrieved_tasklogs = (
        await itl.resource_read_all(CLUSTER, "assistants.thatone.ai", "v1", "TaskLog")
        or []
    )
    for data in retrieved_tasklogs:
        try:
            config = data["config"]
            name = data["name"]

            if not isinstance(config["spec"]["tools"], dict):
                print("Skipping tasklog", name, "because tools is not a dict")
                continue

            tasklogs[name] = config["spec"]
            print("Loaded", f"TaskLog/{name}")
        except KeyError as e:
            print("Failed to load TaskLog", name, e)

    await load_metatool(StringMetatool(itl, CLUSTER))
    await load_metatool(RestApiMetatool(itl, CLUSTER))
    await load_metatool(LanguageMetatool(itl, CLUSTER))
    await load_metatool(TemplateMetatool(itl, CLUSTER))

    modules = await itl.resource_read_all(
        CLUSTER, "assistants.thatone.ai", "v1", "HFAssistant"
    )
    for data in modules:
        try:
            config = data["config"]
            spec = config["spec"]
            name = data["name"]
            assistants[name] = HFMetaAssistantModule(
                itl,
                CLUSTER,
                spec["stream"],
                _create_agent(),
                prompts,
                tasklogs,
                tools,
            )
            assistants[name].configure(config["spec"])
            print("Loaded", f"HFAssistant/{name}")
        except Exception as e:
            print("Failed to load HFAssistant", name, e, traceback.format_exc())


async def load_metatool(metatool: Metatool):
    metatools.add_metatool(metatool)
    try:
        await metatool.load_existing()
    except Exception as e:
        print("Failed to load metatool", metatool, e)


asyncio.run(load_existing())
