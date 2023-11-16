from transformers.tools import Tool
from itllib import Itl
from itllib.clusters import ResourceController

from . import Metatool, _check_config


class StringMetatool(Metatool):
    def __init__(self, itl: Itl, cluster):
        super().__init__(itl, cluster, "StringTool")

    def create_tool(self, config):
        return StringTool.from_config(config)


class StringTool(Tool):
    def __init__(self, description, type, operations):
        super().__init__()
        self.description = description
        self.type = type

        self.operations = operations

    @classmethod
    def from_config(cls, config: dict):
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")

        required_keys = {"type", "description", "operations"}
        optional_keys = {"value"}
        _check_config(config["spec"], required_keys, optional_keys)

        return StringTool(**config["spec"])

    def to_config(self):
        return {
            "apiVersion": "metatools.thatone.ai/v1",
            "kind": "StringTool",
            "metadata": {"name": self.name},
            "spec": {"type": self.type, "value": self.value},
        }

    def __call__(self, table):
        for operation in self.operations:
            result = table
            op = operation["op"]
            if op == "format":
                result = (operation["template"].format(**kwargs) for kwargs in table)
            elif op == "join":
                result = [operation["separator"].join(result)]

        return result


# content
# - generate via api metatool

# questions
# - generate via constants
# - generate via user input
# - maybe generate via llm

# answers
# - generate via llm (note: llm can't output structured data)

# sheets
# - preset

# templates
# - generate via llm

# Consequences:
# - llm invocates must be iterated. the inputs should be tables, not datapoints.

# content = fetch(...)
# questions = constant_qs(content, character)
# answers = answer_questions(questions)
# sheet = link(answers)
# template = compile(sheet)

# # functions are good for creating
# # need ways to edit after creation

# editing content:
# - find & replace
# editing questions:
# - insert, delete, abstract, concretize
# editing answers:
# - focus, zoom out, correct, discard, ...
# editing sheets:
# - include, exclude, condense, ellaborate
# editing prompts:
# - debug
# editing template:
# - format
# ~

# each assistant gets its own channel
# if it generates typed output, that goes into a separate stream
