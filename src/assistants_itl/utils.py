import re
from pydantic import BaseModel
import yaml
import json

from .globals import prompts, configs


def _resolve_config(config_path):
    global prompts, configs
    group, version, kind, name = config_path.split("/")

    if group != "assistants.thatone.ai" or version != "v1":
        print(f"Can't retrieve config from {group}/{version}")
        return ""

    if kind == "Prompt":
        prompt_config = prompts.get(config_path, None)
        if prompt_config == None:
            print("Missing prompt:", config_path)
            return ""
        return prompt_config.prompt

    if kind == "Config":
        config_config = configs.get(config_path, None)
        if config_config == None:
            print("Missing config:", config_path)
            return ""
        return config_config


class ConfigTemplate:
    def __init__(self, template):
        self.template = template
        self.pattern = re.compile(
            r"""
            \$(?:
            (?P<named>[_a-z][_a-z0-9]*)      |   # identifier
            {(?P<braced>[_a-zA-Z][_/.a-zA-Z0-9\-]*) (?:\s*\|\s* (?P<type>(float|string|int|yaml|json)))?}   |   # braced identifier
            (?P<invalid>)                           # ill-formed delimiter expr
            )
            """,
            re.VERBOSE,
        )

    def substitute(self, mapping={}, **kws):
        conversion = str

        def resolve(match):
            nonlocal conversion

            var_name = match.group("named") or match.group("braced")
            type = match.group("type") or "string"
            result = mapping.get(var_name, kws.get(var_name, None))
            if result == None:
                try:
                    result = _resolve_config(var_name)
                except ValueError:
                    print("Failed to resolve variable:", var_name)
                    return ""
            else:
                result = str(result)

            if type == "yaml":
                return yaml.dump(result)
            if type == "json":
                return json.dumps(result)
            if type in ("string", None):
                return result

            start, end = match.span()
            if start != 0 or end != len(self.template):
                return result

            if type == "float":
                conversion = float
                return result
            if type == "int":
                conversion = int
                return result

            print("Unknown type:", type)
            return result

        result = self.pattern.sub(resolve, self.template)
        return conversion(result)
