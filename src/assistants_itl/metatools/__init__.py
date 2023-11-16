import yaml

from transformers.tools import Tool
from itllib import Itl
from itllib.clusters import ResourceController


def _check_config(config, required_keys: set, optional_keys: set):
    if not required_keys.issubset(config.keys()):
        raise ValueError(
            "Config is missing required keys:", required_keys - config.keys()
        )

    allowed_keys = required_keys.union(optional_keys)
    if not allowed_keys.issuperset(config.keys()):
        raise ValueError(
            "Config contains extraneous keys:", config.keys() - allowed_keys
        )


class Metatool:
    def __init__(self, itl: Itl, cluster, kind):
        self.itl = itl
        self.cluster = cluster
        self.tools = {}
        self._parents = []
        self.kind = kind
        print(
            "Loaded metatool controller for",
            cluster,
            "metatools.thatone.ai",
            "v1",
            kind,
        )
        itl.controller(cluster, "metatools.thatone.ai", "v1", kind)(self.controller)

    async def controller(self, pending: ResourceController):
        async for op in pending:
            config = await op.new_config()
            if config == None:
                # Delete the tool
                self._remove_tool(pending.name)
                await op.accept()
                print("Deleted", f"{self.kind}/{pending.name}")
                continue

            try:
                result = self.create_tool(config)
                if result == None:
                    await op.reject()
                    print("Rejected", f"{self.kind}/{pending.name}")
                    continue

                self._add_tool(pending.name, result)
                await op.accept()
                print("Loaded", f"{self.kind}/{pending.name}")

            except ValueError as e:
                await op.reject()
                print(f"Failed to load tool {self.kind}/{pending.name}: {e}")

    def create_tool(self, config):
        raise NotImplementedError()

    async def load_existing(self):
        tools = await self.itl.resource_read_all(
            self.cluster, "metatools.thatone.ai", "v1", self.kind
        )
        if tools:
            for config in tools:
                try:
                    result = self.create_tool(config["config"])
                    self._add_tool(config["name"], result)
                    print("Loaded", self.kind, config["name"])
                except ValueError as e:
                    print(f'Failed to load tool {config["name"]}: {e}')

    def _register_parent(self, toolset):
        self._parents.append(toolset)

    def _add_tool(self, name, tool):
        self.tools[name] = tool
        key = self.kind + "/" + name
        print("Adding tool:", key)
        for parent in self._parents:
            parent.tools[key] = tool

    def _remove_tool(self, name):
        if name in self.tools:
            del self.tools[name]

        key = self.kind + "/" + name
        for parent in self._parents:
            if key in parent.tools:
                del parent.tools[key]


class MetaToolSet:
    def __init__(self, itl: Itl, cluster):
        self.itl = itl
        self.cluster = cluster
        self.metatools = {}
        self.tools = {}

    def add_metatool(self, metatool: Metatool):
        metatool._register_parent(self)
        self.metatools[metatool.kind] = metatool

        for name, tool in metatool.tools.items():
            key = metatool.kind + "/" + name
            self.tools[key] = tool

    def get_tool(self, name):
        return self.tools.get(name)
