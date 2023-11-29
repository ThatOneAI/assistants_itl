import asyncio
from typing import Any, Union
from pydantic import BaseModel
import yaml


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


class ResourceController:
    def __init__(self, itl: Itl, cluster: str, api_version: str, kind: str):
        self.itl = itl
        self.cluster = cluster
        self.kind = kind
        self._parents = []
        self.resources: dict[str, Any] = {}

        self.group, self.version = api_version.split("/")

        print(
            "Loaded resource controller for",
            cluster,
            self.group,
            self.version,
            kind,
        )
        itl.controller(cluster, self.group, self.version, kind)(self.controller)

    async def controller(self, pending: ResourceController):
        async for op in pending:
            config = await op.new_config()
            if config == None:
                # Delete the resource
                resource = self._get_resource(pending.name)
                self.delete_resource(resource)
                self._remove_resource(pending.name)
                await op.accept()
                print("Deleted", f"{self.kind}/{pending.name}")
                continue

            name = config["metadata"]["name"]
            old_config = await op.old_config()
            old_resource = self._get_resource(name)

            try:
                if old_resource == None:
                    result = self.create_resource(config)
                    if result == None:
                        await op.reject()
                        print("Rejected", f"{self.kind}/{pending.name}")
                        continue

                    self._add_resource(pending.name, result)
                    print("Created", f"{self.kind}/{pending.name}")
                else:
                    result = old_resource
                    self.update_resource(result, config)
                    print("Reconfigured", f"{self.kind}/{pending.name}")

                await op.accept()

            except ValueError as e:
                await op.reject()
                print(f"Failed to load resource {self.kind}/{pending.name}: {e}")

    def create_resource(self, config):
        raise ValueError("create_resource not implemented")

    def update_resource(self, resource, config):
        name = config["metadata"]["name"]
        result = self.create_resource(config)
        if result == None:
            raise ValueError("create_resource returned None for", config)
        self._add_resource(name, result)

    def delete_resource(self, resource):
        pass

    async def load_existing(self):
        resources = await self.itl.resource_read_all(
            self.cluster, self.group, self.version, self.kind
        )
        if resources:
            for config in resources:
                try:
                    result = self.create_resource(config["config"])
                    self._add_resource(config["name"], result)
                    print("Loaded", self.kind, config["name"])
                except ValueError as e:
                    print(f'Failed to load resource {config["name"]}: {e}')

    def _register_parent(self, resource_set):
        self._parents.append(resource_set)

    def _add_resource(self, name, resource):
        self.resources[name] = resource
        key = self.group + "/" + self.version + "/" + self.kind + "/" + name
        for parent in self._parents:
            parent.resources[key] = resource

    def _get_resource(self, name):
        return self.resources.get(name)

    def _remove_resource(self, name):
        if name in self.resources:
            del self.resources[name]

        key = self.group + "/" + self.version + "/" + self.kind + "/" + name
        for parent in self._parents:
            if key in parent.resources:
                del parent.resources[key]


class DefaultResourceController(ResourceController):
    def __init__(
        self, resource_cls, itl: Itl, cluster: str, api_version: str, kind: str
    ):
        super().__init__(itl, cluster, api_version, kind)
        self.resource_cls = resource_cls

    def create_resource(self, config):
        if "spec" not in config:
            raise ValueError("Config is missing required key: spec")
        return self.resource_cls(**config["spec"])


class ResourceSet:
    def __init__(self):
        self.resources = {}

    def register(self, itl, cluster, api_version, kind):
        def decorator(controller_cls):
            if issubclass(controller_cls, ResourceController):
                controller = controller_cls(itl, cluster, api_version, kind)
            elif issubclass(controller_cls, BaseModel):
                controller = DefaultResourceController(
                    controller_cls, itl, cluster, api_version, kind
                )

            self.register_controller(controller)
            asyncio.run(controller.load_existing())
            return controller_cls

        return decorator

    def register_controller(self, controller: ResourceController):
        controller._register_parent(self)
        for name, controller in controller.resources.items():
            key = (
                controller.group
                + "/"
                + controller.version
                + "/"
                + controller.kind
                + "/"
                + name
            )
            self.resources[key] = controller

    def get_resource(self, name):
        return self.resources.get(name)
