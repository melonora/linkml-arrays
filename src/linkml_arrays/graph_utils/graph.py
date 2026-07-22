from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from graphlib import TopologicalSorter
from uuid import UUID, uuid4

import h5py
import numpy as np
import yaml
import zarr
from linkml_runtime import SchemaView
from pydantic import BaseModel

@dataclass(slots=True)
class GraphEdge:
    """Relationship between 2 nodes in linkml labeled array object"""
    parent: UUID
    child: UUID
    slot_name: str
    multivalued: bool = False
    inlined: bool = False

    def __repr__(self) -> str:
        return (
            f"{self.parent} --{self.slot_name}--> "
            f"{self.child}"
        )


@dataclass(slots=True)
class GraphNode:
    """Node in a linkml labeled array object."""
    class_name: str
    obj: BaseModel | None = None
    identifier: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    incoming: list[GraphEdge] = field(default_factory=list)
    outgoing: list[GraphEdge] = field(default_factory=list)
    key: int = field(default_factory=uuid4, init=False)

    @property
    def parents(self) -> list[int]:
        return [e.parent for e in self.incoming]

    @property
    def children(self) -> list[int]:
        return [e.child for e in self.outgoing]

    def __repr__(self):
        return (
            f"GraphNode("
            f"{self.class_name}, "
            f"id={self.identifier!r})"
        )

class ObjectGraph:
    def __init__(self):
        self.nodes: dict[UUID, GraphNode] = {}
        self.root: int | None = None
        self._identifier_index: dict[str, UUID] = {}
        self._object_index: dict[int, UUID] = {}

    @classmethod
    def from_root(
        cls,
        root: BaseModel,
        schemaview: SchemaView,
    ) -> "ObjectGraph":

        graph = cls()
        root_node = graph._discover(root, schemaview)
        graph.root = root_node.key

        return graph

    @classmethod
    def from_hdf5(
            cls,
            source: str | Path,
            schemaview: SchemaView,
            root_class: str,
    ) -> "ObjectGraph":
        graph = cls()

        with h5py.File(source, "r") as f:
            schema_class = schemaview.get_class(root_class)

            root = graph._discover_hdf5_group(
                group=f,
                class_definition=schema_class,
                schemaview=schemaview,
            )
            graph.root = root.key

        return graph

    @classmethod
    def from_zarr(
            cls,
            source: str | Path,
            schemaview: SchemaView,
            root_class: str,
    ) -> "ObjectGraph":
        graph = cls()

        z = zarr.open(source, mode="r")
        schema_class = schemaview.get_class(root_class)

        root = graph._discover_zarr_group(
            group=z,
            class_definition=schema_class,
            schemaview=schemaview,
        )
        graph.root = root.key

        return graph

    @classmethod
    def from_yaml(
            cls,
            source: str | Path,
            schemaview: SchemaView,
            root_class: str,
            resolve_arrays=False
    ) -> "ObjectGraph":
        """Construct an ObjectGraph from a YAML document."""

        graph = cls()

        if isinstance(source, (str, Path)) and Path(source).exists():
            with open(source) as f:
                data = yaml.safe_load(f)
            base_path = Path(source).parent
        else:
            data = yaml.safe_load(source)
            base_path = Path(".")

        schema_class = schemaview.get_class(root_class)

        root = graph._discover_yaml_dict(
            input_dict=data,
            class_definition=schema_class,
            schemaview=schemaview,
            base_path=base_path,
            resolve_arrays=resolve_arrays,
        )

        graph.root = root.key

        return graph

    def __contains__(self, obj: BaseModel | int) -> bool:
        if isinstance(obj, int):
            return obj in self.nodes

        return id(obj) in self._object_index

    def __getitem__(self, obj: BaseModel | int):
        if isinstance(obj, UUID):
            return self.nodes[obj]

        return self.nodes[self._object_index[id(obj)]]

    def __iter__(self) -> Iterator[GraphNode]:
        return iter(self.nodes.values())

    def by_identifier(self, identifier: str) -> GraphNode:
        return self.nodes[self._identifier_index[identifier]]

    def _discover_zarr_group(
            self,
            group: zarr.Group,
            class_definition,
            schemaview: SchemaView,
    ) -> GraphNode:
        node = GraphNode(
            class_name=class_definition.name,
        )

        for name, value in group.attrs.items():
            node.values[name] = value

        self.nodes[node.key] = node

        for name, value in group.members():
            slot = schemaview.induced_slot(
                name,
                class_definition.name,
            )

            if slot.array:
                node.values[name] = value[()]
                continue

            child_class = schemaview.get_class(slot.range)

            child = self._discover_zarr_group(
                group=value,
                class_definition=child_class,
                schemaview=schemaview,
            )

            edge = GraphEdge(
                parent=node.key,
                child=child.key,
                slot_name=slot.name,
                multivalued=bool(slot.multivalued),
                inlined=bool(slot.inlined),
            )

            node.outgoing.append(edge)
            child.incoming.append(edge)

        return node

    def _discover_hdf5_group(
            self,
            group: h5py.Group,
            class_definition,
            schemaview: SchemaView,
    ) -> GraphNode:
        node = GraphNode(
            class_name=class_definition.name,
        )

        for name, value in group.attrs.items():
            node.values[name] = value

        self.nodes[node.key] = node

        for name, value in group.items():
            slot = schemaview.induced_slot(
                name,
                class_definition.name,
            )

            if slot.array:
                node.values[name] = value[()]
                continue

            child_class = schemaview.get_class(slot.range)

            child = self._discover_hdf5_group(
                group=value,
                class_definition=child_class,
                schemaview=schemaview,
            )

            edge = GraphEdge(
                parent=node.key,
                child=child.key,
                slot_name=slot.name,
                multivalued=bool(slot.multivalued),
                inlined=bool(slot.inlined),
            )

            node.outgoing.append(edge)
            child.incoming.append(edge)

        return node

    def _discover_yaml_dict(
            self,
            input_dict: dict,
            class_definition,
            schemaview: SchemaView,
            base_path: Path,
            resolve_arrays: bool,
    ) -> GraphNode:
        """Recursively discover a YAML object hierarchy."""

        node = GraphNode(
            class_name=class_definition.name,
        )

        self.nodes[node.key] = node

        for name, value in input_dict.items():

            slot = schemaview.induced_slot(
                name,
                class_definition.name,
            )

            if slot.array:
                if not resolve_arrays:
                    node.values[name] = value
                    continue

                sources = value.get("source")
                if sources is None:
                    raise ValueError(f"Array slot '{name}' has no source.")

                if len(sources) != 1:
                    raise NotImplementedError(
                        "Multiple array sources are not yet supported."
                    )

                source = sources[0]
                fmt = source.get("format")
                file = source.get("file")

                if fmt is None:
                    raise ValueError(
                        f"Array slot '{name}' has no format."
                    )

                if file is None:
                    raise ValueError(
                        f"Array slot '{name}' has no file."
                    )

                file = base_path / file

                if fmt == "numpy":
                    node.values[name] = np.load(file)
                elif fmt == "hdf5":
                    with h5py.File(file, "r") as f:
                        node.values[name] = f["data"][()]
                elif fmt == "zarr":
                    z = zarr.open(file, mode="r")
                    node.values[name] = z["data"][()]
                else:
                    raise ValueError(
                        f"Unsupported array format '{fmt}'."
                    )
                continue

            if isinstance(value, dict):
                child_class = schemaview.get_class(
                    slot.range,
                )

                child = self._discover_yaml_dict(
                    input_dict=value,
                    class_definition=child_class,
                    schemaview=schemaview,
                    base_path=base_path,
                    resolve_arrays=resolve_arrays,
                )

                edge = GraphEdge(
                    parent=node.key,
                    child=child.key,
                    slot_name=slot.name,
                    multivalued=bool(slot.multivalued),
                    inlined=bool(slot.inlined),
                )

                node.outgoing.append(edge)
                child.incoming.append(edge)
                continue

            node.values[name] = value
        return node

    def _discover(
            self,
            obj: BaseModel,
            schemaview: SchemaView,
    ) -> GraphNode:
        obj_id = id(obj)

        if obj_id in self._object_index:
            return self.nodes[self._object_index[obj_id]]

        class_name = type(obj).__name__
        cls = schemaview.get_class(class_name)
        identifier = None

        if cls is not None:
            id_slot = schemaview.get_identifier_slot(cls.name)
            if id_slot is not None:
                identifier = getattr(obj, id_slot.name)

        node = GraphNode(
            obj=obj,
            class_name=class_name,
            identifier=identifier,
        )

        self.nodes[node.key] = node
        self._object_index[obj_id] = node.key

        if identifier is not None:
            self._identifier_index[identifier] = node.key

        for slot_name, value in vars(obj).items():
            slot = schemaview.induced_slot(
                slot_name,
                class_name,
            )

            self._discover_value(
                value=value,
                parent=node,
                slot=slot,
                schemaview=schemaview,
            )

        return node

    def _discover_value(
            self,
            value: Any,
            parent: GraphNode,
            slot,
            schemaview: SchemaView,
    ):
        if isinstance(value, BaseModel):
            child = self._discover(value, schemaview)

            edge = GraphEdge(
                parent=parent.key,
                child=child.key,
                slot_name=slot.name,
                multivalued=False,
                inlined=bool(slot.inlined),
            )

            parent.outgoing.append(edge)
            child.incoming.append(edge)

            return

        if isinstance(value, dict):
            for item in value.values():
                self._discover_value(
                    item,
                    parent,
                    slot,
                    schemaview,
                )

            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                if not isinstance(item, BaseModel):
                    continue

                child = self._discover(
                    item,
                    schemaview,
                )

                edge = GraphEdge(
                    parent=parent.key,
                    child=child.key,
                    slot_name=slot.name,
                    multivalued=True,
                    inlined=bool(slot.inlined),
                )

                parent.outgoing.append(edge)
                child.incoming.append(edge)

    def dependency_order(self) -> Iterator[GraphNode]:
        ts = TopologicalSorter()

        for node in self.nodes.values():
            ts.add(
                node.key,
                *(edge.child for edge in node.outgoing),
            )

        for key in ts.static_order():
            yield self.nodes[key]

    def hierarchy_order(self) -> Iterator[GraphNode]:
        if self.root is None:
            return

        visited: set[int] = set()

        def visit(key: int):
            if key in visited:
                return

            visited.add(key)

            node = self.nodes[key]
            yield node

            for edge in node.outgoing:
                yield from visit(edge.child)

        yield from visit(self.root)