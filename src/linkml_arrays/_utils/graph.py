from __future__ import annotations

from collections.abc import Iterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from graphlib import TopologicalSorter

import h5py
import numpy as np
import zarr
from linkml_runtime import SchemaView
from pydantic import BaseModel

@dataclass(slots=True)
class GraphEdge:
    """Relationship between 2 nodes in linkml labeled array object"""
    parent: int
    child: int
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
    obj: BaseModel
    class_name: str
    identifier: str | None
    incoming: list[GraphEdge] = field(default_factory=list)
    outgoing: list[GraphEdge] = field(default_factory=list)

    @property
    def key(self) -> int:
        return id(self.obj)

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
        self.nodes: dict[int, GraphNode] = {}
        self.root: int | None = None
        self._identifier_index: dict[str, int] = {}

    @classmethod
    def from_root(
        cls,
        root: BaseModel,
        schemaview: SchemaView,
    ) -> "ObjectGraph":

        graph = cls()
        graph.root = id(root)
        graph._discover(root, schemaview)

        return graph

    def __contains__(self, obj: BaseModel | int) -> bool:
        if isinstance(obj, int):
            return obj in self.nodes

        return id(obj) in self.nodes

    def __getitem__(self, obj: BaseModel | int):
        if isinstance(obj, int):
            return self.nodes[obj]

        return self.nodes[id(obj)]

    def __iter__(self) -> Iterator[GraphNode]:
        return iter(self.nodes.values())

    def by_identifier(self, identifier: str) -> GraphNode:
        return self.nodes[self._identifier_index[identifier]]

    def _discover(
            self,
            obj: BaseModel,
            schemaview: SchemaView,
    ) -> GraphNode:
        oid = id(obj)

        if oid in self.nodes:
            return self.nodes[oid]

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

        self.nodes[oid] = node

        if identifier is not None:
            self._identifier_index[identifier] = oid

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

class YAMLGraphArraySerializer:
    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
        output_dir: Path,
        write_array: Callable,
        array_format: str,
    ):
        self.graph = graph
        self.schemaview = schemaview
        self.output_dir = output_dir
        self.write_array = write_array
        self.array_format = array_format

        self.serialized = {}

    def make_filename(self, node: GraphNode, slot) -> Path:
        if node.identifier is not None:
            return self.output_dir / f"{node.identifier}.{slot.name}"

        if node.incoming:
            edge = node.incoming[0]
            parent = self.graph[edge.parent]

            if parent.identifier is not None:
                return (
                        self.output_dir
                        / f"{parent.identifier}.{edge.slot_name}.{slot.name}"
                )

        raise ValueError(
            f"Cannot determine filename for {node.class_name}"
        )

    def serialize(self):
        for node in self.graph.dependency_order():
            self.serialized[node.key] = self.serialize_node(node)

        return self.serialized[self.graph.root]

    def serialize_node(self, node):
        result = {}

        for slot_name, value in vars(node.obj).items():
            slot = self.schemaview.induced_slot(
                slot_name,
                node.class_name,
            )

            if slot.array:
                filename = self.make_filename(
                    node,
                    slot,
                )

                output = self.write_array(
                    value,
                    filename,
                )

                result[slot_name] = {
                    "source": [{
                        "file": f"./{output.as_posix()}",
                        "format": self.array_format,
                    }]
                }

            elif isinstance(value, BaseModel):
                result[slot_name] = self.serialized[id(value)]
            else:
                result[slot_name] = value

        return result


class Hdf5GraphSerializer:
    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
        h5file: h5py.File,
    ):
        self.graph = graph
        self.schemaview = schemaview
        self.h5file = h5file

        self.groups: dict[int, h5py.Group] = {}

    def serialize(self):
        root = self.graph[self.graph.root]
        self.groups[root.key] = self.h5file

        for node in self.graph.hierarchy_order():
            self.serialize_node(node)

    def serialize_node(self, node: GraphNode):
        if node.key not in self.groups:
            edge = node.incoming[0]
            parent_group = self.groups[edge.parent]
            group = parent_group.create_group(edge.slot_name)
            self.groups[node.key] = group

        group = self.groups[node.key]

        for slot_name, value in vars(node.obj).items():
            slot = self.schemaview.induced_slot(
                slot_name,
                node.class_name,
            )

            if slot.array:
                group.create_dataset(
                    slot.name,
                    data=value,
                )
            elif isinstance(value, BaseModel):
                pass
            else:
                group.attrs[slot_name] = value

class ZarrGraphSerializer:
    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
        root: zarr.Group,
    ):
        self.graph = graph
        self.schemaview = schemaview
        self.root = root

        self.groups: dict[int, zarr.Group] = {}

    def serialize(self):
        root_node = self.graph[self.graph.root]
        self.groups[root_node.key] = self.root

        for node in self.graph.hierarchy_order():
            self.serialize_node(node)

    def serialize_node(self, node: GraphNode):
        if node.key not in self.groups:
            edge = node.incoming[0]
            parent_group = self.groups[edge.parent]

            group = parent_group.create_group(edge.slot_name)
            self.groups[node.key] = group

        group = self.groups[node.key]

        for slot_name, value in vars(node.obj).items():
            slot = self.schemaview.induced_slot(
                slot_name,
                node.class_name,
            )

            if slot.array:
                group.create_array(
                    slot.name,
                    data=np.asarray(value),
                )
            elif isinstance(value, BaseModel):
                continue
            else:
                group.attrs[slot_name] = value

class YamlGraphSerializer:
    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
    ):
        self.graph = graph
        self.schemaview = schemaview
        self.serialized: dict[int, dict] = {}

    def serialize(self) -> dict:
        for node in self.graph.dependency_order():
            self.serialized[node.key] = self.serialize_node(node)

        return self.serialized[self.graph.root]

    def serialize_node(self, node: GraphNode) -> dict:
        result = {}

        for slot_name, value in vars(node.obj).items():
            slot = self.schemaview.induced_slot(
                slot_name,
                node.class_name,
            )

            if slot.array:
                result[slot_name] = value
            elif isinstance(value, BaseModel):
                result[slot_name] = self.serialized[id(value)]
            else:
                result[slot_name] = value

        return result