from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import zarr
from linkml_runtime import SchemaView
from pydantic import BaseModel

from linkml_arrays.graph_utils.graph import GraphNode, ObjectGraph


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
                return self.output_dir / f"{parent.identifier}.{edge.slot_name}.{slot.name}"

        raise ValueError(f"Cannot determine filename for {node.class_name}")

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
                    "source": [
                        {
                            "file": f"./{output.as_posix()}",
                            "format": self.array_format,
                        }
                    ]
                }

            elif isinstance(value, BaseModel):
                result[slot_name] = self.serialized[self.graph[value].key]
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
                result[slot_name] = self.serialized[self.graph[value].key]
            else:
                result[slot_name] = value

        return result
