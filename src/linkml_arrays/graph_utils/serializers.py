"""Graph-based serializers for LinkML labeled array models.

The serializers in this module operate on an ``ObjectGraph`` rather than
directly traversing instantiated LinkML objects. This separates graph
construction from serialization and provides a common intermediate
representation for YAML, YAML with external array storage, HDF5, and
Zarr serialization.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import h5py
import numpy as np
import zarr
from linkml_runtime import SchemaView
from pydantic import BaseModel

from linkml_arrays.graph_utils.graph import GraphNode, ObjectGraph


class YAMLGraphArraySerializer:
    """Serialize an ObjectGraph to a YAML representation with external arrays.

    Object-valued LinkML attributes are serialized as nested YAML
    mappings, while array-valued attributes are written to external files
    using a caller-provided array writer. The YAML document stores
    references to the external array files.
    """

    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
        output_dir: Path,
        write_array: Callable,
        array_format: str,
    ):
        """Initialize the serializer.

        Parameters
        ----------
        graph
            Object graph to serialize.
        schemaview
            SchemaView describing the LinkML schema.
        output_dir
            Directory in which external array files are written.
        write_array
            Callable responsible for writing an array to disk.
        array_format
            Name of the external array format recorded in the YAML output.
        """
        self.graph = graph
        self.schemaview = schemaview
        self.output_dir = output_dir
        self.write_array = write_array
        self.array_format = array_format

        self.serialized: dict[UUID, dict[str, Any]] = {}

    def make_filename(self, node: GraphNode, slot) -> Path:
        """Construct an output filename for an array-valued attribute.

        Array filenames are derived from the identifier of the owning object
        when available. Otherwise, the filename is derived from the identifier
        of the parent object together with the intervening LinkML attribute.
        """
        if node.identifier is not None:
            return self.output_dir / f"{node.identifier}.{slot.name}"

        if node.incoming:
            edge = node.incoming[0]
            parent = self.graph[edge.parent]

            if parent.identifier is not None:
                return self.output_dir / f"{parent.identifier}.{edge.slot_name}.{slot.name}"

        raise ValueError(f"Cannot determine filename for {node.class_name}")

    def serialize(self):
        """Serialize the graph into a nested YAML-compatible dictionary.

        Nodes are serialized in dependency order so that referenced child
        objects are available before their parents are constructed.

        Returns
        -------
        dict
            Serialized representation of the root object.
        """
        for node in self.graph.dependency_order():
            self.serialized[node.key] = self.serialize_node(node)

        return self.serialized[self.graph.root]

    def serialize_node(self, node):
        """Serialize a single graph node.

        Array-valued attributes are replaced with external array references,
        object-valued attributes are replaced with serialized child mappings,
        and scalar attributes are copied directly.
        """
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
    """Serialize an ObjectGraph into an HDF5 hierarchy.

    Each graph node becomes an HDF5 group, scalar attributes become
    group attributes, array-valued attributes become datasets, and
    object-valued attributes become child groups.
    """

    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
        h5file: h5py.File,
    ):
        """Initialize the HDF5 serializer.

        Parameters
        ----------
        graph
            Object graph to serialize.
        schemaview
            SchemaView describing the LinkML schema.
        h5file
            Open HDF5 file that will receive the serialized model.
        """
        self.graph = graph
        self.schemaview = schemaview
        self.h5file = h5file

        self.groups: dict[UUID, h5py.Group] = {}

    def serialize(self):
        """Write the complete graph to the HDF5 file.

        Nodes are visited in hierarchy order so that parent groups are created
        before child groups.
        """
        root = self.graph[self.graph.root]
        self.groups[root.key] = self.h5file

        for node in self.graph.hierarchy_order():
            self.serialize_node(node)

    def serialize_node(self, node: GraphNode):
        """Serialize a single graph node as an HDF5 group."""
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
    """Serialize an ObjectGraph into a Zarr hierarchy.

    Each graph node becomes a Zarr group, scalar attributes become group
    attributes, array-valued attributes become Zarr arrays, and
    object-valued attributes become child groups.
    """

    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
        root: zarr.Group,
    ):
        """Initialize the Zarr serializer.

        Parameters
        ----------
        graph
            Object graph to serialize.
        schemaview
            SchemaView describing the LinkML schema.
        root
            Root Zarr group that will receive the serialized model.
        """
        self.graph = graph
        self.schemaview = schemaview
        self.root = root

        self.groups: dict[UUID, zarr.Group] = {}

    def serialize(self):
        """Write the complete graph to the Zarr hierarchy.

        Nodes are visited in hierarchy order so that parent groups are created
        before child groups.
        """
        root_node = self.graph[self.graph.root]
        self.groups[root_node.key] = self.root

        for node in self.graph.hierarchy_order():
            self.serialize_node(node)

    def serialize_node(self, node: GraphNode):
        """Serialize a single graph node as a Zarr group."""
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
    """Serialize an ObjectGraph into a nested YAML representation.

    All arrays remain embedded within the YAML document and object-valued
    LinkML attributes are represented as nested mappings.
    """

    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
    ):
        """Initialize the YAML serializer.

        Parameters
        ----------
        graph
            Object graph to serialize.
        schemaview
            SchemaView describing the LinkML schema.
        """
        self.graph = graph
        self.schemaview = schemaview
        self.serialized: dict[UUID, dict] = {}

    def serialize(self) -> dict:
        """Serialize the graph into a nested YAML-compatible dictionary.

        Nodes are serialized in dependency order so that child objects are
        available before their parents.

        Returns
        -------
        dict
            Serialized representation of the root object.
        """
        for node in self.graph.dependency_order():
            self.serialized[node.key] = self.serialize_node(node)
        assert self.graph.root is not None
        return self.serialized[self.graph.root]

    def serialize_node(self, node: GraphNode) -> dict:
        """Serialize a single graph node into a YAML-compatible mapping."""
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
