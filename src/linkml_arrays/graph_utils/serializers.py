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
import xarray as xr
from xarray import DataTree
from linkml_runtime import SchemaView
from pydantic import BaseModel

from linkml_arrays.graph_utils.graph import GraphNode, ObjectGraph, GraphEdge


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

    def serialize(self, dir_path = None):
        """Serialize the graph into a nested YAML-compatible dictionary.

        Nodes are serialized in dependency order so that referenced child
        objects are available before their parents are constructed.

        Returns
        -------
        dict
            Serialized representation of the root object.
        """
        for node in self.graph.dependency_order():
            self.serialized[node.key] = self.serialize_node(node, dir_path)

        return self.serialized[self.graph.root]

    def serialize_node(self, node, dir_path):
        """Serialize a single graph node.

        Array-valued attributes are replaced with external array references,
        object-valued attributes are replaced with serialized child mappings,
        and scalar attributes are copied directly.
        """
        result = {}

        # TODO: discuss whether there is a way to also get rid of using the schema here, e.g.
        # can we purely use ObjectGraph?
        for slot_name, value in node.values.items():
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

                if filename.is_absolute():
                    yaml_file_value = output.as_posix()
                else:
                    yaml_file_value = f"./{output.as_posix()}"

                result[slot_name] = {
                    "source": [
                        {
                            "file": yaml_file_value,
                            "format": self.array_format,
                        }
                    ]
                }
            else:
                result[slot_name] = value

        for edge in node.outgoing:
            child = self.serialized[edge.child]

            if edge.multivalued:
                result.setdefault(edge.slot_name, []).append(child)
            else:
                result[edge.slot_name] = child

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

        for slot_name, value in node.values.items():
            slot = self.schemaview.induced_slot(
                slot_name,
                node.class_name,
            )

            if slot.array:
                group.create_dataset(
                    slot.name,
                    data=value,
                )
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

        for slot_name, value in node.values.items():
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
        result = dict(node.values)

        for edge in node.outgoing:
            child = self.serialized[edge.child]

            if edge.multivalued:
                result.setdefault(edge.slot_name, []).append(child)
            else:
                result[edge.slot_name] = child

        return result

class XarrayGraphSerializer:
    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
    ):
        self.graph = graph
        self.schemaview = schemaview

    def serialize(self) -> DataTree:
        if self.graph.root is None:
            raise ValueError("ObjectGraph has no root node.")

        root = self.graph[self.graph.root]
        return self._serialize_node(root)

    def _serialize_node(
        self,
        node: GraphNode,
    ) -> DataTree:
        attrs: dict[str, Any] = {}
        coords: dict[str, xr.DataArray] = {}
        data_vars: dict[str, xr.DataArray] = {}
        children: dict[str, DataTree] = {}

        for slot_name, value in node.values.items():
            slot = self.schemaview.induced_slot(
                slot_name,
                node.class_name,
            )

            if slot.array:
                data_vars[slot_name] = self._make_data_array(
                    node=node,
                    slot_name=slot_name,
                    value=value,
                )
            else:
                attrs[slot_name] = value

        for edge in node.outgoing:
            child = self.graph[edge.child]

            if self._is_array_node(child):
                array_slot_name, value = self._array_value(child)
                data = np.asarray(value)

                metadata = {
                    name: child_value
                    for name, child_value in child.values.items()
                    if name != array_slot_name
                }

                if data.ndim == 1:
                    if coords:
                        dims = (next(iter(coords)),)
                    else:
                        dims = (edge.slot_name,)

                    coords[edge.slot_name] = xr.DataArray(
                        data=data,
                        dims=dims,
                        attrs=metadata,
                    )

                else:
                    data_vars[edge.slot_name] = self._make_data_array(
                        node=child,
                        slot_name=array_slot_name,
                        value=data,
                        attrs=metadata,
                    )

            else:
                children[edge.slot_name] = self._serialize_node(child)

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
        )

        tree = DataTree(dataset=dataset)

        for name, child_tree in children.items():
            tree[name] = child_tree

        return tree

    def _is_array_node(
        self,
        node: GraphNode,
    ) -> bool:
        for slot_name in node.values:
            slot = self.schemaview.induced_slot(
                slot_name,
                node.class_name,
            )

            if slot.array:
                return True

        return False

    def _array_value(
        self,
        node: GraphNode,
    ) -> tuple[str, Any]:
        for slot_name, value in node.values.items():
            slot = self.schemaview.induced_slot(
                slot_name,
                node.class_name,
            )

            if slot.array:
                return slot_name, value

        raise ValueError(
            f"{node.class_name} does not contain an array-valued slot."
        )

    def _array_dims(
        self,
        node: GraphNode,
        slot_name: str,
        value: Any,
    ) -> tuple[str, ...]:
        slot = self.schemaview.induced_slot(
            slot_name,
            node.class_name,
        )

        data = np.asarray(value)

        dimensions = slot.array.dimensions or []

        dims = tuple(
            str(dimension.alias)
            for dimension in dimensions
            if dimension.alias is not None
        )

        if len(dims) == data.ndim:
            return dims

        return tuple(
            f"dim_{i}"
            for i in range(data.ndim)
        )

    def _make_data_array(
        self,
        node: GraphNode,
        slot_name: str,
        value: Any,
        attrs: dict[str, Any] | None = None,
    ) -> xr.DataArray:
        """Construct a DataArray for an array-valued graph value."""
        data = np.asarray(value)

        return xr.DataArray(
            data=data,
            dims=self._array_dims(
                node=node,
                slot_name=slot_name,
                value=data,
            ),
            attrs=attrs,
        )