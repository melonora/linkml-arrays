"""Deserializer using backend independent ObjectGraph representation."""

import inspect
import sys

from linkml_runtime import SchemaView
from pydantic import BaseModel

from linkml_arrays.graph_utils.graph import GraphNode, ObjectGraph


class GraphDeserializer:
    """Instantiate a graph of LinkML objects."""

    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
        class_map: dict[str, type[BaseModel]],
    ):
        self.graph = graph
        self.schemaview = schemaview
        self.class_map = class_map

    @classmethod
    def from_target_class(
        cls,
        graph: ObjectGraph,
        schemaview: SchemaView,
        target_class: type[BaseModel],
    ) -> "GraphDeserializer":
        """Construct a deserializer using the module containing ``target_class``."""
        module = sys.modules[target_class.__module__]

        class_map = {
            name: obj
            for name, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, BaseModel)
        }

        return cls(
            graph=graph,
            schemaview=schemaview,
            class_map=class_map,
        )

    def deserialize(
        self,
        target_class: type[BaseModel],
    ) -> BaseModel:
        """Instantiate every node in the graph and return the root object."""
        for node in self.graph.dependency_order():
            self._instantiate(node)

        root = self.graph[self.graph.root]

        assert root.obj is not None

        if not isinstance(root.obj, target_class):
            raise TypeError(f"Expected {target_class.__name__}, " f"got {type(root.obj).__name__}")
        return root.obj

    def _instantiate(
        self,
        node: GraphNode,
    ) -> None:
        cls = self.class_map[node.class_name]
        kwargs = dict(node.values)

        for edge in node.outgoing:
            child = self.graph[edge.child]

            if child.obj is None:
                raise RuntimeError(f"{child.class_name} has not yet been instantiated.")

            if edge.multivalued:
                kwargs.setdefault(edge.slot_name, []).append(child.obj)
            else:
                kwargs[edge.slot_name] = child.obj

        node.obj = cls(**kwargs)
