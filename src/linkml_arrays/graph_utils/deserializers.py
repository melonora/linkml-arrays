"""Deserializer using backend independent ObjectGraph representation."""

import inspect
import sys

from linkml_runtime import SchemaView
from pydantic import BaseModel

from linkml_arrays.graph_utils.graph import GraphNode, ObjectGraph


class GraphDeserializer:
    """Instantiate a graph of LinkML objects from an object graph.

    The deserializer walks a dependency-ordered graph of nodes, creates
    corresponding Pydantic model instances representing the linkml model,
    resolves relationships between objects, and returns the instantiated
    root model.
    """

    def __init__(
        self,
        graph: ObjectGraph,
        schemaview: SchemaView,
        class_map: dict[str, type[BaseModel]],
    ):
        """Initialize a graph deserializer.

        Parameters
        ----------
        graph
            The object graph containing nodes and relationships to deserialize.
        schemaview
            The LinkML schema view associated with the graph.
        class_map
            Mapping from LinkML class names to their corresponding model classes.
        """
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
        """Construct a deserializer from a target class.

        The module containing ``target_class`` is inspected to discover all
        available model classes that are part of the ``target_class``. These
        classes are used to construct a mapping from LinkML class names to
        Pydantic model types.

        Parameters
        ----------
        graph
            The object graph to deserialize.
        schemaview
            The LinkML schema view associated with the graph.
        target_class
            The expected root object class. The module containing this class is
            searched for additional model classes.

        Returns
        -------
        GraphDeserializer
            A configured graph deserializer instance.
        """
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
        """Deserialize the graph into an instantiated object hierarchy.

        Nodes are instantiated according to their dependency order, ensuring
        that referenced child objects exist before parent objects are created.

        Parameters
        ----------
        target_class
            The expected type of the root model.

        Returns
        -------
        BaseModel
            The instantiated root model.

        Raises
        ------
        TypeError:
            If the instantiated root object does not match ``target_class``.
        """
        for node in self.graph.dependency_order():
            self._instantiate(node)

        root = self.graph[self.graph.root]

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
