"""Class for loading a LinkML model from a Zarr directory store."""

from typing import Type, Union

from linkml_runtime import SchemaView
from linkml_runtime.loaders.loader_root import Loader
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

from linkml_arrays.graph_utils.deserializers import GraphDeserializer
from linkml_arrays.graph_utils.graph import ObjectGraph


class ZarrDirectoryStoreLoader(Loader):
    """Class for loading a LinkML model from a Zarr directory store."""

    def load_any(self, source: str, **kwargs):
        """Create an instance of the target class from a Zarr directory store."""
        return self.load(source, **kwargs)

    def loads(self, source: str, **kwargs):
        """Create an instance of the target class from a Zarr directory store."""
        return self.load(source, **kwargs)

    def load(
        self,
        source: str,
        target_class: Type[Union[YAMLRoot, BaseModel]],
        schemaview: SchemaView,
        **kwargs,
    ):
        """Create an instance of the target class from a Zarr directory store."""
        graph = ObjectGraph.from_zarr(
            source,
            schemaview,
            target_class.__name__,
        )

        deserializer = GraphDeserializer.from_target_class(
            graph,
            schemaview,
            target_class,
        )

        return deserializer.deserialize(target_class)
