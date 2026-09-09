from pathlib import Path
from typing import Type, Union

from linkml_runtime import SchemaView
from linkml_runtime.loaders.loader_root import Loader
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel
from xarray import open_datatree

from linkml_arrays.graph_utils.deserializers import GraphDeserializer
from linkml_arrays.graph_utils.graph import ObjectGraph


class XarrayZarrLoader(Loader):
    def load_any(self, source: str, **kwargs):
        return self.load(source, **kwargs)

    def loads(self, source: str, **kwargs):
        return self.load(source, **kwargs)

    def load(
        self,
        source: str,
        target_class: Type[Union[YAMLRoot, BaseModel]],
        schemaview: SchemaView,
        **kwargs,
    ):
        tree = open_datatree(
            Path(source),
            engine="zarr",
        )

        graph = ObjectGraph.from_xarray(
            source=tree,
            schemaview=schemaview,
            root_class=target_class.__name__,
        )

        return GraphDeserializer.from_target_class(
            graph=graph,
            schemaview=schemaview,
            target_class=target_class,
        ).deserialize(target_class)


class XarrayNetCDFLoader(Loader):
    def load_any(self, source: str, **kwargs):
        return self.load(source, **kwargs)

    def loads(self, source: str, **kwargs):
        return self.load(source, **kwargs)

    def load(
        self,
        source: str,
        target_class: Type[Union[YAMLRoot, BaseModel]],
        schemaview: SchemaView,
        **kwargs,
    ):
        tree = open_datatree(
            Path(source),
            engine="h5netcdf",
        )

        graph = ObjectGraph.from_xarray(
            source=tree,
            schemaview=schemaview,
            root_class=target_class.__name__,
        )

        return GraphDeserializer.from_target_class(
            graph=graph,
            schemaview=schemaview,
            target_class=target_class,
        ).deserialize(target_class)
