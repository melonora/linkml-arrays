from typing import Union

from linkml_runtime import SchemaView
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

from linkml_arrays.graph_utils.graph import ObjectGraph
from linkml_runtime.dumpers.dumper_root import Dumper
from linkml_arrays.graph_utils.serializers import XarrayGraphSerializer


class XarrayNetCDFDumper(Dumper):
    """Dump a LinkML model to NetCDF through an ObjectGraph."""

    def dump(
        self,
        element: Union[YAMLRoot, BaseModel],
        to_file: str,
        schemaview: SchemaView,
        **kwargs,
    ):
        graph = ObjectGraph.from_root(
            element,
            schemaview,
        )

        datatree = XarrayGraphSerializer(
            graph=graph,
            schemaview=schemaview,
        ).serialize()

        datatree.to_netcdf(
            to_file,
            engine="h5netcdf",
        )

    def dumps(self, element: Union[YAMLRoot, BaseModel], **kwargs):
        raise NotImplementedError(
            "This method is not sensible for this dumper."
        )

class XarrayZarrDumper(Dumper):
    """Dump a LinkML model to Zarr through an ObjectGraph."""

    def dump(
        self,
        element: Union[YAMLRoot, BaseModel],
        to_file: str,
        schemaview: SchemaView,
        **kwargs,
    ):
        graph = ObjectGraph.from_root(
            element,
            schemaview,
        )

        datatree = XarrayGraphSerializer(
            graph=graph,
            schemaview=schemaview,
        ).serialize()

        datatree.to_zarr(to_file)

    def dumps(self, element: Union[YAMLRoot, BaseModel], **kwargs):
        raise NotImplementedError(
            "This method is not sensible for this dumper."
        )