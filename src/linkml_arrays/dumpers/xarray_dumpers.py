from pathlib import Path
from typing import Union, List

import numpy as np
from linkml_runtime import SchemaView
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

import xarray as xr
from linkml_arrays.dumpers.yaml_array_file_dumper import YamlArrayFileDumper
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
        ).serialize()

        datatree.to_zarr(to_file)

    def dumps(self, element: Union[YAMLRoot, BaseModel], **kwargs):
        raise NotImplementedError(
            "This method is not sensible for this dumper."
        )

class YamlXarrayNetCDFDumper(YamlArrayFileDumper):
    FILE_SUFFIX = "_xarray.nc"
    FORMAT = "xarray_netcdf"

    @classmethod
    def write_array(
        cls,
        array: Union[List, np.ndarray],
        output_file_path_no_suffix: Union[str, Path],
    ):
        if isinstance(output_file_path_no_suffix, str):
            output_file_path_no_suffix = Path(output_file_path_no_suffix)

        output_file_path = output_file_path_no_suffix.parent / (
            output_file_path_no_suffix.name + cls.FILE_SUFFIX
        )

        data_array = xr.DataArray(data=np.asarray(array))
        data_array.to_netcdf(
            output_file_path,
            engine="h5netcdf",
        )
        return output_file_path


class YamlXarrayZarrDumper(YamlArrayFileDumper):
    FILE_SUFFIX = "_xarray.zarr"
    FORMAT = "xarray_zarr"

    @classmethod
    def write_array(
        cls,
        array: Union[List, np.ndarray],
        output_file_path_no_suffix: Union[str, Path],
        mode: str = "w",
    ):
        if isinstance(output_file_path_no_suffix, str):
            output_file_path_no_suffix = Path(output_file_path_no_suffix)

        output_file_path = output_file_path_no_suffix.parent / (
            output_file_path_no_suffix.name + cls.FILE_SUFFIX
        )

        data_array = xr.DataArray(data=np.asarray(array))
        data_array.to_zarr(output_file_path, mode=mode)
        return output_file_path