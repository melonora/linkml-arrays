from __future__ import annotations

import re
import sys
from datetime import (
    date,
    datetime,
    time
)
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Union
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    conlist,
    field_validator,
    model_serializer
)


metamodel_version = "1.11.0"
version = "None"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias = True,
        validate_by_name = True,
        validate_assignment = True,
        validate_default = True,
        extra = "forbid",
        arbitrary_types_allowed = True,
        use_enum_values = True,
        strict = False,
    )





class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key:str):
        return getattr(self.root, key)

    def __getitem__(self, key:str):
        return self.root[key]

    def __setitem__(self, key:str, value):
        self.root[key] = value

    def __contains__(self, key:str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta({'default_prefix': 'example',
     'description': 'Example LinkML schema to demonstrate a 3D DataArray of '
                    'temperature values with labeled axes\n'
                    'using classes containing arrays for the axes and data instead '
                    'of using array slots/attributes.\n'
                    'Creating separate types for the array slots enables reuse and '
                    'extension.',
     'id': 'https://example.org/arrays',
     'imports': ['linkml:types'],
     'license': 'MIT',
     'name': 'arrays-temperature-example-2',
     'prefixes': {'example': {'prefix_prefix': 'example',
                              'prefix_reference': 'https://example.org/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'wgs84': {'prefix_prefix': 'wgs84',
                            'prefix_reference': 'http://www.w3.org/2003/01/geo/wgs84_pos#'}},
     'source_file': 'tests/input/temperature_schema.yaml',
     'title': 'Array Temperature Example Using NDArray Classes'} )


class Container(ConfiguredBaseModel):
    """
    A container for a temperature dataset
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://example.org/arrays', 'tree_root': True})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Container',
                       'TemperatureDataset',
                       'LatitudeInDegSeries',
                       'LongitudeInDegSeries']} })
    temperature_dataset: TemperatureDataset = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Container']} })
    latitude_series: LatitudeInDegSeries = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Container']} })
    longitude_series: LongitudeInDegSeries = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Container']} })


class TemperatureDataset(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://example.org/arrays',
         'implements': ['linkml:DataArray'],
         'tree_root': True})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Container',
                       'TemperatureDataset',
                       'LatitudeInDegSeries',
                       'LongitudeInDegSeries']} })
    latitude_in_deg: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['TemperatureDataset']} })
    longitude_in_deg: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['TemperatureDataset']} })
    date: DateSeries = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['TemperatureDataset']} })
    day_in_d: Optional[DaysInDSinceSeries] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['TemperatureDataset']} })
    temperatures_in_K: TemperaturesInKMatrix = Field(default=..., json_schema_extra = { "linkml_meta": {'annotations': {'labeled_by': {'tag': 'labeled_by',
                                        'value': [{'alias': 'lat',
                                                   'labeled_dimensions': [0, 1],
                                                   'labeling_slot': 'latitude_in_deg'},
                                                  {'alias': 'lon',
                                                   'labeled_dimensions': [0, 1],
                                                   'labeling_slot': 'longitude_in_deg'},
                                                  {'alias': 'date',
                                                   'labeled_dimensions': [2],
                                                   'labeling_slot': 'date'},
                                                  {'alias': 'day',
                                                   'labeled_dimensions': [2],
                                                   'labeling_slot': 'day_in_d'}]}},
         'domain_of': ['TemperatureDataset']} })


class LatitudeInDegSeries(ConfiguredBaseModel):
    """
    A 2D array whose values represent latitude
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://example.org/arrays'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Container',
                       'TemperatureDataset',
                       'LatitudeInDegSeries',
                       'LongitudeInDegSeries']} })
    values: list[list[float]] = Field(default=..., json_schema_extra = { "linkml_meta": {'array': {'exact_number_dimensions': 2},
         'domain_of': ['LatitudeInDegSeries',
                       'LongitudeInDegSeries',
                       'DateSeries',
                       'DaysInDSinceSeries',
                       'TemperaturesInKMatrix'],
         'unit': {'ucum_code': 'deg'}} })


class LongitudeInDegSeries(ConfiguredBaseModel):
    """
    A 2D array whose values represent longitude
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://example.org/arrays'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Container',
                       'TemperatureDataset',
                       'LatitudeInDegSeries',
                       'LongitudeInDegSeries']} })
    values: list[list[float]] = Field(default=..., json_schema_extra = { "linkml_meta": {'array': {'exact_number_dimensions': 2},
         'domain_of': ['LatitudeInDegSeries',
                       'LongitudeInDegSeries',
                       'DateSeries',
                       'DaysInDSinceSeries',
                       'TemperaturesInKMatrix'],
         'unit': {'ucum_code': 'deg'}} })


class DateSeries(ConfiguredBaseModel):
    """
    A 1D series of dates
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://example.org/arrays'})

    values: list[str] = Field(default=..., json_schema_extra = { "linkml_meta": {'array': {'exact_number_dimensions': 1},
         'domain_of': ['LatitudeInDegSeries',
                       'LongitudeInDegSeries',
                       'DateSeries',
                       'DaysInDSinceSeries',
                       'TemperaturesInKMatrix']} })


class DaysInDSinceSeries(ConfiguredBaseModel):
    """
    A 1D series whose values represent the number of days since a reference date
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://example.org/arrays'})

    values: list[int] = Field(default=..., json_schema_extra = { "linkml_meta": {'array': {'exact_number_dimensions': 1},
         'domain_of': ['LatitudeInDegSeries',
                       'LongitudeInDegSeries',
                       'DateSeries',
                       'DaysInDSinceSeries',
                       'TemperaturesInKMatrix'],
         'unit': {'ucum_code': 'd'}} })
    reference_date: str = Field(default=..., description="""The reference date for the `day_in_d` values""", json_schema_extra = { "linkml_meta": {'domain_of': ['DaysInDSinceSeries']} })


class TemperaturesInKMatrix(ConfiguredBaseModel):
    """
    A 3D array of temperatures
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://example.org/arrays'})

    conversion_factor: Optional[float] = Field(default=None, description="""A conversion factor to apply to the temperature values""", json_schema_extra = { "linkml_meta": {'domain_of': ['TemperaturesInKMatrix'], 'unit': {'ucum_code': 'K'}} })
    values: list[list[list[float]]] = Field(default=..., json_schema_extra = { "linkml_meta": {'array': {'dimensions': [{'alias': 'x'}, {'alias': 'y'}, {'alias': 'date'}],
                   'exact_number_dimensions': 3},
         'domain_of': ['LatitudeInDegSeries',
                       'LongitudeInDegSeries',
                       'DateSeries',
                       'DaysInDSinceSeries',
                       'TemperaturesInKMatrix'],
         'unit': {'ucum_code': 'K'}} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Container.model_rebuild()
TemperatureDataset.model_rebuild()
LatitudeInDegSeries.model_rebuild()
LongitudeInDegSeries.model_rebuild()
DateSeries.model_rebuild()
DaysInDSinceSeries.model_rebuild()
TemperaturesInKMatrix.model_rebuild()
