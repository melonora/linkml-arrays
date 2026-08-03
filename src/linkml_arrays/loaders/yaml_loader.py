"""Class for loading a LinkML model from a YAML file."""

from typing import Type, Union

from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition
from linkml_runtime.loaders.loader_root import Loader
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

from linkml_arrays.graph_utils.deserializers import GraphDeserializer
from linkml_arrays.graph_utils.graph import ObjectGraph


def _iterate_element(
    input_dict: dict, element_type: ClassDefinition, schemaview: SchemaView
) -> dict:
    """Recursively iterate through the elements of a LinkML model and load them into a dict."""
    ret_dict = dict()
    for k, v in input_dict.items():
        found_slot = schemaview.induced_slot(k, element_type.name)
        if isinstance(v, dict):
            found_slot_range = schemaview.get_class(found_slot.range)
            v = _iterate_element(v, found_slot_range, schemaview)
        # else: do not transform v
        ret_dict[k] = v

    return ret_dict


class YamlLoader(Loader):
    """Class for loading a LinkML model from a YAML file."""

    def load_any(self, source: str, **kwargs):
        """Create an instance of the target class from a YAML file."""
        return self.load(source, **kwargs)

    def loads(self, source: str, **kwargs):
        """Create an instance of the target class from a YAML file."""
        return self.load(source, **kwargs)

    def load(
        self,
        source: str,
        target_class: Type[Union[YAMLRoot, BaseModel]],
        schemaview: SchemaView,
        resolve_arrays: bool = False,
        **kwargs,
    ):
        """Create an instance of the target class from a YAML file."""
        graph = ObjectGraph.from_yaml(
            source,
            schemaview,
            target_class.__name__,
            resolve_arrays=resolve_arrays,
        )

        deserializer = GraphDeserializer.from_target_class(
            graph,
            schemaview,
            target_class,
        )

        return deserializer.deserialize(target_class)
