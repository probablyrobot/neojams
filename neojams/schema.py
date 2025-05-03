#!/usr/bin/env python
"""
NeoJAMS schema validation
=================

.. autosummary::
    :toctree: generated/

    is_valid
    validate
    schema_path
    schema
    values
    add_namespace
    list_namespaces
"""

import json
import os
import pprint
import re
import warnings
from collections import defaultdict

import jsonschema

try:
    from importlib import resources
except ImportError:
    # use backported importlib_resources from PyPI
    import importlib_resources as resources

from . import util
from .exceptions import NamespaceError, SchemaError

# Static variables
__NAMESPACE__ = defaultdict(list)
__SCHEMA__ = None

# We need these in order to import resources
__RESOURCE_SCHEMA_DIR = "schemata"
__RESOURCE_NAMESPACE_DIR = "schemata/namespaces"

# The top-level schema doesn't need to be JAMS_SCHEMA as long as namespaces match
NS_SCHEMA_DIR = "namespaces"

# Local schema names can include these prefixes and still be valid
NS_REGEX = r"^(namespace|.*jams)\-[a-zA-Z0-9_]+\.json$"

__all__ = [
    "is_valid",
    "validate",
    "schema_path",
    "JAMS_SCHEMA",
    "values",
    "add_namespace",
    "list_namespaces",
    "namespace",
    "namespace_array",
]


# Define namespace validation functions and store them
def _validate_time(value, **kwargs):
    if kwargs.get("duration", 0.0) < 0.0:
        return False
    return value >= 0


def _validate_confidence(value, **kwargs):
    return 0.0 <= value <= 1.0


def _validate_value(value, namespace, **kwargs):
    if namespace in __NAMESPACE__:
        namespace_schema = schema(namespace)
        if "enum" in namespace_schema["properties"]["value"]:
            return value in namespace_schema["properties"]["value"]["enum"]
    return True


# For legacy compatibility
VALIDATOR = None


def namespace_array(namespace: str) -> dict:
    """Get the schema for a namespace's array type.

    Parameters
    ----------
    namespace : str
        Namespace to get schema for

    Returns
    -------
    schema : dict
        Schema definition for the namespace's array type

    Raises
    ------
    NamespaceError
        If the namespace is not found
    """
    if namespace not in __NAMESPACE__:
        raise NamespaceError(f"Unknown namespace: {namespace}")

    schema_def = schema(namespace)

    # The namespace schema follows the pattern {"properties": { ... }}
    # Extract the schema for the `value` field.  Fall back to an empty schema
    # if it cannot be found (this mirrors legacy behaviour).
    value_schema = schema_def.get("properties", {}).get("value", {})

    # Build conformant observation schema
    return {
        "type": "object",
        "properties": {
            "time": {"type": "number", "minimum": 0},
            "duration": {"type": "number", "minimum": 0},
            "value": value_schema,
            # confidence is optional in many datasets; allow null in addition to number
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        },
        # No additional properties allowed by default
        "additionalProperties": False,
    }


def is_dense(namespace: str) -> bool:
    """Test if a namespace is dense.

    Parameters
    ----------
    namespace : str
        Namespace

    Returns
    -------
    dense : bool
        True if the namespace is time-dense, False if sparse

    Raises
    ------
    NamespaceError
        If the namespace is not found
    """
    if namespace not in __NAMESPACE__:
        raise NamespaceError(f"Unknown namespace: {namespace}")

    # Get the schema for this namespace
    schema_def = schema(namespace)

    # Check if the schema has a 'dense' property
    if "dense" in schema_def:
        return schema_def["dense"]

    # Default to sparse if not specified
    return False


def is_valid(obj, schema=None):
    """Validate a JAMS object against the schema.

    Parameters
    ----------
    obj : dict
        The JAMS object

    schema : dict or None
        Optionally, a schema definition in JSON-schema format
        If `None`, the schema will be inferred from the JAMS object.

    Returns
    -------
    valid : bool
        True if `obj` validates against `schema`.
        False otherwise.
    """

    try:
        if schema is None:
            schema = obj["json_schema"]

        jsonschema.validate(obj, schema)
        return True

    except (jsonschema.ValidationError, jsonschema.SchemaError, KeyError):
        return False


def validate(obj, schema=None):
    """Validate a JAMS object against the schema.

    Parameters
    ----------
    obj : dict
        The JAMS object

    schema : dict or None
        Optionally, a schema definition in JSON-schema format
        If `None`, the schema will be inferred from the JAMS object.

    Returns
    -------
    valid : bool
        True if `obj` is valid.

    Raises
    ------
    SchemaError
        If `obj` fails to validate.
    """

    valid = is_valid(obj, schema)

    if not valid:
        try:
            if schema is None:
                schema = obj["json_schema"]
            validator = jsonschema.validators.validator_for(schema)(schema)
            for e in validator.iter_errors(obj):
                raise SchemaError(f"{str(e.message):s}\n{str(e.schema_path):s}")
            raise SchemaError("Failed to validate: " f"{pprint.pformat(obj):s}")
        except (jsonschema.ValidationError, jsonschema.SchemaError, KeyError, TypeError) as e:
            raise SchemaError("Failed to validate: " f"{pprint.pformat(obj):s}") from None

    return valid


def schema_path(namespace):
    """Find the path to the schema for a given namespace.

    Parameters
    ----------
    namespace : str
        Namespace to find

    Returns
    -------
    schema_path : str
        Full path to the namespace schema definition.

    See Also
    --------
    schema

    Examples
    --------
    >>> jams.schema.schema_path('tag_open')    # doctest: +SKIP
    '/.../site-packages/jams/schemata/namespaces/tag/tag_open.json'
    """

    values = __NAMESPACE__.get(namespace, [])

    if not values:
        raise NamespaceError(f"Unknown namespace: {namespace:s}")

    return values[0]


def schema(namespace):
    """Return a copy of the schema for a given namespace.

    Parameters
    ----------
    namespace : str
        The namespace to fetch schema for

    Returns
    -------
    schema : dict
        The schema definition object for the namespace

    See Also
    --------
    schema_path

    Examples
    --------
    >>> tag_schema = jams.schema('tag_open')
    >>> tag_schema['properties'].keys()    # doctest: +SKIP
    [u'confidence', u'tag', u'id', u'value']
    >>> tag_schema['properties']['tag']['description']    # doctest: +SKIP
    u'The open vocabulary tag label'
    """

    with open(schema_path(namespace)) as fdesc:
        schema_def = json.load(fdesc)
        # The schema files use a different format where the namespace is the key
        # and the schema definition is the value
        return schema_def[namespace]


def values(namespace):
    """Return the allowed values for a given namespace (if any).

    Parameters
    ----------
    namespace : str
        Namespace to query

    Returns
    -------
    values : list or None
        If the namespace corresponds to a controlled vocabulary, then the list of
        allowed values is returned.
        Otherwise, `None` is returned.

    Raises
    ------
    NamespaceError
        If the namespace is not found or does not have an enum constraint

    Examples
    --------
    >>> jams.schema.values('tag_gtzan')    # doctest: +SKIP
    ['blues', 'classical', 'country', 'disco',
     'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    """

    schema_def = schema(namespace)

    if "enum" in schema_def.get("value", {}):
        return schema_def["value"]["enum"]

    raise NamespaceError(f"Namespace {namespace} does not have an enum constraint")


def add_namespace(filename):
    """Add a namespace definition to the schema.

    Parameters
    ----------
    filename : str
        Path to the json schema file for the namespace

    Notes
    -----
    This function only needs to be called once per namespace.
    Subsequent calls on the same namespace will overwrite the previous definition.

    May fail to load if the schema is incompatible.

    Examples
    --------
    >>> jams.schema.add_namespace(my_namespace_schema_file)   # doctest: +SKIP
    """

    def __load_namespace(filename):
        """Load a namespace schema file"""

        with open(filename) as fileobj:
            try:
                schema_def = json.load(fileobj)
            except ValueError:
                warnings.warn(f"Unable to load namespace from file: {filename}", stacklevel=2)
                return False

        # The schema files use a different format where the namespace is the key
        # and the schema definition is the value
        try:
            # Get the first (and only) key from the schema definition
            namespace = list(schema_def.keys())[0]
            __NAMESPACE__[namespace].append(filename)
            return True
        except (KeyError, IndexError):
            warnings.warn(f"Schema missing namespace: {filename}", stacklevel=2)
            return False

    if os.path.exists(filename):
        # Only warn about namespace file naming if not in test mode
        if not os.environ.get("NEOJAMS_SUPPRESS_WARNINGS") and not os.path.basename(filename).startswith("namespace-"):
            if not re.match(NS_REGEX, os.path.basename(filename), flags=re.IGNORECASE):
                warnings.warn(
                    'Namespace files should begin with "namespace-", ' f'"{os.path.basename(filename)}" does not',
                    stacklevel=2,
                )
        return __load_namespace(filename)
    return False


def list_namespaces():
    """Return a list of all known namespace identifiers.

    Returns
    -------
    namespaces : list
        All known namespace identifiers.

    Examples
    --------
    >>> jams.schema.list_namespaces()    # doctest: +SKIP
    ['chord', 'tag_gtzan', 'beat', ... ]
    """

    return list(__NAMESPACE__.keys())


def get_dtypes(namespace):
    """Get the expected datatypes for each field in a namespace

    Parameters
    ----------
    namespace : str
        The namespace to examine

    Returns
    -------
    dtypes : dict
        A dictionary mapping field names to datatype descriptors
    """
    if namespace not in __NAMESPACE__:
        raise NamespaceError(f"Unknown namespace: {namespace}")

    schema_def = schema(namespace)

    dtypes = {}

    # Handle the value field
    if "value" in schema_def:
        value_schema = schema_def["value"]
        if "oneOf" in value_schema:
            dtypes["value"] = [t["type"] for t in value_schema["oneOf"]]
        elif "type" in value_schema:
            dtypes["value"] = value_schema["type"]

    # Add standard fields
    dtypes["time"] = "number"
    dtypes["duration"] = "number"
    dtypes["confidence"] = ["number", "null"]

    return dtypes


def validate_annotation(annotation):
    """Validate an annotation object against its schema.

    Parameters
    ----------
    annotation : Annotation
        The annotation to validate

    Returns
    -------
    bool
        True if the annotation is valid, False otherwise

    Raises
    ------
    NamespaceError
        If the namespace is not registered
    SchemaError
        If the annotation contains invalid data
    """
    if annotation.namespace not in __NAMESPACE__:
        raise NamespaceError(f"Unknown namespace: {annotation.namespace}")

    # Get the schema for this namespace
    namespace_schema = schema(annotation.namespace)

    # Basic validation for observations
    for obs in annotation.data:
        if hasattr(obs, "time") and obs.time < 0:
            raise SchemaError(f"Observation has negative time: {obs.time}")

        if hasattr(obs, "duration") and obs.duration < 0:
            raise SchemaError(f"Observation has negative duration: {obs.duration}")

        if hasattr(obs, "confidence") and obs.confidence is not None and (obs.confidence < 0 or obs.confidence > 1):
            raise SchemaError(f"Observation has invalid confidence: {obs.confidence}")

        # Validate the value against the schema
        if hasattr(obs, "value"):
            try:
                value_schema = namespace_schema.get("value", {})

                # Check for enum constraints
                if "enum" in value_schema and obs.value not in value_schema["enum"]:
                    raise SchemaError(f"Value '{obs.value}' not in enum for namespace '{annotation.namespace}'")

                # Check for oneOf constraints
                if "oneOf" in value_schema:
                    valid = False
                    error_msgs = []

                    for option in value_schema["oneOf"]:
                        if "type" in option:
                            if option["type"] == "null" and obs.value is None:
                                valid = True
                                break
                            elif option["type"] == "string" and isinstance(obs.value, str):
                                valid = True
                                break
                            elif option["type"] == "number" and isinstance(obs.value, (int, float)):
                                valid = True
                                break
                            elif option["type"] == "integer" and isinstance(obs.value, int):
                                valid = True
                                break
                            elif option["type"] == "object" and isinstance(obs.value, dict):
                                valid = True
                                break
                            elif option["type"] == "array" and isinstance(obs.value, (list, tuple)):
                                valid = True
                                break
                            elif option["type"] == "boolean" and isinstance(obs.value, bool):
                                valid = True
                                break
                            else:
                                error_msgs.append(f"Expected {option['type']}, got {type(obs.value).__name__}")

                    if not valid:
                        raise SchemaError(
                            f"Value does not match any allowed types for namespace '{annotation.namespace}': {'; '.join(error_msgs)}"
                        )

                # Check for simple type constraints
                elif "type" in value_schema:
                    valid_types = value_schema["type"]
                    if isinstance(valid_types, str):
                        valid_types = [valid_types]

                    type_valid = False
                    for valid_type in valid_types:
                        if valid_type == "null" and obs.value is None:
                            type_valid = True
                            break
                        elif valid_type == "string" and isinstance(obs.value, str):
                            type_valid = True
                            break
                        elif valid_type == "number" and isinstance(obs.value, (int, float)):
                            type_valid = True
                            break
                        elif valid_type == "integer" and isinstance(obs.value, int):
                            type_valid = True
                            break
                        elif valid_type == "object" and isinstance(obs.value, dict):
                            type_valid = True
                            break
                        elif valid_type == "array" and isinstance(obs.value, (list, tuple)):
                            type_valid = True
                            break
                        elif valid_type == "boolean" and isinstance(obs.value, bool):
                            type_valid = True
                            break

                    if not type_valid:
                        raise SchemaError(
                            f"Expected {', '.join(valid_types)}, got {type(obs.value).__name__} for namespace '{annotation.namespace}'"
                        )

                # Additional validation for object types with properties
                if isinstance(obs.value, dict) and "properties" in value_schema:
                    properties = value_schema["properties"]

                    # Check required properties if specified
                    if "required" in value_schema:
                        for required_prop in value_schema["required"]:
                            if required_prop not in obs.value:
                                raise SchemaError(
                                    f"Missing required property '{required_prop}' in namespace '{annotation.namespace}'"
                                )

                    # Validate each property against its schema
                    for prop_name, prop_value in obs.value.items():
                        if prop_name in properties:
                            prop_schema = properties[prop_name]

                            # Check enum
                            if "enum" in prop_schema and prop_value not in prop_schema["enum"]:
                                raise SchemaError(
                                    f"Property '{prop_name}' value '{prop_value}' not in enum {prop_schema['enum']} for namespace '{annotation.namespace}'"
                                )

                            # Check type
                            if "type" in prop_schema:
                                prop_types = prop_schema["type"]
                                if isinstance(prop_types, str):
                                    prop_types = [prop_types]

                                type_valid = False
                                for prop_type in prop_types:
                                    if prop_type == "null" and prop_value is None:
                                        type_valid = True
                                        break
                                    elif prop_type == "string" and isinstance(prop_value, str):
                                        type_valid = True
                                        break
                                    elif prop_type == "number" and isinstance(prop_value, (int, float)):
                                        type_valid = True
                                        break
                                    elif prop_type == "integer" and isinstance(prop_value, int):
                                        type_valid = True
                                        break
                                    elif prop_type == "object" and isinstance(prop_value, dict):
                                        type_valid = True
                                        break
                                    elif prop_type == "array" and isinstance(prop_value, (list, tuple)):
                                        type_valid = True
                                        break
                                    elif prop_type == "boolean" and isinstance(prop_value, bool):
                                        type_valid = True
                                        break

                                if not type_valid:
                                    raise SchemaError(
                                        f"Property '{prop_name}' expected type {', '.join(prop_types)}, got {type(prop_value).__name__} for namespace '{annotation.namespace}'"
                                    )

                            # Check minimum value
                            if "minimum" in prop_schema and isinstance(prop_value, (int, float)):
                                min_value = prop_schema["minimum"]
                                is_exclusive = prop_schema.get("exclusiveMinimum", False)

                                if is_exclusive and prop_value <= min_value:
                                    raise SchemaError(
                                        f"Property '{prop_name}' must be greater than {min_value} for namespace '{annotation.namespace}'"
                                    )
                                elif not is_exclusive and prop_value < min_value:
                                    raise SchemaError(
                                        f"Property '{prop_name}' must be greater than or equal to {min_value} for namespace '{annotation.namespace}'"
                                    )

                            # Check maximum value
                            if "maximum" in prop_schema and isinstance(prop_value, (int, float)):
                                max_value = prop_schema["maximum"]
                                is_exclusive = prop_schema.get("exclusiveMaximum", False)

                                if is_exclusive and prop_value >= max_value:
                                    raise SchemaError(
                                        f"Property '{prop_name}' must be less than {max_value} for namespace '{annotation.namespace}'"
                                    )
                                elif not is_exclusive and prop_value > max_value:
                                    raise SchemaError(
                                        f"Property '{prop_name}' must be less than or equal to {max_value} for namespace '{annotation.namespace}'"
                                    )

            except SchemaError:
                raise
            except Exception as e:
                raise SchemaError(f"Invalid value for namespace '{annotation.namespace}': {str(e)}")

    return True


def _get_schema_paths():
    """Find the schema files"""

    if resources.is_resource("neojams", __RESOURCE_SCHEMA_DIR):
        # The package has been installed
        search_path = [resources.files("neojams").joinpath(__RESOURCE_SCHEMA_DIR)]
    else:
        # We're running from a source checkout
        search_path = []
        try:
            # Try to read namespaces from the resource
            search_path.append(resources.files("neojams").joinpath(__RESOURCE_SCHEMA_DIR))
        except (ModuleNotFoundError, TypeError):
            # Try to read from another location
            abs_schema_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), __RESOURCE_SCHEMA_DIR))
            search_path.append(abs_schema_dir)

    if "JAMS_SCHEMA_DIR" in os.environ:
        search_path.extend(os.environ["JAMS_SCHEMA_DIR"].split(":"))

    paths = []
    for spath in search_path:
        paths.extend(util.find_with_extension(os.path.join(spath, NS_SCHEMA_DIR), "json"))

    return paths


def _load_all_namespaces():
    """Find and load all namespace schema definitions."""

    for schema_file in _get_schema_paths():
        add_namespace(schema_file)


def __load_jams_schema():
    """Load the jams schema file"""

    # Try to read from the resource bundle first
    try:
        schema_path = resources.files("neojams") / "schemata" / "jams_schema.json"
        with open(schema_path) as fdesc:
            jams_schema = json.load(fdesc)
    except (FileNotFoundError, ModuleNotFoundError, ValueError):
        abs_schema_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), __RESOURCE_SCHEMA_DIR))
        schema_file = os.path.join(abs_schema_dir, "jams_schema.json")
        with open(schema_file) as fdesc:
            jams_schema = json.load(fdesc)

    if jams_schema is None:
        warnings.warn("Unable to locate JAMS schema. " "Validation will not be available.", stacklevel=2)

    return jams_schema


# Create the global schema mapping object
_load_all_namespaces()
JAMS_SCHEMA = __load_jams_schema()
VALIDATOR = jsonschema.validators.Draft4Validator(JAMS_SCHEMA)


def namespace(namespace: str) -> dict:
    """Alias for namespace_array for backward compatibility."""
    return namespace_array(namespace)
