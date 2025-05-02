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
from typing import Any, Dict, List, Optional, Tuple, Union

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
NS_REGEX = r"^(namespace|.*jams)\-[a-z]+.json$"

__all__ = ["is_valid", "validate", "schema_path", "JAMS_SCHEMA", "values", "add_namespace", "list_namespaces"]

# For legacy compatibility
VALIDATOR = None
namespace_array = {}


def is_dense(namespace: str) -> bool:
    """Test if a namespace is dense.

    This is stub for backward compatibility.

    Parameters
    ----------
    namespace : str
        Namespace

    Returns
    -------
    dense : bool
        True if the namespace is time-dense

    Raises
    ------
    NamespaceError
        If the namespace is not found
    """
    if namespace in __NAMESPACE__:
        return True
    raise NamespaceError("Unknown namespace: {}".format(namespace))


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
        raise SchemaError(f"Unknown namespace: {namespace:s}")

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
    >>> tag_schema = jams.schema.schema('tag_open')
    >>> tag_schema['properties'].keys()    # doctest: +SKIP
    [u'confidence', u'tag', u'id', u'value']
    >>> tag_schema['properties']['tag']['description']    # doctest: +SKIP
    u'The open vocabulary tag label'
    """

    with open(schema_path(namespace)) as fdesc:
        return json.load(fdesc)


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

    Examples
    --------
    >>> jams.schema.values('tag_gtzan')    # doctest: +SKIP
    ['blues', 'classical', 'country', 'disco',
     'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    """

    schema_def = schema(namespace)

    if "enum" in schema_def["properties"]["value"]:
        return schema_def["properties"]["value"]["enum"]

    return None


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

        # Store schema by its namespace
        try:
            namespace = schema_def["namespace"]
            __NAMESPACE__[namespace].append(filename)
            return True
        except KeyError:
            warnings.warn(f"Schema missing namespace: {filename}", stacklevel=2)
            return False

    if os.path.exists(filename):
        if not os.path.basename(filename).startswith("namespace-"):
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
