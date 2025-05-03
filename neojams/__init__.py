#!/usr/bin/env python
"""Top-level module for NeoJAMS"""

import glob
import os
import sys
import warnings
from importlib import resources
from itertools import chain

# Import the necessary modules
from . import display, eval, schema, sonify, util
from .core import (
    JAMS,
    Annotation,
    AnnotationArray,
    AnnotationMetadata,
    Curator,
    FileMetadata,
    JObject,
    Observation,
    Sandbox,
    load,
)
from .exceptions import JamsError, NamespaceError, ParameterError, SchemaError

# Import the Pydantic models
from .models import JAMS as JAMSModel
from .models import Annotation as AnnotationModel
from .models import AnnotationMetadata as AnnotationMetadataModel
from .models import Curator as CuratorModel
from .models import FileMetadata as FileMetadataModel
from .models import Observation as ObservationModel
from .models import Sandbox as SandboxModel
from .nsconvert import convert
from .schema import list_namespaces
from .version import JAMS_VERSION, __version__

# Populate the namespace mapping
try:
    # Set suppress warnings for initial loading if in test mode
    initial_suppress = os.environ.get("NEOJAMS_SUPPRESS_WARNINGS")
    if initial_suppress is None and "pytest" in sys.modules:
        os.environ["NEOJAMS_SUPPRESS_WARNINGS"] = "1"

    # Modern importlib.resources approach
    for ns in (p.rglob("*.json") for p in resources.files("neojams.schemata.namespaces").iterdir()):
        for file_path in ns:
            schema.add_namespace(file_path)
except (ModuleNotFoundError, TypeError):
    # Fallback for direct file access
    ns_dir = os.path.join(os.path.dirname(__file__), "schemata", schema.NS_SCHEMA_DIR)
    for ns_file in util.find_with_extension(ns_dir, "json"):
        schema.add_namespace(ns_file)
finally:
    # Restore the original suppress warnings setting
    if initial_suppress is None and "NEOJAMS_SUPPRESS_WARNINGS" in os.environ:
        del os.environ["NEOJAMS_SUPPRESS_WARNINGS"]

# Populate local namespaces
if "JAMS_SCHEMA_DIR" in os.environ:
    for ns in util.find_with_extension(os.environ["JAMS_SCHEMA_DIR"], "json"):
        schema.add_namespace(ns)

# Add all namespace files to the schema
for file_path in glob.glob(os.path.join(os.path.dirname(__file__), "schemata", "namespaces", "*.json")):
    schema.add_namespace(file_path)

__all__ = [
    "JAMS",
    "JObject",
    "Annotation",
    "AnnotationArray",
    "AnnotationMetadata",
    "Curator",
    "FileMetadata",
    "Observation",
    "Sandbox",
    "load",
    "JamsError",
    "NamespaceError",
    "ParameterError",
    "SchemaError",
    "convert",
    "list_namespaces",
    "__version__",
    "JAMS_VERSION",
    "display",
    # Pydantic models
    "JAMSModel",
    "AnnotationModel",
    "AnnotationMetadataModel",
    "CuratorModel",
    "FileMetadataModel",
    "ObservationModel",
    "SandboxModel",
]
