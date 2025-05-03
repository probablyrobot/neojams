#!/usr/bin/env python
# CREATED:2015-02-04 16:39:00 by Brian McFee <brian.mcfee@nyu.edu>
"""Schema tests"""

import json
import os

# Set environment variable to suppress namespace file warnings during tests
os.environ["NEOJAMS_SUPPRESS_WARNINGS"] = "1"

import pytest
from six.moves import reload_module

import neojams
from neojams import NamespaceError


@pytest.mark.parametrize("ns_key", ["pitch_hz", "beat"])
def test_schema_namespace(ns_key):
    # Get the schema
    schema = neojams.schema.namespace(ns_key)
    # Make sure it has the correct properties
    valid_keys = set(["time", "duration", "value", "confidence"])
    for key in schema["properties"]:
        assert key in valid_keys
    for key in ["time", "duration"]:
        assert key in schema["properties"]


@pytest.mark.parametrize("ns_key", ["DNE"])
def test_schema_namespace_exception(ns_key):
    with pytest.raises(NamespaceError):
        neojams.schema.namespace(ns_key)


@pytest.mark.parametrize("ns, dense", [("pitch_hz", True), ("beat", False)])
def test_schema_is_dense(ns, dense):
    assert dense == neojams.schema.is_dense(ns)


@pytest.mark.parametrize("ns", ["DNE"])
def test_schema_is_dense_exception(ns):
    with pytest.raises(NamespaceError):
        neojams.schema.is_dense(ns)


@pytest.fixture
def local_namespace():
    # Save the current value of NEOJAMS_SUPPRESS_WARNINGS
    suppress_warnings = os.environ.get("NEOJAMS_SUPPRESS_WARNINGS")
    
    # Set environment variables for the test
    os.environ["JAMS_SCHEMA_DIR"] = os.path.join("tests", "fixtures", "schema")
    # Temporarily unset the suppress warnings flag to test the warning logic
    if "NEOJAMS_SUPPRESS_WARNINGS" in os.environ:
        del os.environ["NEOJAMS_SUPPRESS_WARNINGS"]
    
    reload_module(neojams)

    # This one should pass
    yield "testing_tag_upper", True

    # Cleanup
    del os.environ["JAMS_SCHEMA_DIR"]
    # Restore the suppress warnings flag
    if suppress_warnings is not None:
        os.environ["NEOJAMS_SUPPRESS_WARNINGS"] = suppress_warnings
    
    reload_module(neojams)


def test_schema_local(local_namespace):
    ns_key, exists = local_namespace

    # Get the schema
    if exists:
        schema = neojams.schema.namespace(ns_key)

        # Make sure it has the correct properties
        valid_keys = set(["time", "duration", "value", "confidence"])
        for key in schema["properties"]:
            assert key in valid_keys

        for key in ["time", "duration"]:
            assert key in schema["properties"]
    else:
        with pytest.raises(NamespaceError):
            schema = neojams.schema.namespace(ns_key)


def test_schema_values_pass():
    values = neojams.schema.values("tag_gtzan")

    assert values == ["blues", "classical", "country", "disco", "hip-hop", "jazz", "metal", "pop", "reggae", "rock"]


def test_schema_values_missing():
    with pytest.raises(NamespaceError):
        neojams.schema.values("imaginary namespace")


def test_schema_values_notenum():
    with pytest.raises(NamespaceError):
        neojams.schema.values("chord_harte")


def test_schema_dtypes():
    for n in neojams.schema.__NAMESPACE__:
        neojams.schema.get_dtypes(n)


def test_schema_dtypes_badns():
    with pytest.raises(NamespaceError):
        neojams.schema.get_dtypes("unknown namespace")


def test_list_namespaces():
    neojams.schema.list_namespaces()
    
    
# Clean up environment variable after tests
def test_cleanup():
    # This test runs last to clean up the environment
    if "NEOJAMS_SUPPRESS_WARNINGS" in os.environ:
        del os.environ["NEOJAMS_SUPPRESS_WARNINGS"]
