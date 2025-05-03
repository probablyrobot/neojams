#!/usr/bin/env python
"""
Compatibility utilities for NeoJAMS

This module provides compatibility tools for Python 3.12+ support
and migration from legacy code.
"""

import inspect

# String type definition
string_types = (str,)


# Dictionary helpers
def iteritems(d):
    """Return an iterator over dictionary (key, value) pairs."""
    # Handle case where d is a method (like __json__) that returns a dictionary
    if callable(d) and not hasattr(d, "items"):
        return d().items()
    # Handle regular dictionary case
    return d.items()


def itervalues(d):
    """Return an iterator over dictionary values."""
    return d.values()


# Function code inspection
def get_function_code(func):
    """Get the code object of a function."""
    return inspect.getfullargspec(func)


# Safe isinstance replacement (Python 3.10+ supports X | Y syntax)
def safe_isinstance(obj, classes):
    """Safely check isinstance with multiple types.

    This function handles both the older tuple approach and the newer
    union (|) syntax in Python 3.10+.

    Parameters
    ----------
    obj : object
        The object to check
    classes : type or tuple of types or union of types
        The type(s) to check against

    Returns
    -------
    bool
        True if obj is an instance of classes
    """
    # Always use isinstance directly, as Python 3.12+ handles union types correctly
    return isinstance(obj, classes)


# Helper for super() calls
def super_call(cls=None, instance=None):
    """Call super() in a way that works in Python 3.12+."""
    if cls is None and instance is None:
        return super()
    return super(cls, instance)


# Python 3-native zip
moves_zip = zip

# Python 3-native callable
callable = callable
