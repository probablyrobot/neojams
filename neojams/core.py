"""
Core functionality
------------------

This library provides an interface for reading JAMS into Python, or creating
them programatically.

.. currentmodule:: neojams

Function reference
^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/

    load

Object reference
^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated/
    :template: class.rst

    JAMS
    FileMetadata
    AnnotationArray
    AnnotationMetadata
    Curator
    Annotation
    Observation
    Sandbox
    JObject
    Observation
"""

import contextlib
import gzip
import json
import os
import re
import warnings

import jsonschema
import numpy as np
import pandas as pd
from decorator import decorator
from sortedcontainers import SortedKeyList

from . import schema
from .compatibility import iteritems, string_types
from .exceptions import JamsError, ParameterError, SchemaError
from .models import Observation
from .version import JAMS_VERSION


# -- Helper functions -- #
def serialize_obj(obj):
    """Custom serialization functionality for working with advanced data types.
    - numpy arrays are converted to lists
    - lists are recursively serialized element-wise
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [serialize_obj(x) for x in obj]
    elif isinstance(obj, Observation):
        return {k: serialize_obj(v) for k, v in obj.model_dump().items()}
    return obj


def summary(obj, indent=0):
    """Helper function to format repr strings for JObjects and friends."""
    if hasattr(obj, "__summary__"):
        rep = obj.__summary__()
    elif isinstance(obj, SortedKeyList):
        rep = f"<{len(obj):d} observations>"
    else:
        rep = repr(obj)
    return rep.replace("\n", "\n" + " " * indent)


def summary_html(obj):
    if hasattr(obj, "_repr_html_"):
        return obj._repr_html_()
    elif isinstance(obj, dict):
        out = '<table class="table"><tbody>'
        for key in obj:
            out += rf""" <tr>
                            <th scope="row">{key}</th>
                            <td>{summary_html(obj[key])}</td>
                        </tr>"""
        out += "</tbody></table>"
        return out
    elif isinstance(obj, list):
        return "".join([summary_html(x) for x in obj])
    else:
        return str(obj)


def match_query(string, query):
    """Test if a string matches a query."""
    if callable(query):
        return query(string)
    elif isinstance(query, str) and isinstance(string, str):
        return re.match(query, string) is not None
    else:
        return query == string


__all__ = [
    "load",
    "JObject",
    "Sandbox",
    "Annotation",
    "Curator",
    "AnnotationMetadata",
    "FileMetadata",
    "AnnotationArray",
    "JAMS",
    "Observation",
]


def deprecated(version, version_removed):
    """This is a decorator which can be used to mark functions
    as deprecated.

    It will result in a warning being emitted when the function is used."""

    def __wrapper(func, *args, **kwargs):
        """Warn the user, and then proceed."""
        warnings.warn(
            f"{func.__module__:s}.{func.__name__:s}\n\tDeprecated as of JAMS version {version:s}."
            f"\n\tIt will be removed in JAMS version {version_removed:s}.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)


@contextlib.contextmanager
def _open(name_or_fdesc, mode="r", fmt="auto"):
    """An intelligent wrapper for ``open``.

    Parameters
    ----------
    name_or_fdesc : string-type or open file descriptor
        If a string type, refers to the path to a file on disk.

        If an open file descriptor, it is returned as-is.

    mode : string
        The mode with which to open the file.
        See ``open`` for details.

    fmt : string ['auto', 'jams', 'json', 'jamz']
        The encoding for the input/output stream.

        If `auto`, the format is inferred from the filename extension.

        Otherwise, use the specified coding.


    See Also
    --------
    open
    gzip.open
    """

    open_map = {"jams": open, "json": open, "jamz": gzip.open, "gz": gzip.open}

    # If we've been given an open descriptor, do the right thing
    if hasattr(name_or_fdesc, "read") or hasattr(name_or_fdesc, "write"):
        yield name_or_fdesc

    elif isinstance(name_or_fdesc, string_types):
        # Infer the opener from the extension

        if fmt == "auto":
            _, ext = os.path.splitext(name_or_fdesc)

            # Pull off the extension separator
            ext = ext[1:]
        else:
            ext = fmt

        try:
            ext = ext.lower()

            # Force text mode if we're using gzip
            if ext in ["jamz", "gz"] and "t" not in mode:
                mode = f"{mode:s}t"

            with open_map[ext](name_or_fdesc, mode=mode) as fdesc:
                yield fdesc

        except KeyError:
            raise ParameterError("Unknown JAMS extension " f'format: "{ext:s}"') from None

    else:
        # Don't know how to handle this. Raise a parameter error
        raise ParameterError("Invalid filename or " f"descriptor: {name_or_fdesc}")


def load(path_or_file, validate=True, strict=True, fmt="auto"):
    r"""Load a JAMS Annotation from a file.


    Parameters
    ----------
    path_or_file : str or file-like
        Path to the JAMS file to load
        OR
        An open file handle to load from.

    validate : bool
        Attempt to validate the JAMS object

    strict : bool
        if `validate == True`, enforce strict schema validation

    fmt : str ['auto', 'jams', 'jamz']
        The encoding format of the input

        If `auto`, encoding is inferred from the file name.

        If the input is an open file handle, `jams` encoding
        is used.


    Returns
    -------
    jam : JAMS
        The loaded JAMS object


    Raises
    ------
    SchemaError
        if `validate == True`, `strict==True`, and validation fails


    See also
    --------
    JAMS.validate
    JAMS.save


    Examples
    --------
    >>> # Load a jams object from a file name
    >>> J = neojams.load('data.jams')
    >>> # Or from an open file descriptor
    >>> with open('data.jams', 'r') as fdesc:
    ...     J = neojams.load(fdesc)
    >>> # Non-strict validation
    >>> J = neojams.load('data.jams', strict=False)
    >>> # No validation at all
    >>> J = neojams.load('data.jams', validate=False)
    """

    with _open(path_or_file, mode="r", fmt=fmt) as fdesc:
        jam = JAMS(**json.load(fdesc))

    if validate:
        jam.validate(strict=strict)

    return jam


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class JObject:
    """Base class for all JAMS objects."""

    def __init__(self, **kwargs):
        """Create a JObject.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to store in the object.
        """
        super().__init__()
        for key, value in kwargs.items():
            if key != "_data":
                setattr(self, key, value)

    @property
    def __schema__(self):
        """Return the JSON schema for this object."""
        return schema.JAMS_SCHEMA

    def __json__(self):
        """Return a JSON representation of this object."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __json_light__(self, data=True):
        """Return a lightweight JSON representation of this object."""
        return self.__json__()

    @classmethod
    def __json_init__(cls, **kwargs):
        """Create a new object from JSON data."""
        return cls(**kwargs)

    def __eq__(self, other):
        """Compare two objects for equality."""
        if not isinstance(other, type(self)):
            return False
        return self.__json__() == other.__json__()

    def __nonzero__(self):
        """Return True if the object is non-empty."""
        return bool(self.__json__())

    def __getitem__(self, key):
        """Get an item by key."""
        return self.__dict__[key]

    def __setattr__(self, name, value):
        """Set an attribute by name."""
        self.__dict__[name] = value

    def __contains__(self, key):
        """Check if a key exists."""
        return key in self.__dict__

    def __len__(self):
        """Return the number of attributes."""
        return len(self.__json__())

    def __repr__(self):
        """Return a string representation of this object."""
        items = [f"{k}={v}" for k, v in sorted(self.__json__().items())]
        if items:
            return f"<{self.__class__.__name__}({',\n      '.join(items)})>"
        else:
            return f"<{self.__class__.__name__}()>"

    def _display_properties(self):
        """Return a list of (property, label) pairs for display."""
        return []

    def _repr_html_(self):
        """Return an HTML representation of this object."""
        return summary_html(self)

    def __summary__(self):
        """Return a summary string for this object."""
        return summary(self)

    def __str__(self):
        """Return a string representation of this object."""
        return self.__repr__()

    def dumps(self, **kwargs):
        """Return a JSON string representation of this object."""
        kwargs.setdefault("cls", NumpyJSONEncoder)
        return json.dumps(self.__json__(), **kwargs)

    def keys(self):
        """Return a list of attribute names."""
        return self.__json__().keys()

    def update(self, **kwargs):
        """Update attributes from keyword arguments."""
        for key, value in kwargs.items():
            if key != "_data":
                setattr(self, key, value)

    @property
    def type(self):
        """Return the type of this object."""
        return self.__class__.__name__

    @classmethod
    def loads(cls, string):
        """Create a new object from a JSON string."""
        return cls.__json_init__(**json.loads(string))

    def search(self, **kwargs):
        """Search for attributes matching the given criteria."""
        results = []
        for key, value in kwargs.items():
            if key in self.__dict__:
                if callable(value):
                    if value(self.__dict__[key]):
                        results.append(self)
                elif isinstance(value, string_types):
                    if value in str(self.__dict__[key]):
                        results.append(self)
                else:
                    if self.__dict__[key] == value:
                        results.append(self)
        return results

    def validate(self, strict=True):
        """Validate this object against its schema."""
        valid = True
        try:
            schema.VALIDATOR.validate(self.__json_light__(), self.__schema__)
        except jsonschema.ValidationError as invalid:
            if strict:
                raise SchemaError(str(invalid)) from None
            else:
                warnings.warn(str(invalid), stacklevel=2)
                valid = False
        return valid


class Sandbox(JObject):
    """Sandbox

    Container object for arbitrary metadata.
    """

    def __init__(self, **kwargs):
        """Create a Sandbox.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to store in the sandbox.
        """
        super().__init__()
        self._data = {}
        for key, value in kwargs.items():
            self[key] = value

    def __getitem__(self, key):
        """Get an item by key."""
        return self._data[key]

    def __setitem__(self, key, value):
        """Set an item by key."""
        self._data[key] = value

    def __delitem__(self, key):
        """Delete an item by key."""
        del self._data[key]

    def __contains__(self, key):
        """Check if a key exists."""
        return key in self._data

    def __len__(self):
        """Return the number of items."""
        return len(self._data)

    def __iter__(self):
        """Return an iterator over the items."""
        return iter(self._data)

    def __repr__(self):
        """Return a string representation of this object."""
        return f"<Sandbox({self._data})>"

    def __str__(self):
        """Return a string representation of this object."""
        return str(self._data)

    def __eq__(self, other):
        """Compare two objects for equality."""
        if not isinstance(other, type(self)):
            return False
        return self._data == other._data

    def __json__(self):
        """Return a JSON representation of this object."""
        return self._data

    def __json_light__(self, data=True):
        """Return a lightweight JSON representation of this object."""
        return self._data

    def keys(self):
        """Return a list of keys."""
        return self._data.keys()

    def values(self):
        """Return a list of values."""
        return self._data.values()

    def items(self):
        """Return a list of (key, value) pairs."""
        return self._data.items()

    def update(self, **kwargs):
        """Update the sandbox with new key-value pairs."""
        self._data.update(kwargs)


class Annotation(JObject):
    """Annotation base class."""

    def __init__(self, namespace, data=None, annotation_metadata=None, sandbox=None, time=0, duration=None):
        """Create an Annotation.

        Parameters
        ----------
        namespace : str
            The namespace for this annotation.
        data : list or None
            List of observation dicts.
        annotation_metadata : AnnotationMetadata or None
            Metadata for this annotation.
        sandbox : Sandbox or None
            Sandbox data for this annotation.
        time : float
            The time offset of this annotation.
        duration : float or None
            The duration of this annotation.
        """
        super().__init__()

        self.namespace = namespace
        self.time = time
        self.duration = duration

        if annotation_metadata is None:
            annotation_metadata = AnnotationMetadata()
        self.annotation_metadata = annotation_metadata

        if sandbox is None:
            sandbox = Sandbox()
        self.sandbox = sandbox

        self.data = []
        if data is not None:
            self.append_records(data)

    def to_dataframe(self):
        """Convert the annotation data to a pandas DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame containing the annotation data.
        """
        if not self.data:
            return pd.DataFrame(columns=["time", "duration", "value", "confidence"])

        data = []
        for obs in self.data:
            data.append({"time": obs.time, "duration": obs.duration, "value": obs.value, "confidence": obs.confidence})
        return pd.DataFrame(data)

    def append(self, time=None, duration=None, value=None, confidence=None):
        """Add an observation to this annotation.

        Parameters
        ----------
        time : float or None
            The time of the observation.
        duration : float or None
            The duration of the observation.
        value : any or None
            The value of the observation.
        confidence : float or None
            The confidence of the observation.
        """
        self.data.append(Observation(time=time, duration=duration, value=value, confidence=confidence))

    def append_records(self, records):
        """Add multiple observations to this annotation.

        Parameters
        ----------
        records : list or dict
            Either a list of observation dicts or a dict of lists.
        """
        if isinstance(records, dict):
            # Handle dictionary of lists format
            n_obs = len(next(iter(records.values())))
            for i in range(n_obs):
                self.data.append(Observation(**{k: v[i] for k, v in records.items()}))
        else:
            # Handle list of dictionaries format
            for record in records:
                self.data.append(Observation(**record))

    def append_columns(self, columns):
        """Add multiple observations to this annotation from columns.

        Parameters
        ----------
        columns : dict
            Dictionary of column arrays.
        """
        n_obs = len(next(iter(columns.values())))
        for i in range(n_obs):
            self.data.append(Observation(**{k: v[i] for k, v in columns.items()}))

    def validate(self, strict=True):
        """Validate this annotation against its schema.

        Parameters
        ----------
        strict : bool
            If `True`, raise an exception on validation failure.
            If `False`, issue a warning on validation failure.

        Returns
        -------
        valid : bool
            `True` if the annotation is valid.
            `False` otherwise.
        """
        valid = True
        try:
            schema.VALIDATOR.validate(self.__json_light__, self.__schema__)
        except jsonschema.ValidationError as invalid:
            if strict:
                raise SchemaError(str(invalid)) from None
            else:
                warnings.warn(str(invalid), stacklevel=2)
                valid = False
        return valid

    def trim(self, start_time, end_time, strict=False):
        """Trim this annotation to a given time range.

        Parameters
        ----------
        start_time : float
            The start time for the trimmed annotation.
        end_time : float
            The end time for the trimmed annotation.
        strict : bool
            If `True`, observations that lie completely outside the given
            range will be removed.

        Returns
        -------
        trimmed : Annotation
            A new annotation containing only the trimmed observations.
        """
        if end_time <= start_time:
            raise ParameterError("end_time must be greater than start_time")

        if self.duration is None:
            warnings.warn("annotation.duration is not defined", stacklevel=2)

        # Check for no overlap
        ann_start = self.time
        ann_end = self.time + (self.duration if self.duration is not None else 0)
        if not strict and (end_time <= ann_start or start_time >= ann_end):
            warnings.warn("No overlap between trim range and annotation", stacklevel=2)
            trimmed = Annotation(
                namespace=self.namespace,
                time=self.time,
                duration=0,
                annotation_metadata=self.annotation_metadata,
                sandbox=self.sandbox,
            )
            if not hasattr(trimmed.sandbox, "trim"):
                trimmed.sandbox.trim = []
            trimmed.sandbox.trim.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "trim_start": max(start_time, ann_start),
                    "trim_end": min(end_time, ann_end),
                }
            )
            return trimmed

        trimmed = Annotation(
            namespace=self.namespace,
            time=max(self.time, start_time),
            duration=max(
                0,
                min(self.time + (self.duration if self.duration is not None else 0), end_time)
                - max(self.time, start_time),
            ),
            annotation_metadata=self.annotation_metadata,
            sandbox=self.sandbox,
        )
        if not hasattr(trimmed.sandbox, "trim"):
            trimmed.sandbox.trim = []
        trimmed.sandbox.trim.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "trim_start": max(start_time, ann_start),
                "trim_end": min(end_time, ann_end),
            }
        )

        for obs in self.data:
            obs_start = obs.time
            obs_end = obs.time + obs.duration
            # Completely within
            if obs_start >= start_time and obs_end <= end_time:
                trimmed.data.append(
                    Observation(time=obs_start, duration=obs.duration, value=obs.value, confidence=obs.confidence)
                )
            elif not strict:
                # Partial overlap
                new_start = max(obs_start, start_time)
                new_end = min(obs_end, end_time)
                new_duration = new_end - new_start
                if new_duration > 0:
                    trimmed.data.append(
                        Observation(time=new_start, duration=new_duration, value=obs.value, confidence=obs.confidence)
                    )
        return trimmed

    def slice(self, start_time, end_time, strict=False):
        """Slice this annotation to a given time range.

        Parameters
        ----------
        start_time : float
            The start time for the sliced annotation.
        end_time : float
            The end time for the sliced annotation.
        strict : bool
            If `True`, observations that lie completely outside the given
            range will be removed.

        Returns
        -------
        sliced : Annotation
            A new annotation containing only the sliced observations.
        """
        if end_time <= start_time:
            raise ParameterError("end_time must be greater than start_time")

        ann_start = self.time
        ann_end = self.time + (self.duration if self.duration is not None else 0)
        sliced = Annotation(
            namespace=self.namespace,
            time=0,
            duration=max(0, min(ann_end, end_time) - max(ann_start, start_time)),
            annotation_metadata=self.annotation_metadata,
            sandbox=self.sandbox,
        )
        if not hasattr(sliced.sandbox, "slice"):
            sliced.sandbox.slice = []
        sliced.sandbox.slice.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "slice_start": max(start_time, ann_start),
                "slice_end": min(end_time, ann_end),
            }
        )

        for obs in self.data:
            obs_start = obs.time
            obs_end = obs.time + obs.duration
            # Completely within
            if obs_start >= start_time and obs_end <= end_time:
                new_obs = Observation(
                    time=obs_start - start_time, duration=obs.duration, value=obs.value, confidence=obs.confidence
                )
                sliced.data.append(new_obs)
            elif not strict:
                # Partial overlap
                new_start = max(obs_start, start_time)
                new_end = min(obs_end, end_time)
                new_duration = new_end - new_start
                if new_duration > 0:
                    new_obs = Observation(
                        time=new_start - start_time, duration=new_duration, value=obs.value, confidence=obs.confidence
                    )
                    sliced.data.append(new_obs)
        return sliced

    def __iter__(self):
        """Iterate over the observations in this annotation."""
        return iter(self.data)

    def __repr__(self):
        """Return a string representation of this annotation."""
        return f"<Annotation(namespace={self.namespace}, data={len(self.data)} observations)>"

    def __str__(self):
        """Return a string representation of this annotation."""
        return self.__repr__()

    def __json_light__(self, data=True):
        """Return a lightweight JSON representation of this annotation."""
        result = {
            "namespace": self.namespace,
            "time": self.time,
            "duration": self.duration,
            "annotation_metadata": self.annotation_metadata.__json_light__(),
            "sandbox": self.sandbox.__json_light__(),
        }
        if data:
            result["data"] = [obs.__json_light__() for obs in self.data]
        return result

    def __json__(self):
        """Return a JSON representation of this annotation."""
        return self.__json_light__(data=True)

    def to_interval_values(self):
        """Convert the annotation data to interval-value pairs.

        Returns
        -------
        intervals : list
            List of time intervals (start, end).
        values : list
            List of corresponding values.
        """
        intervals = [(obs.time, obs.time + obs.duration) for obs in self.data]
        values = [obs.value for obs in self.data]
        return intervals, values

    def to_samples(self, times, confidence=False):
        """Sample the annotation at specified times.

        Parameters
        ----------
        times : array-like
            Times to sample the annotation.
        confidence : bool
            If True, return confidence values along with values.

        Returns
        -------
        values : list
            List of lists of values at each time point.
        confidences : list, optional
            List of lists of confidence values at each time point.
            Only returned if confidence=True.
        """
        if not isinstance(times, (list, np.ndarray)):
            raise ParameterError("times must be a list or numpy array")

        if isinstance(times, list) and any(isinstance(t, list) for t in times):
            raise ParameterError("times must be a flat list or numpy array")

        if any(t < 0 for t in times):
            raise ParameterError("times must be non-negative")

        values = []
        confidences = []

        for t in times:
            time_values = []
            time_confidences = []
            for obs in self.data:
                if obs.time <= t <= obs.time + obs.duration:
                    time_values.append(obs.value)
                    time_confidences.append(obs.confidence)
            values.append(time_values if time_values else [])
            confidences.append(time_confidences if time_confidences else [])

        if confidence:
            return values, confidences
        return values


class Curator(JObject):
    """Curator

    Container object for curator metadata.
    """

    def __init__(self, name="", email="", **kwargs):
        """Create a Curator.

        Parameters
        ----------
        name : str
            The name of the curator.
        email : str
            The email address of the curator.
        kwargs : dict
            Additional keyword arguments to store in the curator.
        """
        super().__init__()
        self.name = name
        self.email = email
        for key, value in kwargs.items():
            if key != "_data":
                setattr(self, key, value)

    def __repr__(self):
        """Return a string representation of this object."""
        return f"<Curator(,\\n      email={self.email},\\n      name={self.name})>"

    def __str__(self):
        """Return a string representation of this object."""
        return f"Curator(name={self.name}, email={self.email})"

    def __eq__(self, other):
        """Compare two objects for equality."""
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name and self.email == other.email

    def __json__(self):
        """Return a JSON representation of this object."""
        return {"name": self.name, "email": self.email}

    def __json_light__(self, data=True):
        """Return a lightweight JSON representation of this object."""
        return self.__json__()


class AnnotationMetadata(JObject):
    """AnnotationMetadata

    Container object for annotation metadata.
    """

    def __init__(
        self,
        curator=None,
        version="",
        corpus="",
        annotator=None,
        annotation_tools="",
        annotation_rules="",
        validation="",
        data_source="",
        **kwargs,
    ):
        """Create an AnnotationMetadata object.

        Parameters
        ----------
        curator : Curator or None
            Object documenting a name and email address for the person of
            correspondence.
        version : str
            Version of this annotation.
        corpus : str
            Collection assignment.
        annotator : dict or None
            Sandbox for information about the specific annotator, such as
            musical experience, skill level, principal instrument, etc.
        annotation_tools : str
            Description of the tools used to create the annotation.
        annotation_rules : str
            Description of the rules provided to the annotator.
        validation : str
            Methods for validating the integrity of the data.
        data_source : str
            Description of where the data originated, e.g. 'Manual Annotation'.
        kwargs : dict
            Additional keyword arguments to store in the metadata.
        """
        super().__init__()

        if curator is None:
            curator = Curator()

        if annotator is None:
            annotator = JObject()

        self.curator = curator if isinstance(curator, Curator) else Curator(**curator)
        self.version = version
        self.corpus = corpus
        self.annotator = annotator if isinstance(annotator, JObject) else JObject(**annotator)
        self.annotation_tools = annotation_tools
        self.annotation_rules = annotation_rules
        self.validation = validation
        self.data_source = data_source

        for key, value in kwargs.items():
            if key != "_data":
                setattr(self, key, value)

    def __repr__(self):
        """Return a string representation of this object."""
        return f"<AnnotationMetadata(curator={self.curator}, version={self.version}, corpus={self.corpus})>"

    def __str__(self):
        """Return a string representation of this object."""
        return f"AnnotationMetadata(curator={self.curator}, version={self.version}, corpus={self.corpus})"

    def __eq__(self, other):
        """Compare two objects for equality."""
        if not isinstance(other, type(self)):
            return False
        return (
            self.curator == other.curator
            and self.version == other.version
            and self.corpus == other.corpus
            and self.annotator == other.annotator
            and self.annotation_tools == other.annotation_tools
            and self.annotation_rules == other.annotation_rules
            and self.validation == other.validation
            and self.data_source == other.data_source
        )

    def __json__(self):
        """Return a JSON representation of this object."""
        return {
            "curator": self.curator.__json__() if hasattr(self.curator, "__json__") else self.curator,
            "version": self.version,
            "corpus": self.corpus,
            "annotator": self.annotator.__json__() if hasattr(self.annotator, "__json__") else self.annotator,
            "annotation_tools": self.annotation_tools,
            "annotation_rules": self.annotation_rules,
            "validation": self.validation,
            "data_source": self.data_source,
        }

    def __json_light__(self, data=True):
        """Return a lightweight JSON representation of this object."""
        return self.__json__()


class FileMetadata(JObject):
    """FileMetadata

    Container object for file metadata.
    """

    def __init__(
        self,
        title="",
        artist="",
        release="",
        duration=None,
        identifiers=None,
        jams_version=JAMS_VERSION,
        **kwargs,
    ):
        """Create a FileMetadata object.

        Parameters
        ----------
        title : str
            The title of the track.
        artist : str
            The artist of the track.
        release : str
            The release name of the track.
        duration : float or None
            The duration of the track in seconds.
        identifiers : Sandbox or None
            A sandbox for file ID information.
        jams_version : str
            The version of the JAMS Schema.
        kwargs : dict
            Additional keyword arguments to store in the metadata.
        """
        super().__init__()

        if identifiers is None:
            identifiers = Sandbox()

        self.title = title
        self.artist = artist
        self.release = release
        self.duration = duration
        self.identifiers = identifiers if isinstance(identifiers, Sandbox) else Sandbox(**identifiers)
        self.jams_version = jams_version

        for key, value in kwargs.items():
            if key != "_data":
                setattr(self, key, value)

    def __repr__(self):
        """Return a string representation of this object."""
        return f"<FileMetadata(title={self.title}, artist={self.artist}, release={self.release})>"

    def __str__(self):
        """Return a string representation of this object."""
        return f"FileMetadata(title={self.title}, artist={self.artist}, release={self.release})"

    def __eq__(self, other):
        """Compare two objects for equality."""
        if not isinstance(other, type(self)):
            return False
        return (
            self.title == other.title
            and self.artist == other.artist
            and self.release == other.release
            and self.duration == other.duration
            and self.identifiers == other.identifiers
            and self.jams_version == other.jams_version
        )

    def __json__(self):
        """Return a JSON representation of this object."""
        return {
            "title": self.title,
            "artist": self.artist,
            "release": self.release,
            "duration": self.duration,
            "identifiers": self.identifiers.__json__() if hasattr(self.identifiers, "__json__") else self.identifiers,
            "jams_version": self.jams_version,
        }

    def __json_light__(self, data=True):
        """Return a lightweight JSON representation of this object."""
        return self.__json__()


class AnnotationArray(list):
    """Array of Annotation objects.

    This class extends the standard python list to allow for searching
    through annotation objects.

    Fancy-indexing can be used to directly search for annotations
    belonging to a particular namespace. Three types of indexing
    are supported:

    - integer or slice : acts just as in `list`, e.g., `arr[0]` or `arr[1:3]`
    - string : acts like a search, e.g.,
      `arr['beat'] == arr.search(namespace='beat')`
    - (string, integer or slice) acts like a search followed by index/slice
    """

    def __init__(self, annotations=None):
        """Create an array of annotations.

        Parameters
        ----------
        annotations : list-like
            A list of Annotation objects or dictionaries.
        """
        super().__init__()
        if annotations is not None:
            for obj in annotations:
                if isinstance(obj, Annotation):
                    self.append(obj)
                elif isinstance(obj, dict):
                    self.append(Annotation(**obj))
                else:
                    raise TypeError("AnnotationArray only accepts Annotation objects or dicts")

    def extend(self, iterable):
        for obj in iterable:
            if isinstance(obj, Annotation):
                super().append(obj)
            elif isinstance(obj, dict):
                super().append(Annotation(**obj))
            else:
                raise TypeError("AnnotationArray only accepts Annotation objects or dicts")

    def search(self, **kwargs):
        """Filter the annotation array down to only those whose properties match
        the given keys.

        Parameters
        ----------
        kwargs : keyword arguments
            Each key represents a field name to match.
            If the value is a string, then substring matching is performed.
            If the value is callable, then it is taken as a predicate function.

        Returns
        -------
        results : AnnotationArray
            A new annotation array containing only the matched annotations.

        Examples
        --------
        >>> # Find annotations with a namespace containing 'chord'
        >>> jam.search(namespace='chord')
        >>> # Find annotations with a namespace containing 'chord' and a curator
        >>> # named 'Brian'
        >>> jam.search(namespace='chord', curator='Brian')
        >>> # Find annotations with a namespace containing 'chord' and a value
        >>> # greater than 0.5
        >>> jam.search(namespace='chord', value=lambda x: x > 0.5)
        """
        results = []
        for annotation in self:
            if all(match_query(annotation.__dict__.get(key, ""), value) for key, value in kwargs.items()):
                results.append(annotation)
        return AnnotationArray(results)

    def __getitem__(self, idx):
        """Get an annotation by index.

        Parameters
        ----------
        idx : int, slice, str, or tuple
            The index or slice to get.
            If a string, it is treated as a namespace search.
            If a tuple, it is treated as (namespace, index).

        Returns
        -------
        annotation : Annotation or AnnotationArray
            The annotation(s) at the given index/slice.
        """
        if isinstance(idx, (int, slice)):
            result = super().__getitem__(idx)
            if isinstance(idx, slice):
                return AnnotationArray([result] if isinstance(result, Annotation) else result)
            return result
        elif isinstance(idx, str):
            return self.search(namespace=idx)
        elif isinstance(idx, tuple) and len(idx) == 2:
            namespace, sub_idx = idx
            return self.search(namespace=namespace)[sub_idx]
        raise IndexError(f"Invalid index: {idx}")

    @property
    def __json__(self):
        """Return a JSON representation of the annotation array."""
        return [annotation.__json__() for annotation in self]

    def trim(self, start_time, end_time, strict=False):
        """Trim all annotations to a given time range.

        Parameters
        ----------
        start_time : float
            The start time for the trimmed annotations.
        end_time : float
            The end time for the trimmed annotations.
        strict : bool
            If `True`, annotations that lie completely outside the given
            range will be removed.

        Returns
        -------
        trimmed : AnnotationArray
            A new annotation array containing only the trimmed annotations.
        """
        trimmed = []
        for annotation in self:
            try:
                trimmed.append(annotation.trim(start_time, end_time, strict=strict))
            except ParameterError:
                if strict:
                    continue
                trimmed.append(annotation)
        return AnnotationArray(trimmed)

    def slice(self, start_time, end_time, strict=False):
        """Slice all annotations to a given time range.

        Parameters
        ----------
        start_time : float
            The start time for the sliced annotations.
        end_time : float
            The end time for the sliced annotations.
        strict : bool
            If `True`, annotations that lie completely outside the given
            range will be removed.

        Returns
        -------
        sliced : AnnotationArray
            A new annotation array containing only the sliced annotations.
        """
        sliced = []
        for annotation in self:
            try:
                sliced.append(annotation.slice(start_time, end_time, strict=strict))
            except ParameterError:
                if strict:
                    continue
                sliced.append(annotation)
        return AnnotationArray(sliced)

    def __repr__(self):
        """Return a string representation of the annotation array."""
        n_annot = len(self)
        if n_annot == 0:
            return "[]"
        return f"[{n_annot:d} annotations]"

    def _repr_html_(self):
        """Return an HTML representation of the annotation array."""
        out = f"<div><pre>{self.__repr__()}</pre></div>"
        return out

    def append(self, obj):
        if isinstance(obj, Annotation):
            super().append(obj)
        elif isinstance(obj, dict):
            super().append(Annotation(**obj))
        else:
            raise TypeError("AnnotationArray only accepts Annotation objects or dicts")

    def __setitem__(self, idx, obj):
        if isinstance(obj, Annotation):
            super().__setitem__(idx, obj)
        elif isinstance(obj, dict):
            super().__setitem__(idx, Annotation(**obj))
        else:
            raise TypeError("AnnotationArray only accepts Annotation objects or dicts")

    def insert(self, idx, obj):
        if isinstance(obj, Annotation):
            super().insert(idx, obj)
        elif isinstance(obj, dict):
            super().insert(idx, Annotation(**obj))
        else:
            raise TypeError("AnnotationArray only accepts Annotation objects or dicts")

    def __add__(self, other):
        result = AnnotationArray(self)
        result.extend(other)
        return result

    def __iadd__(self, other):
        self.extend(other)
        return self


class JAMS(JObject):
    """Top-level NEOJAMS Object"""

    def __init__(self, annotations=None, file_metadata=None, sandbox=None):
        """Create a NEOJAMS object.

        Parameters
        ----------
        annotations : list of Annotations
            Zero or more Annotation objects

        file_metadata : FileMetadata (or dict), default=None
            Metadata corresponding to the audio file.

        sandbox : Sandbox (or dict), default=None
            Unconstrained global sandbox for additional information.

        """
        super().__init__()

        if file_metadata is None:
            file_metadata = FileMetadata()
        elif isinstance(file_metadata, dict):
            file_metadata = FileMetadata(**file_metadata)
        # else, assume it's already a FileMetadata

        if sandbox is None:
            sandbox = Sandbox()
        elif isinstance(sandbox, dict):
            sandbox = Sandbox(**sandbox)
        # else, assume it's already a Sandbox

        self.annotations = AnnotationArray(annotations=annotations)
        self.file_metadata = file_metadata
        self.sandbox = sandbox

    def _display_properties(self):
        return [("file_metadata", "File Metadata"), ("annotations", "Annotations"), ("sandbox", "Sandbox")]

    @property
    def __schema__(self):
        return schema.JAMS_SCHEMA

    def add(self, jam, on_conflict="fail"):
        """Add the contents of another jam to this object.

        Note that, by default, this method fails if file_metadata is not
        identical and raises a ValueError; either resolve this manually
        (because conflicts should almost never happen), force an 'overwrite',
        or tell the method to 'ignore' the metadata of the object being added.

        Parameters
        ----------
        jam: NEOJAMS object
            Object to add to this jam

        on_conflict: str, default='fail'
            Strategy for resolving metadata conflicts; one of
                ['fail', 'overwrite', or 'ignore'].

        Raises
        ------
        ParameterError
            if `on_conflict` is an unknown value

        JamsError
            If a conflict is detected and `on_conflict='fail'`
        """

        if on_conflict not in ["overwrite", "fail", "ignore"]:
            raise ParameterError(f"on_conflict='{on_conflict}' is not in ['fail', 'overwrite', 'ignore'].")

        if not self.file_metadata == jam.file_metadata:
            if on_conflict == "overwrite":
                self.file_metadata = jam.file_metadata
            elif on_conflict == "fail":
                raise JamsError("Metadata conflict! " "Resolve manually or force-overwrite it.")

        self.annotations.extend(jam.annotations)
        # Re-wrap as AnnotationArray to ensure all items are Annotation objects
        self.annotations = AnnotationArray(self.annotations)
        self.sandbox.update(**jam.sandbox)

    def search(self, **kwargs):
        """Search a NEOJAMS object for matching objects.

        Parameters
        ----------
        kwargs : keyword arguments
            Keyword query

        Returns
        -------
        AnnotationArray
            All annotation objects in this NEOJAMS which match the query

        See Also
        --------
        JObject.search
        AnnotationArray.search


        Examples
        --------
        A simple query to get all beat annotations

        >>> beats = my_jams.search(namespace='beat')

        """

        return self.annotations.search(**kwargs)

    def save(self, path_or_file, strict=True, fmt="auto"):
        """Serialize annotation as a JSON formatted stream to file.

        Parameters
        ----------
        path_or_file : str or file-like
            Path to save the NEOJAMS object on disk
            OR
            An open file descriptor to write into

        strict : bool
            Force strict schema validation

        fmt : str ['auto', 'jams', 'jamz']
            The output encoding format.

            If `auto`, it is inferred from the file name.

            If the input is an open file handle, `jams` encoding
            is used.


        Raises
        ------
        SchemaError
            If `strict == True` and the NEOJAMS object fails schema
            or namespace validation.

        See also
        --------
        validate
        """

        self.validate(strict=strict)

        with _open(path_or_file, mode="w", fmt=fmt) as fdesc:
            json.dump(self.__json__, fdesc, indent=2)

    def validate(self, strict=True):
        """Validate a NEOJAMS object against the schema.

        Parameters
        ----------
        strict : bool
            If `True`, an exception will be raised on validation failure.
            If `False`, a warning will be raised on validation failure.

        Returns
        -------
        valid : bool
            `True` if the object passes schema validation.
            `False` otherwise.

        Raises
        ------
        SchemaError
            If `strict==True` and the NEOJAMS object does not match the schema

        See Also
        --------
        jsonschema.validate

        """
        valid = True
        try:
            schema.VALIDATOR.validate(self.__json_light__, schema.JAMS_SCHEMA)

            for ann in self.annotations:
                if isinstance(ann, Annotation):
                    valid &= ann.validate(strict=strict)
                else:
                    msg = f"{ann} is not a well-formed NEOJAMS Annotation"
                    valid = False
                    if strict:
                        raise SchemaError(msg)
                    else:
                        warnings.warn(str(msg), stacklevel=2)

        except jsonschema.ValidationError as invalid:
            if strict:
                raise SchemaError(str(invalid)) from None
            else:
                warnings.warn(str(invalid), stacklevel=2)

            valid = False

        return valid

    def trim(self, start_time, end_time, strict=False):
        """
        Trim all the annotations inside the jam and return as a new `JAMS`
        object.

        See `Annotation.trim` for details about how the annotations
        are trimmed.

        This operation is also documented in the jam-level sandbox
        with a list keyed by ``JAMS.sandbox.trim`` containing a tuple for each
        jam-level trim of the form ``(start_time, end_time)``.

        This function also copies over all of the file metadata from the
        original jam.

        Note: trimming does not affect the duration of the jam, i.e. the value
        of ``JAMS.file_metadata.duration`` will be the same for the original
        and trimmed jams.

        Parameters
        ----------
        start_time : float
            The desired start time for the trimmed annotations in seconds.
        end_time
            The desired end time for trimmed annotations in seconds. Must be
            greater than ``start_time``.
        strict : bool
            When ``False`` (default) observations that lie at the boundaries of
            the trimming range (see `Annotation.trim` for details), will have
            their time and/or duration adjusted such that only the part of the
            observation that lies within the trim range is kept. When ``True``
            such observations are discarded and not included in the trimmed
            annotation.

        Returns
        -------
        jam_trimmed : JAMS
            The trimmed jam with trimmed annotations, returned as a new JAMS
            object.

        """
        # Make sure duration is set in file metadata
        if self.file_metadata.duration is None:
            raise JamsError("Duration must be set (jam.file_metadata.duration) before " "trimming can be performed.")

        # Make sure start and end times are within the file start/end times
        if not (0 <= start_time <= end_time <= float(self.file_metadata.duration)):
            raise ParameterError(
                "start_time and end_time must be within the original file "
                f"duration ({float(self.file_metadata.duration):f}) and end_time cannot be smaller than "
                "start_time."
            )

        # Create a new jams
        jam_trimmed = JAMS(annotations=None, file_metadata=self.file_metadata, sandbox=self.sandbox)

        # trim annotations
        jam_trimmed.annotations = self.annotations.trim(start_time, end_time, strict=strict)

        # Document jam-level trim in top level sandbox
        if "trim" not in jam_trimmed.sandbox.keys():
            jam_trimmed.sandbox.update(trim=[{"start_time": start_time, "end_time": end_time}])
        else:
            jam_trimmed.sandbox.trim.append({"start_time": start_time, "end_time": end_time})

        return jam_trimmed

    def slice(self, start_time, end_time, strict=False):
        """
        Slice all the annotations inside the jam and return as a new `JAMS`
        object.

        See `Annotation.slice` for details about how the annotations
        are sliced.

        This operation is also documented in the jam-level sandbox
        with a list keyed by ``JAMS.sandbox.slice`` containing a tuple for each
        jam-level slice of the form ``(start_time, end_time)``.

        Since slicing is implemented using trimming, the operation will also be
        documented in ``JAMS.sandbox.trim`` as described in `JAMS.trim`.

        This function also copies over all of the file metadata from the
        original jam.

        Note: slicing will affect the duration of the jam, i.e. the new value
        of ``JAMS.file_metadata.duration`` will be ``end_time - start_time``.

        Parameters
        ----------
        start_time : float
            The desired start time for slicing in seconds.
        end_time
            The desired end time for slicing in seconds. Must be greater than
            ``start_time``.
        strict : bool
            When ``False`` (default) observations that lie at the boundaries of
            the slicing range (see `Annotation.slice` for details), will have
            their time and/or duration adjusted such that only the part of the
            observation that lies within the slice range is kept. When ``True``
            such observations are discarded and not included in the sliced
            annotation.

        Returns
        -------
        jam_sliced: JAMS
            The sliced jam with sliced annotations, returned as a new
            JAMS object.

        """
        # Make sure duration is set in file metadata
        if self.file_metadata.duration is None:
            raise JamsError("Duration must be set (jam.file_metadata.duration) before " "slicing can be performed.")

        # Make sure start and end times are within the file start/end times
        if (
            start_time < 0
            or start_time > float(self.file_metadata.duration)
            or end_time < start_time
            or end_time > float(self.file_metadata.duration)
        ):
            raise ParameterError(
                "start_time and end_time must be within the original file "
                f"duration ({float(self.file_metadata.duration):f}) and end_time cannot be smaller than "
                "start_time."
            )

        # Create a new jams
        jam_sliced = JAMS(annotations=None, file_metadata=self.file_metadata, sandbox=self.sandbox)

        # trim annotations
        jam_sliced.annotations = self.annotations.slice(start_time, end_time, strict=strict)

        # adjust dutation
        jam_sliced.file_metadata.duration = end_time - start_time

        # Document jam-level trim in top level sandbox
        if "slice" not in jam_sliced.sandbox.keys():
            jam_sliced.sandbox.update(slice=[{"start_time": start_time, "end_time": end_time}])
        else:
            jam_sliced.sandbox.slice.append({"start_time": start_time, "end_time": end_time})

        return jam_sliced

    @property
    def __json_light__(self):
        r"""Return the JObject as a set of native data types for serialization.

        Note: attributes beginning with underscores are suppressed.

        This also skips the `annotations` field, which will be validated separately.
        """
        filtered_dict = {}

        for k, item in iteritems(self.__dict__):
            if k.startswith("_"):
                continue
            if k == "annotations":
                # Serialize annotations as a list of dicts
                filtered_dict[k] = [ann.__json_light__() for ann in self.annotations]
            elif hasattr(item, "__json__"):
                filtered_dict[k] = item.__json__()
            else:
                filtered_dict[k] = serialize_obj(item)

        return filtered_dict
