#!/usr/bin/env python
"""
Visualization
============

Visualization tools for JAMS

.. autosummary::
    :toctree: generated/

    display
    display_multi
"""

import json
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import mir_eval.display
import numpy as np
from matplotlib.offsetbox import AnchoredText

from .compatibility import iteritems
from .eval import coerce_annotation, hierarchy_flatten
from .exceptions import NamespaceError, ParameterError
from .nsconvert import can_convert


def pprint_jobject(obj, **kwargs):
    """Pretty-print a jobject.

    Parameters
    ----------
    obj : jams.JObject

    kwargs
        additional parameters to `json.dumps`

    Returns
    -------
    string
        A simplified display of `obj` contents.
    """

    obj_simple = {k: v for k, v in iteritems(obj.__json__) if v}

    string = json.dumps(obj_simple, **kwargs)

    # Suppress braces and quotes
    string = re.sub(r'[{}"]', "", string)

    # Kill trailing commas
    string = re.sub(r",\n", "\n", string)

    # Kill blank lines
    string = re.sub(r"^\s*$", "", string)

    return string


def intervals(annotation, **kwargs):
    """Plotting wrapper for labeled intervals"""
    times, labels = annotation.to_interval_values()

    return mir_eval.display.labeled_intervals(times, labels, **kwargs)


def hierarchy(annotation, **kwargs):
    """Plotting wrapper for hierarchical segmentations"""
    htimes, hlabels = hierarchy_flatten(annotation)

    htimes = [np.asarray(_) for _ in htimes]
    return mir_eval.display.hierarchy(htimes, hlabels, **kwargs)


def pitch_contour(annotation, **kwargs):
    """Plotting wrapper for pitch contours"""
    ax = kwargs.pop("ax", None)

    # If the annotation is empty, we need to construct a new axes
    ax = mir_eval.display.__get_axes(ax=ax)[0]

    times, values = annotation.to_interval_values()

    # Convert times to numpy array if it's a list
    if isinstance(times, list):
        times_array = np.array(times)
    else:
        times_array = times

    indices = np.unique([v["index"] for v in values])

    for idx in indices:
        rows = [i for (i, v) in enumerate(values) if v["index"] == idx]
        freqs = np.asarray([values[r]["frequency"] for r in rows])
        unvoiced = ~np.asarray([values[r]["voiced"] for r in rows])
        freqs[unvoiced] *= -1

        # Use the first column of the times array for the time values
        ax = mir_eval.display.pitch(times_array[rows, 0], freqs, unvoiced=True, ax=ax, **kwargs)
    return ax


def event(annotation, **kwargs):
    """Plotting wrapper for events"""

    times, values = annotation.to_interval_values()

    if any(values):
        labels = values
    else:
        labels = None

    return mir_eval.display.events(times, labels=labels, **kwargs)


def beat_position(annotation, **kwargs):
    """Plotting wrapper for beat-position data"""

    times, values = annotation.to_interval_values()

    labels = [_["position"] for _ in values]

    # TODO: plot time signature, measure number
    return mir_eval.display.events(times, labels=labels, **kwargs)


def piano_roll(annotation, **kwargs):
    """Plotting wrapper for piano rolls"""
    times, midi = annotation.to_interval_values()

    return mir_eval.display.piano_roll(times, midi=midi, **kwargs)


VIZ_MAPPING = OrderedDict()

VIZ_MAPPING["segment_open"] = intervals
VIZ_MAPPING["chord"] = intervals
VIZ_MAPPING["multi_segment"] = hierarchy
VIZ_MAPPING["pitch_contour"] = pitch_contour
VIZ_MAPPING["beat_position"] = beat_position
VIZ_MAPPING["beat"] = event
VIZ_MAPPING["onset"] = event
VIZ_MAPPING["note_midi"] = piano_roll
VIZ_MAPPING["tag_open"] = intervals


def display(annotation, meta=True, **kwargs):
    """Visualize a jams annotation through mir_eval

    Parameters
    ----------
    annotation : jams.Annotation
        The annotation to display

    meta : bool
        If `True`, include annotation metadata in the figure

    kwargs
        Additional keyword arguments to mir_eval.display functions

    Returns
    -------
    ax
        Axis handles for the new display

    Raises
    ------
    NamespaceError
        If the annotation cannot be visualized
    """

    for namespace, func in iteritems(VIZ_MAPPING):
        try:
            ann = coerce_annotation(annotation, namespace)

            axes = func(ann, **kwargs)

            # Title should correspond to original namespace, not the coerced version
            axes.set_title(annotation.namespace)
            if meta:
                description = pprint_jobject(annotation.annotation_metadata)

                anchored_box = AnchoredText(
                    description.strip("\n"),
                    loc=2,
                    frameon=True,
                    bbox_to_anchor=(1.02, 1.0),
                    bbox_transform=axes.transAxes,
                    borderpad=0.0,
                )
                axes.add_artist(anchored_box)

                axes.figure.subplots_adjust(right=0.8)

            return axes
        except NamespaceError:
            pass

    raise NamespaceError(f'Unable to visualize annotation of namespace="{annotation.namespace:s}"')


def display_multi(annotations, fig_kw=None, meta=True, **kwargs):
    """Display multiple annotations with shared axes

    Parameters
    ----------
    annotations : jams.AnnotationArray
        A collection of annotations to display

    fig_kw : dict
        Keyword arguments to `plt.figure`

    meta : bool
        If `True`, display annotation metadata for each annotation

    kwargs
        Additional keyword arguments to the `mir_eval.display` routines

    Returns
    -------
    fig
        The created figure
    axs
        List of subplot axes corresponding to each displayed annotation
    """
    if fig_kw is None:
        fig_kw = {}

    fig_kw.setdefault("sharex", True)
    fig_kw.setdefault("squeeze", True)

    # Filter down to coercable annotations first
    display_annotations = []
    for ann in annotations:
        for namespace in VIZ_MAPPING:
            if can_convert(ann, namespace):
                display_annotations.append(ann)
                break

    # If there are no displayable annotations, fail here
    if not len(display_annotations):
        raise ParameterError("No displayable annotations found")

    fig, axs = plt.subplots(nrows=len(display_annotations), ncols=1, **fig_kw)

    # MPL is stupid when making singleton subplots.
    # We catch this and make it always iterable.
    if len(display_annotations) == 1:
        axs = [axs]

    for ann, ax in zip(display_annotations, axs, strict=False):
        kwargs["ax"] = ax
        display(ann, meta=meta, **kwargs)

    return fig, axs


def display_beat(jam, annotation_ids=None, time_range=None, label=None, **kwargs):
    """Display beat annotations in a jams object.

    Parameters
    ----------
    jam : jams.JAMS
        The JAMS object containing beat annotations

    annotation_ids : list or None
        IDs of the annotations to include
        If None, all beat annotations are displayed

    time_range : list of two floats or None
        Time range (in seconds) for the visualization
        If None, the entire timeline is displayed

    label : str or None
        Title for the plot

    kwargs
        Additional keyword arguments to mir_eval.display functions

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the beat annotations displayed
    """
    if annotation_ids is None:
        # Direct search might not work, so let's filter manually
        annotations = []
        for ann in jam.annotations:
            if ann.namespace == "beat":
                annotations.append(ann)
    else:
        annotations = []
        for ann_id in annotation_ids:
            if ann_id < len(jam.annotations):
                if jam.annotations[ann_id].namespace == "beat":
                    annotations.append(jam.annotations[ann_id])

    if not annotations:
        raise ParameterError("No beat annotations found")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for ann in annotations:
        times, values = ann.to_interval_values()
        if time_range is not None:
            # Filter by time range
            if isinstance(times, list):
                mask = np.array([(t[0] >= time_range[0]) and (t[0] <= time_range[1]) for t in times])
                times = [times[i] for i in range(len(times)) if mask[i]]
                if values:
                    values = [v for i, v in enumerate(values) if mask[i]]
            else:
                mask = (times[:, 0] >= time_range[0]) & (times[:, 0] <= time_range[1])
                times = times[mask]
                if values:
                    values = [v for i, v in enumerate(values) if mask[i]]

        if any(values):
            labels = values
        else:
            labels = None

        ax = mir_eval.display.events(times, labels=labels, ax=ax, **kwargs)

    if label is not None:
        ax.set_title(label)

    return fig


def display_hierarchy(jam, annotation_ids=None, time_range=None, label=None, **kwargs):
    """Display hierarchical segmentation annotations in a jams object.

    Parameters
    ----------
    jam : jams.JAMS
        The JAMS object containing segmentation annotations

    annotation_ids : list or None
        IDs of the annotations to include
        If None, all segment annotations are displayed

    time_range : list of two floats or None
        Time range (in seconds) for the visualization
        If None, the entire timeline is displayed

    label : str or None
        Title for the plot

    kwargs
        Additional keyword arguments to mir_eval.display functions

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the segmentation hierarchy displayed
    """
    segment_namespaces = [
        "segment_open",
        "segment_tut",
        "segment_salami_lower",
        "segment_salami_upper",
        "segment_salami_function",
    ]

    if annotation_ids is None:
        # Get all segment annotations manually
        annotations = []
        for ann in jam.annotations:
            if ann.namespace in segment_namespaces:
                annotations.append(ann)
    else:
        annotations = []
        for ann_id in annotation_ids:
            if ann_id < len(jam.annotations):
                if jam.annotations[ann_id].namespace in segment_namespaces:
                    annotations.append(jam.annotations[ann_id])

    if not annotations:
        raise NamespaceError("No segmentation annotations found")

    # Flatten the hierarchy
    htimes, hlabels = [], []
    for ann in annotations:
        h_t, h_l = hierarchy_flatten(ann)
        htimes.append(h_t[0])
        hlabels.append(h_l[0])

    if time_range is not None:
        # Filter by time range
        for i in range(len(htimes)):
            if isinstance(htimes[i], list):
                mask = np.array([(t[0] >= time_range[0]) and (t[1] <= time_range[1]) for t in htimes[i]])
                htimes[i] = [htimes[i][j] for j in range(len(htimes[i])) if mask[j]]
                hlabels[i] = [label_item for j, label_item in enumerate(hlabels[i]) if mask[j]]
            else:
                mask = (htimes[i][:, 0] >= time_range[0]) & (htimes[i][:, 1] <= time_range[1])
                htimes[i] = htimes[i][mask]
                hlabels[i] = [label_item for j, label_item in enumerate(hlabels[i]) if mask[j]]

    # Convert lists to numpy arrays
    htimes = [np.asarray(ht) for ht in htimes]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax = mir_eval.display.hierarchy(htimes, hlabels, ax=ax, **kwargs)

    if label is not None:
        ax.set_title(label)

    return fig


def display_jam(jam, annotation_ids=None, time_range=None, label=None, **kwargs):
    """Display all visualizable annotations in a JAMS object.

    Parameters
    ----------
    jam : jams.JAMS
        The JAMS object to display

    annotation_ids : list or None
        IDs of the annotations to include
        If None, all visualizable annotations are displayed

    time_range : list of two floats or None
        Time range (in seconds) for the visualization
        If None, the entire timeline is displayed

    label : str or None
        Title for the plot

    kwargs
        Additional keyword arguments to mir_eval.display functions

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with all visualizable annotations
    """
    if annotation_ids is None:
        # Get all displayable annotations
        annotations = []
        for ann in jam.annotations:
            for namespace in VIZ_MAPPING:
                if can_convert(ann, namespace):
                    annotations.append(ann)
                    break
    else:
        annotations = []
        for ann_id in annotation_ids:
            if ann_id < len(jam.annotations):
                for namespace in VIZ_MAPPING:
                    if can_convert(jam.annotations[ann_id], namespace):
                        annotations.append(jam.annotations[ann_id])
                        break

    if not annotations:
        raise ParameterError("No displayable annotations found")

    fig, axs = plt.subplots(nrows=len(annotations), ncols=1, sharex=True)

    # MPL is stupid when making singleton subplots.
    # We catch this and make it always iterable.
    if len(annotations) == 1:
        axs = [axs]

    for ann, ax in zip(annotations, axs, strict=False):
        kwargs["ax"] = ax

        if time_range is not None:
            # Create a time-range filtered version of the annotation
            filtered_ann = ann.trim(time_range[0], time_range[1], strict=False)
            display(filtered_ann, meta=True, **kwargs)
        else:
            display(ann, meta=True, **kwargs)

    if label is not None:
        fig.suptitle(label)

    return fig


def __check_axes(ax):
    """Check if an axis is provided and create one if not.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis to check

    Returns
    -------
    ax : matplotlib.axes.Axes
        New or provided axis
    """
    if ax is None:
        _, ax = plt.subplots()
    return ax
