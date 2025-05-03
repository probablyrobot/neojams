#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-05-18 14:35:43 by Brian McFee <brian.mcfee@nyu.edu>
"""Test display functions"""

import os
import tempfile

import matplotlib
import numpy as np
import six

import neojams
from neojams import NamespaceError

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest


def test_display_jam():

    jam = neojams.JAMS()

    # Add a beat annotation
    beat = neojams.Annotation(namespace='beat')
    beat.append(time=0, duration=0.0, value=1)
    beat.append(time=1, duration=0.0, value=2)
    beat.append(time=2, duration=0.0, value=3)
    beat.append(time=3, duration=0.0, value=1)

    jam.annotations.append(beat)

    # Add a segment annotation
    segment = neojams.Annotation(namespace='segment_open')
    segment.append(time=0, duration=2, value='A')
    segment.append(time=2, duration=2, value='B')

    jam.annotations.append(segment)

    # Set the file metadata
    jam.file_metadata.duration = (
        max(jam.annotations[1].data[-1].time + jam.annotations[1].data[-1].duration,
            jam.annotations[0].data[-1].time) + 1)

    plt.figure()
    neojams.display.display_jam(jam)

    plt.figure()
    neojams.display.display_jam(jam, annotation_ids=set([1]))

    plt.figure()
    neojams.display.display_jam(jam, annotation_ids=set([0, -1]))

    plt.figure()
    neojams.display.display_jam(jam, annotation_ids=set([0, 2]),
                             label='Testing display_jam')

    plt.figure()
    neojams.display.display_jam(jam, annotation_ids=set([0, 2]), time_range=[1, 3],
                             label='Testing display_jam')

    plt.figure()
    neojams.display.display_jam(jam, time_range=[1, 3],
                             label='Testing display_jam')

    plt.figure()
    neojams.display.display_jam(jam, annotation_ids=[0, 1], time_range=[0, 4],
                             label='Testing display_jam')


@pytest.mark.parametrize(
    "namespace",
    [
        "segment_open",
        "chord",
        "multi_segment",
        "pitch_contour",
        "beat_position",
        "beat",
        "onset",
        "note_midi",
        "tag_open",
    ],
)
@pytest.mark.parametrize("meta", [False, True])
def test_display(namespace, meta):
    ann = neojams.Annotation(namespace=namespace)
    neojams.display.display(ann, meta=meta)


@pytest.mark.parametrize("namespace", ["tempo"])
@pytest.mark.parametrize("meta", [False, True])
def test_display_exception(namespace, meta):
    with pytest.raises(NamespaceError):
        ann = neojams.Annotation(namespace=namespace)
        neojams.display.display(ann, meta=meta)


def test_display_multi():
    jam = neojams.JAMS()
    jam.annotations.append(neojams.Annotation(namespace="beat"))
    neojams.display.display_multi(jam.annotations)


def test_display_multi_multi():
    jam = neojams.JAMS()
    jam.annotations.append(neojams.Annotation(namespace="beat"))
    jam.annotations.append(neojams.Annotation(namespace="chord"))

    neojams.display.display_multi(jam.annotations)


def test_display_pitch_contour():
    ann = neojams.Annotation(namespace="pitch_hz", duration=5)

    values = list(range(100, 200))
    times = list(range(len(values)))

    for t, v in zip(times, values, strict=False):
        ann.append(time=t, value=v, duration=0)

    neojams.display.display(ann)


def test_display_labeled_events():
    times = list(range(40))
    values = [t % 4 for t in times]

    ann = neojams.Annotation(namespace="beat", duration=60)

    for t, v in zip(times, values, strict=False):
        ann.append(time=t, value=v, duration=0)

    neojams.display.display(ann)


def test_display_multi_fail():
    anns = neojams.AnnotationArray()

    with pytest.raises(neojams.ParameterError):
        neojams.display.display_multi(anns)


def test_display_hierarchy_bad():
    jam = neojams.JAMS()

    with pytest.raises(NamespaceError):
        neojams.display.display_hierarchy(jam)


def test_display_hierarchy():

    jam = neojams.JAMS()

    # Add a segment annotation
    segment = neojams.Annotation(namespace='segment_tut')
    segment.append(time=0, duration=2.0, value='A', confidence=1.0)
    segment.append(time=2, duration=2.0, value='B', confidence=1.0)
    segment.append(time=4, duration=4.0, value='A', confidence=1.0)

    jam.annotations.append(segment)

    segment = neojams.Annotation(namespace='segment_tut')
    segment.append(time=0, duration=2.0, value='a', confidence=1.0)
    segment.append(time=2, duration=1.0, value='b', confidence=1.0)
    segment.append(time=3, duration=1.0, value='c', confidence=1.0)
    segment.append(time=4, duration=2.0, value='a', confidence=1.0)
    segment.append(time=6, duration=2.0, value='d', confidence=1.0)
    segment.sandbox.poch = 1

    jam.annotations.append(segment)

    # Set the file metadata
    jam.file_metadata.duration = 8

    plt.figure()
    neojams.display.display_hierarchy(jam)

    plt.figure()
    neojams.display.display_hierarchy(jam, label='hierarchy')

    plt.figure()
    neojams.display.display_hierarchy(jam, annotation_ids=[0], time_range=[0, 4])

    plt.figure()
    neojams.display.display_hierarchy(jam, time_range=[0, 4], label='hier')

    plt.figure()
    neojams.display.display_hierarchy(jam, time_range=[0, 4], label=None)


def test_display_beat():

    jam = neojams.JAMS()

    # Add a beat annotation
    beat = neojams.Annotation(namespace='beat')
    beat.append(time=0, duration=0.0, value=1)
    beat.append(time=1, duration=0.0, value=2)
    beat.append(time=2, duration=0.0, value=3)
    beat.append(time=3, duration=0.0, value=1)

    jam.annotations.append(beat)

    # Set the file metadata
    jam.file_metadata.duration = 4

    plt.figure()
    neojams.display.display_beat(jam)

    plt.figure()
    neojams.display.display_beat(jam, annotation_ids=[0])

    plt.figure()
    neojams.display.display_beat(jam, label='Testing display_beat')

    plt.figure()
    neojams.display.display_beat(jam, annotation_ids=[0, 1], time_range=[1, 3],
                              label='Testing display_beat')
