#!/usr/bin/env python
# CREATED:2016-01-16 10:38:56 by Brian McFee <brian.mcfee@nyu.edu>
"""Testing the eval module"""

import numpy as np
import pytest
from mir_eval.chord import NO_CHORD
from test_util import srand

import neojams


# Fixtures
def create_annotation(values, namespace="beat", offset=0.0, duration=1, confidence=1):
    ann = neojams.Annotation(namespace=namespace)

    time = np.arange(offset, offset + len(values))

    if np.isscalar(duration):
        time = time * duration
        duration = [duration] * len(time)

    if np.isscalar(confidence):
        confidence = [confidence] * len(time)

    for t, d, v, c in zip(time, duration, values, confidence, strict=False):
        ann.append(time=t, duration=d, value=v, confidence=c)

    return ann


def create_hierarchy(values, offset=0.0, duration=20):
    ann = neojams.Annotation(namespace="multi_segment")

    for level, labels in enumerate(values):
        times = np.linspace(offset, offset + duration, num=len(labels), endpoint=False)

        durations = list(np.diff(times))
        durations.append(duration + offset - times[-1])

        for t, d, v in zip(times, durations, labels, strict=False):
            ann.append(time=t, duration=d, value=dict(label=v, level=level))

    return ann


@pytest.fixture(scope="module")
def ref_beat():
    return create_annotation(values=np.arange(10) % 4 + 0.5, namespace="beat")


@pytest.fixture(scope="module")
def est_beat():
    return create_annotation(values=np.arange(9) % 4 + 1, namespace="beat", offset=0.01)


@pytest.fixture(scope="module")
def ref_onset():
    return create_annotation(values=np.arange(10) % 4 + 1.0, namespace="onset")


@pytest.fixture(scope="module")
def est_onset():
    return create_annotation(values=np.arange(9) % 4 + 1.0, namespace="onset", offset=0.01)


@pytest.fixture(scope="module")
def ref_chord():
    return create_annotation(values=["C", "E", "G:min7"], namespace="chord")


@pytest.fixture(scope="module")
def est_chord():
    return create_annotation(values=["D", "E", "G:maj"], namespace="chord_harte")


@pytest.fixture(scope="module")
def est_roman():
    return create_annotation(values=[{"tonic": "C", "chord": "I"}], namespace="chord_roman")


@pytest.fixture(scope="module")
def est_badchord():
    return create_annotation(values=["D", "E", "not at all a chord"], namespace="chord_harte", offset=0.01)


@pytest.fixture(scope="module")
def ref_segment():
    return create_annotation(values=["A", "B", "A", "C"], namespace="segment_open")


@pytest.fixture(scope="module")
def est_segment():
    return create_annotation(values=["E", "B", "E", "B"], namespace="segment_open")


@pytest.fixture(scope="module")
def est_segtut():
    return create_annotation(values=[["F"], "E", "B", "E", "B"], namespace="segment_tut")


@pytest.fixture(scope="module")
def ref_tempo():
    return create_annotation(values=[120.0, 60.0], confidence=[0.75, 0.25], namespace="tempo")


@pytest.fixture(scope="module")
def est_tempo():
    return create_annotation(values=[120.0, 80.0], confidence=[0.5, 0.5], namespace="tempo")


@pytest.fixture(scope="module")
def est_badtempo():
    return create_annotation(values=[120.0, 80.0], confidence=[-5, 1.5], namespace="tempo")


@pytest.fixture(scope="module")
def est_tag():
    return create_annotation(values=["120.0", "80.0"], confidence=[0.5, 0.5], namespace="tag_open")


@pytest.fixture(scope="module")
def ref_melody():
    srand()
    freq = np.linspace(110.0, 440.0, 10)
    voice = np.sign(np.random.randn(len(freq)))
    return create_annotation(values=freq * voice, confidence=1.0, duration=0.01, namespace="pitch_hz")


@pytest.fixture(scope="module")
def est_melody():
    srand()
    freq = np.linspace(110.0, 440.0, 10)
    voice = np.sign(np.random.randn(len(freq)))
    return create_annotation(values=freq * voice, confidence=1.0, duration=0.01, namespace="pitch_hz")


@pytest.fixture(scope="module")
def est_badmelody():
    return create_annotation(values=["a", "b", "c"], confidence=1.0, duration=0.01, namespace="pitch_hz")


@pytest.fixture(scope="module")
def ref_pattern():
    ref_jam = neojams.load("tests/fixtures/pattern_data.jams", validate=False)
    return ref_jam.annotations["pattern_jku", 0]


@pytest.fixture(scope="module")
def est_badpattern():
    pattern = {"midi_pitch": 3, "morph_pitch": 5, "staff": 1, "pattern_id": None, "occurrence_id": 1}

    return create_annotation(values=[pattern], confidence=1.0, duration=0.01, namespace="pattern_jku")


@pytest.fixture(scope="module")
def ref_hier():
    return create_hierarchy(values=["AB", "abac"])


@pytest.fixture(scope="module")
def est_hier():
    return create_hierarchy(values=["ABCD", "abacbcbd"])


@pytest.fixture(scope="module")
def est_badhier():
    return create_hierarchy(values=[[1, 2], [1, 2, 1, 3]])


@pytest.fixture(scope="module")
def ref_transcript():
    ref_jam = neojams.load("tests/fixtures/transcription_ref.jams", validate=False)
    return ref_jam.annotations["pitch_hz", 0]


@pytest.fixture(scope="module")
def est_transcript():
    est_jam = neojams.load("tests/fixtures/transcription_est.jams", validate=False)
    return est_jam.annotations["pitch_hz", 0]


@pytest.fixture(scope="module")
def est_badtranscript():
    est_jam = neojams.load("tests/fixtures/transcription_est.jams", validate=False)
    ann = est_jam.annotations["pitch_hz", 0]
    ann.append(time=2.0, duration=1.0, value=None, confidence=1)
    return ann


# Beat tracking
def test_beat_valid(ref_beat, est_beat):
    neojams.eval.beat(ref_beat, est_beat)


def test_beat_invalid(ref_beat, est_onset):
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.beat(ref_beat, est_onset)
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.beat(est_onset, ref_beat)


# Onset detection
def test_onset_valid(ref_onset, est_onset):
    neojams.eval.onset(ref_onset, est_onset)


def test_onset_invalid(ref_onset, est_beat):
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.onset(ref_onset, est_beat)
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.onset(est_beat, ref_onset)


# Chord estimation
def test_chord_valid(ref_chord, est_chord):
    neojams.eval.chord(ref_chord, est_chord)


def test_chord_noconvert(ref_chord, est_roman):
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.chord(ref_chord, est_roman)
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.chord(est_roman, ref_chord)


def test_chord_invalid(ref_chord, est_badchord):
    with pytest.raises(neojams.SchemaError):
        neojams.eval.chord(ref_chord, est_badchord)
    with pytest.raises(neojams.SchemaError):
        neojams.eval.chord(est_badchord, ref_chord)


# Segmentation
def test_segment_valid(ref_segment, est_segment):
    neojams.eval.segment(ref_segment, est_segment)


def test_segment_noconvert(ref_segment, est_chord):
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.segment(ref_segment, est_chord)
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.segment(est_chord, ref_segment)


def test_segment_invalid(ref_segment, est_segtut):
    with pytest.raises(neojams.SchemaError):
        neojams.eval.segment(ref_segment, est_segtut)
    with pytest.raises(neojams.SchemaError):
        neojams.eval.segment(est_segtut, ref_segment)


# Tempo estimation
def test_tempo_valid(ref_tempo, est_tempo):
    neojams.eval.tempo(ref_tempo, est_tempo)


def test_tempo_noconvert(ref_tempo, est_tag):
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.tempo(ref_tempo, est_tag)
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.tempo(est_tag, ref_tempo)


def test_tempo_invalid(ref_tempo, est_badtempo):
    with pytest.raises(neojams.SchemaError):
        neojams.eval.tempo(ref_tempo, est_badtempo)
    with pytest.raises(neojams.SchemaError):
        neojams.eval.tempo(est_badtempo, ref_tempo)


# Melody
def test_melody_valid(ref_melody, est_melody):
    neojams.eval.melody(ref_melody, est_melody)


def test_melody_noconvert(ref_melody, est_tag):
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.melody(ref_melody, est_tag)
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.melody(est_tag, ref_melody)


def test_melody_invalid(ref_melody, est_badmelody):
    with pytest.raises(neojams.SchemaError):
        neojams.eval.melody(ref_melody, est_badmelody)
    with pytest.raises(neojams.SchemaError):
        neojams.eval.melody(est_badmelody, ref_melody)


# Pattern discovery
def test_pattern_valid(ref_pattern):
    neojams.eval.pattern(ref_pattern, ref_pattern)


def test_pattern_noconvert(ref_pattern, est_beat):
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.pattern(ref_pattern, est_beat)
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.pattern(est_beat, ref_pattern)


def test_pattern_invalid(ref_pattern, est_badpattern):
    # Check for failure on a badly formed pattern
    with pytest.raises(neojams.SchemaError):
        neojams.eval.pattern(ref_pattern, est_badpattern)
    with pytest.raises(neojams.SchemaError):
        neojams.eval.pattern(est_badpattern, ref_pattern)


# Hierarchical segmentation
def test_hierarchy_valid(ref_hier, est_hier):
    neojams.eval.hierarchy(ref_hier, est_hier)


def test_hierarchy_noconvert(ref_hier, est_tag):
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.hierarchy(ref_hier, est_tag)
    with pytest.raises(neojams.NamespaceError):
        neojams.eval.hierarchy(est_tag, ref_hier)


def test_hierarchy_invalid(ref_hier, est_badhier):
    with pytest.raises(neojams.SchemaError):
        neojams.eval.hierarchy(ref_hier, est_badhier)
    with pytest.raises(neojams.SchemaError):
        neojams.eval.hierarchy(est_badhier, ref_hier)


def test_transcription_valid(ref_transcript, est_transcript):
    neojams.eval.transcription(ref_transcript, est_transcript)


def test_transcription_invalid(ref_transcript, est_badtranscript):
    with pytest.raises(neojams.SchemaError):
        neojams.eval.transcription(ref_transcript, est_badtranscript)
    with pytest.raises(neojams.SchemaError):
        neojams.eval.transcription(est_badtranscript, ref_transcript)
