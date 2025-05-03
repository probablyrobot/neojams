#!/usr/bin/env python
# CREATED:2023-06-12 15:12:15 by User <user@example.com>
"""Basic functionality tests"""


import neojams
from neojams import Annotation


def test_create_annotation():
    """Test creating an annotation object"""
    
    ann = Annotation(namespace='tag_open')
    ann.append(time=0, duration=1, value='foo', confidence=1.0)
    
    assert ann.namespace == 'tag_open'
    assert len(ann.data) == 1
    assert ann.data[0].time == 0
    assert ann.data[0].duration == 1
    assert ann.data[0].value == 'foo'
    assert ann.data[0].confidence == 1.0


def test_create_jams():
    """Test creating a JAMS object"""
    
    jam = neojams.JAMS()
    
    ann = Annotation(namespace='tag_open')
    ann.append(time=0, duration=1, value='foo', confidence=1.0)
    
    jam.annotations.append(ann)
    
    assert len(jam.annotations) == 1
    assert jam.annotations[0].namespace == 'tag_open' 