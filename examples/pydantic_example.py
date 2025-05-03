#!/usr/bin/env python
"""
Example script demonstrating the use of Pydantic models with NeoJAMS.

This script shows how to create and validate JAMS objects using Pydantic models.
"""

import json

from neojams import AnnotationMetadataModel, AnnotationModel, CuratorModel, JAMSModel, ObservationModel


def main():
    """Example of using Pydantic models for JAMS data."""
    # Create a curator with validation
    curator = CuratorModel(name="Example Curator", email="curator@example.com")

    # Create some observations with validation
    observations = [
        ObservationModel(time=1.0, duration=1.0, value="C:maj", confidence=0.9),
        ObservationModel(time=2.0, duration=1.0, value="G:maj", confidence=0.85),
        ObservationModel(time=3.0, duration=1.0, value="F:maj", confidence=0.75),
    ]

    # Create an annotation with metadata
    annotation = AnnotationModel(
        namespace="chord",
        data=observations,
        time=0.0,
        duration=4.0,
        annotation_metadata=AnnotationMetadataModel(
            curator=curator,
            version="1.0",
            corpus="example",
            annotation_tools="Pydantic example",
            data_source="manual annotation",
        ),
    )

    # Create a complete JAMS object
    jams = JAMSModel(
        annotations=[annotation], file_metadata={"title": "Example Song", "artist": "Example Artist", "duration": 4.0}
    )

    # Validate and convert to dict
    jams_dict = jams.model_dump()

    # Pretty print the resulting JSON
    print(json.dumps(jams_dict, indent=2))

    print("\nValidation successful!")
    print(f"Created JAMS object with {len(jams.annotations)} annotation(s)")
    print(f"First annotation has {len(jams.annotations[0].data)} observation(s)")

    # Example of validation error
    try:
        # This will fail because duration cannot be negative
        ObservationModel(time=1.0, duration=-1.0, value="D:min")
    except Exception as e:
        print(f"\nValidation error example: {e}")


if __name__ == "__main__":
    main()
