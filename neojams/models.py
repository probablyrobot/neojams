"""
Pydantic models for NeoJAMS
---------------------------

This module contains Pydantic models that represent the core NeoJAMS data structures
with proper type checking and validation.
"""

from typing import Any, ClassVar, Tuple

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Pydantic model for an Observation in JAMS."""

    time: float = Field(..., description="The time of the observation in seconds", ge=0)
    duration: float = Field(..., description="The duration of the observation in seconds", ge=0)
    value: Any = Field(..., description="The value of the observation")
    confidence: float | None = Field(None, description="Confidence value", ge=0, le=1)

    # Provide compatibility with namedtuple interface used in legacy core code
    _fields: ClassVar[Tuple[str, ...]] = ("time", "duration", "value", "confidence")

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "arbitrary_types_allowed": True,
        "from_attributes": True,
    }

    def __json_light__(self) -> dict:
        """Return a lightweight JSON representation of the observation."""
        return self.model_dump()

    def __getstate__(self) -> dict:
        """Return the state for pickling (used by pickle)."""
        return self.model_dump()

    # Maintain compatibility with namedtuple _asdict method expected elsewhere
    def _asdict(self) -> dict:  # noqa: D401
        """Return a dictionary representation of the observation."""
        return self.model_dump()

    def model_dump(self) -> dict:
        """Return a dictionary representation of the observation."""
        return {"time": self.time, "duration": self.duration, "value": self.value, "confidence": self.confidence}

    def __getattr__(self, name: str) -> Any:
        """Handle attribute access for compatibility."""
        if name == "model_dump":
            return self.model_dump
        return super().__getattr__(name)


class Sandbox(BaseModel):
    """Pydantic model for unconstrained Sandbox data."""

    model_config = {"extra": "allow"}


class Curator(BaseModel):
    """Pydantic model for a Curator."""

    name: str = Field("", description="Name of the curator")
    email: str = Field("", description="Email address of the curator")


class AnnotationMetadata(BaseModel):
    """Pydantic model for Annotation Metadata."""

    curator: Curator = Field(default_factory=Curator, description="Curator information")
    version: str = Field("", description="Version of this annotation")
    corpus: str = Field("", description="Collection assignment")
    annotator: dict[str, Any] | None = Field(default_factory=dict, description="Information about the annotator")
    annotation_tools: str = Field("", description="Description of the tools used")
    annotation_rules: str = Field("", description="Description of the annotation rules")
    validation: str = Field("", description="Methods for validation")
    data_source: str = Field("", description="Where the data originated from")


class FileMetadata(BaseModel):
    """Pydantic model for File Metadata."""

    title: str = Field("", description="Name of the recording")
    artist: str = Field("", description="Name of the artist/musician")
    release: str = Field("", description="Name of the release")
    duration: float | None = Field(None, description="Duration in seconds", ge=0)
    identifiers: dict[str, Any] = Field(default_factory=dict, description="Identifier keys (e.g., musicbrainz ids)")
    jams_version: str = Field("0.1.0", description="Version of the JAMS Schema")


class Annotation(BaseModel):
    """Pydantic model for an Annotation."""

    namespace: str = Field(..., description="The namespace for this annotation")
    data: list[Observation] = Field(default_factory=list, description="The observation data")
    annotation_metadata: AnnotationMetadata = Field(
        default_factory=AnnotationMetadata, description="Metadata corresponding to this annotation"
    )
    sandbox: Sandbox = Field(default_factory=Sandbox, description="Miscellaneous information")
    time: float = Field(0, description="The starting time for this annotation", ge=0)
    duration: float | None = Field(None, description="The duration of this annotation", ge=0)


class JAMS(BaseModel):
    """Pydantic model for a top-level JAMS object."""

    annotations: list[Annotation] = Field(default_factory=list, description="List of annotations")
    file_metadata: FileMetadata = Field(default_factory=FileMetadata, description="Metadata for the audio file")
    sandbox: Sandbox = Field(default_factory=Sandbox, description="Unconstrained global sandbox")
