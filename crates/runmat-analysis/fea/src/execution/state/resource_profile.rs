//! Borrowed publication encoding projections for producer-side admission metadata.
//!
//! These projections never construct a semantic state or clone its numerical arrays. They
//! bound canonical output construction, not the complete worker working set. Callers must
//! separately account source decoding, retained transport, typed state storage and validation.
use runmat_canonical_codec::{profile_serialized, SerializedCanonicalProfile};
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

use super::*;
use crate::execution::{
    StructuralModalResponseSpectrumSolverResult, StructuralRandomResponseSolverResult,
    ThermalSolverResultManifest,
};
use crate::pipeline::{
    modal_basis::ModalBasisArtifact, structural_linearization::StructuralLinearizationArtifact,
};

#[cfg(test)]
mod tests;
mod thermal;

/// Borrowed data already held by a producing worker. Structural callers must not decode a
/// complete result on the driver solely to obtain this projection. Thermal requires metadata
/// only; no field partitions are consumed.
pub enum SemanticStatePublicationProfileSource<'a> {
    StructuralPrimaryFields(&'a [AnalysisField]),
    StructuralLinearization(&'a StructuralLinearizationArtifact),
    ModalBasis(&'a ModalBasisArtifact),
    StructuralResponseEnvelope(&'a StructuralModalResponseSpectrumSolverResult),
    StructuralRandomStatistics(&'a StructuralRandomResponseSolverResult),
    Thermal(&'a ThermalSolverResultManifest),
}

impl SemanticStatePublicationRequest {
    /// Project the published wire shape, without computing state identity or publishing it.
    /// This does not replace source validation or bind a future resource plan by itself.
    pub fn encoding_profile(
        &self,
        source: SemanticStatePublicationProfileSource<'_>,
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        self.validate()?;
        match (self.source, source) {
            (
                SemanticStatePublicationSource::ThermalPrimaryField,
                SemanticStatePublicationProfileSource::Thermal(manifest),
            ) => thermal::profile(self, manifest),
            (
                SemanticStatePublicationSource::StructuralPrimaryFields,
                SemanticStatePublicationProfileSource::StructuralPrimaryFields(fields),
            ) => {
                let selected = select_structural_state_fields(self.declaration.kind, fields)?;
                profile_state_payload(
                    self,
                    &Tagged {
                        kind: "fields",
                        key: "fields",
                        value: &selected,
                    },
                )
            }
            (
                SemanticStatePublicationSource::StructuralLinearization,
                SemanticStatePublicationProfileSource::StructuralLinearization(value),
            ) => {
                if value.topology_digest() != self.topology_digest {
                    return Err(invalid("state profile linearization topology differs"));
                }
                profile_state_payload(
                    self,
                    &Tagged {
                        kind: "structural_linearization",
                        key: "linearization",
                        value,
                    },
                )
            }
            (
                SemanticStatePublicationSource::ModalBasis,
                SemanticStatePublicationProfileSource::ModalBasis(value),
            ) => profile_state_payload(
                self,
                &Tagged {
                    kind: "modal_basis",
                    key: "basis",
                    value,
                },
            ),
            (
                SemanticStatePublicationSource::StructuralResponseEnvelope,
                SemanticStatePublicationProfileSource::StructuralResponseEnvelope(value),
            ) => {
                if value.assembly().scope().topology_digest() != self.topology_digest {
                    return Err(invalid("state profile envelope topology differs"));
                }
                profile_state_payload(
                    self,
                    &Tagged {
                        kind: "structural_response_envelope",
                        key: "envelope",
                        value: &Envelope(value),
                    },
                )
            }
            (
                SemanticStatePublicationSource::StructuralRandomStatistics,
                SemanticStatePublicationProfileSource::StructuralRandomStatistics(value),
            ) => {
                if value.assembly().scope().topology_digest() != self.topology_digest {
                    return Err(invalid("state profile statistics topology differs"));
                }
                profile_state_payload(
                    self,
                    &Tagged {
                        kind: "structural_random_statistics",
                        key: "statistics",
                        value: &Statistics(value),
                    },
                )
            }
            _ => Err(invalid(
                "state profile source differs from publication source",
            )),
        }
    }
}

fn invalid(reason: &str) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(reason.into())
}

fn profile_state_payload<T: Serialize + ?Sized>(
    request: &SemanticStatePublicationRequest,
    payload: &T,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    Ok(profile_serialized(
        &StateProjection { request, payload },
        STATE_LIMITS,
    )?)
}

struct StateProjection<'a, T: ?Sized> {
    request: &'a SemanticStatePublicationRequest,
    payload: &'a T,
}

impl<T: Serialize + ?Sized> Serialize for StateProjection<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(11))?;
        map.serialize_entry("schema_version", &1_u16)?;
        map.serialize_entry("identity", &Digest::sha256([]))?;
        map.serialize_entry("content_digest", &Digest::sha256([]))?;
        map.serialize_entry("producer", &self.request.producer)?;
        map.serialize_entry("state_id", &self.request.declaration.state_id)?;
        map.serialize_entry("kind", &self.request.declaration.kind)?;
        map.serialize_entry("completeness", &self.request.declaration.completeness)?;
        map.serialize_entry("persistence", &self.request.declaration.persistence)?;
        map.serialize_entry("topology_digest", &self.request.topology_digest)?;
        map.serialize_entry("source_result", &self.request.source_result)?;
        map.serialize_entry("payload", self.payload)?;
        map.end()
    }
}

struct Tagged<'a, T: ?Sized> {
    kind: &'static str,
    key: &'static str,
    value: &'a T,
}

impl<T: Serialize + ?Sized> Serialize for Tagged<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("kind", self.kind)?;
        map.serialize_entry(self.key, self.value)?;
        map.end()
    }
}

struct Envelope<'a>(&'a StructuralModalResponseSpectrumSolverResult);
impl Serialize for Envelope<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("scope", self.0.assembly().scope())?;
        map.serialize_entry("response", &self.0.numerical().response())?;
        map.serialize_entry("semantics", &self.0.numerical().semantics())?;
        map.serialize_entry("values", self.0.numerical().combined_response())?;
        map.end()
    }
}

struct Statistics<'a>(&'a StructuralRandomResponseSolverResult);
impl Serialize for Statistics<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("scope", self.0.assembly().scope())?;
        map.serialize_entry("numerical", self.0.numerical())?;
        map.end()
    }
}
