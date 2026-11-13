//! Borrowed publication encoding projections for producer-side admission metadata.
//!
//! These projections never construct a semantic state or clone its numerical arrays. They
//! bound canonical output construction, not the complete worker working set. Callers must
//! separately account source decoding, retained transport, typed state storage and validation.
use runmat_analysis_core::AnalysisField;
use runmat_canonical_codec::{profile_serialized, SerializedCanonicalProfile};
use serde::ser::{SerializeMap, SerializeStruct};
use serde::{Serialize, Serializer};

use super::*;
use crate::execution::{
    StructuralModalResponseSpectrumSolverResult, StructuralRandomResponseSolverResult,
    ThermalSolverResultManifest,
};
use crate::pipeline::{
    modal_basis::ModalBasisArtifact, structural_linearization::StructuralLinearizationArtifact,
};

mod acoustic;
mod electrical;
pub(super) mod fluid;
mod magnetic;
mod maxwell;
#[cfg(test)]
mod tests;
mod thermal;

/// Borrowed data already held by a producing worker. Structural callers must not decode a
/// complete result on the driver solely to obtain this projection. Thermal requires metadata
/// only; no field partitions are consumed.
pub enum SemanticStatePublicationProfileSource<'a> {
    NonlinearSolidOriginalState {
        workload: &'a crate::execution::NonlinearSolidWorkload,
        block: &'a crate::execution::NonlinearSolidBlockArtifact,
    },
    FluidPrimaryFields {
        workload: &'a crate::execution::FluidSteadyWorkload,
        mesh: &'a runmat_meshing_core::SolverMeshArtifact,
        result: &'a crate::execution::FluidSteadySolverResult,
    },
    StructuralPrimaryFields(&'a [AnalysisField]),
    StructuralLinearization(&'a StructuralLinearizationArtifact),
    ModalBasis(&'a ModalBasisArtifact),
    StructuralResponseEnvelope(&'a StructuralModalResponseSpectrumSolverResult),
    StructuralRandomStatistics(&'a StructuralRandomResponseSolverResult),
    Thermal(&'a ThermalSolverResultManifest),
    AcousticNativeFields(&'a crate::execution::AcousticHarmonicSolverResult),
    /// Exact portable fields of an acoustic cavity-mode or pressure-transient state.
    AcousticResponseFields(&'a [AnalysisField]),
    StaticElectricalNativeFields(&'a crate::execution::ElectricalStaticSolverResult),
    MagnetostaticNativeFields(&'a crate::execution::MagnetostaticSolverResult),
    MaxwellNativeFields(&'a crate::execution::MaxwellHarmonicSolverResult),
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
                SemanticStatePublicationSource::NonlinearSolidOriginalState,
                SemanticStatePublicationProfileSource::NonlinearSolidOriginalState {
                    workload,
                    block,
                },
            ) => {
                super::nonlinear::validate_request(self, workload, block)?;
                #[derive(Serialize)]
                struct Projection<'a> {
                    workload: &'a crate::execution::NonlinearSolidWorkload,
                    block: &'a crate::execution::NonlinearSolidBlockArtifact,
                }
                profile_state_payload(
                    self,
                    &Tagged {
                        kind: "nonlinear_solid",
                        key: "state",
                        value: &Projection { workload, block },
                    },
                )
            }
            (
                SemanticStatePublicationSource::FluidPrimaryFields,
                SemanticStatePublicationProfileSource::FluidPrimaryFields {
                    workload,
                    mesh,
                    result,
                },
            ) => fluid::profile(self, workload, mesh, result),
            (
                SemanticStatePublicationSource::MaxwellNativeFields,
                SemanticStatePublicationProfileSource::MaxwellNativeFields(result),
            ) => maxwell::profile(self, result),
            (
                SemanticStatePublicationSource::MagnetostaticNativeFields,
                SemanticStatePublicationProfileSource::MagnetostaticNativeFields(result),
            ) => magnetic::profile(self, result),
            (
                SemanticStatePublicationSource::StaticElectricalNativeFields,
                SemanticStatePublicationProfileSource::StaticElectricalNativeFields(result),
            ) => electrical::profile(self, result),
            (
                SemanticStatePublicationSource::AcousticNativeFields,
                SemanticStatePublicationProfileSource::AcousticNativeFields(result),
            ) => acoustic::profile(self, result),
            (
                SemanticStatePublicationSource::ThermalPrimaryField,
                SemanticStatePublicationProfileSource::Thermal(manifest),
            ) => thermal::profile(self, manifest),
            (
                SemanticStatePublicationSource::AcousticResponseFields,
                SemanticStatePublicationProfileSource::AcousticResponseFields(fields),
            ) => profile_state_payload(
                self,
                &Tagged {
                    kind: "fields",
                    key: "fields",
                    value: fields,
                },
            ),
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

fn profile_state_payload_direct<T: Serialize + ?Sized>(
    request: &SemanticStatePublicationRequest,
    payload: &T,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    Ok(runmat_canonical_codec::profile_direct_materialization(
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
        let mut state = serializer.serialize_struct("CanonicalSemanticState", 11)?;
        state.serialize_field("schema_version", &1_u16)?;
        state.serialize_field("identity", &Digest::sha256([]))?;
        state.serialize_field("content_digest", &Digest::sha256([]))?;
        state.serialize_field("producer", &self.request.producer)?;
        state.serialize_field("state_id", &self.request.declaration.state_id)?;
        state.serialize_field("kind", &self.request.declaration.kind)?;
        state.serialize_field("completeness", &self.request.declaration.completeness)?;
        state.serialize_field("persistence", &self.request.declaration.persistence)?;
        state.serialize_field("topology_digest", &self.request.topology_digest)?;
        state.serialize_field("source_result", &self.request.source_result)?;
        state.serialize_field("payload", self.payload)?;
        state.end()
    }
}

struct Tagged<'a, T: ?Sized> {
    kind: &'static str,
    key: &'static str,
    value: &'a T,
}

impl<T: Serialize + ?Sized> Serialize for Tagged<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut variant = serializer.serialize_struct("CanonicalSemanticStatePayload", 2)?;
        variant.serialize_field("kind", self.kind)?;
        variant.serialize_field(self.key, self.value)?;
        variant.end()
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
