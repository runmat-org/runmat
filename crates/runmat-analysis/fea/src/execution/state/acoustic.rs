//! A completed topology-bound acoustic phasor state from one harmonic solve.

use runmat_analysis_core::{FieldLocation, Frequency, ObservationQuantity, ObservationValueShape};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::{Deserialize, Serialize};

use crate::results::field::{
    RecoveredFieldArtifact, RecoveredFieldDerivation, RecoveredFieldEntityDomain,
    RecoveredFieldNumericRepresentation, RecoveredFieldSample,
};

use super::{
    field_validation::validate_portable_field, AnalysisExecutionError, AnalysisExecutionResult,
    CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
    SemanticStatePublicationRequest, SolverResultIdentity, StepIdentity,
};
use crate::execution::{
    AcousticHarmonicSolverResult, AcousticHarmonicWorkload, FieldRecoveryRequest,
    FieldRecoverySource, FieldTopologyIdentity,
};

#[cfg(test)]
mod tests;

/// RunMat's acoustic phasor convention: physical values are the real part of `p exp(+i wt)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticPhasorConvention {
    PositiveAngularFrequency,
}

/// Native acoustic state at one authored frequency.
///
/// Pressure is nodal. Particle velocity is the canonical kernel's quadrature-volume average for
/// each volume element. `fields` is the shared recovered-field authority, not a second copy with
/// a private acoustic field schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalAcousticStatePayload {
    pub frequency: Frequency,
    pub phasor_convention: AcousticPhasorConvention,
    pub ordered_node_ids: Vec<u64>,
    pub ordered_element_ids: Vec<u64>,
    pub recovery_request: FieldRecoveryRequest,
    pub fields: RecoveredFieldArtifact,
}

impl CanonicalAcousticStatePayload {
    pub fn validate_for(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
        producer: StepIdentity,
    ) -> AnalysisExecutionResult<()> {
        self.recovery_request.validate()?;
        self.fields
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let [pressure, velocity, intensity, port] = self.fields.fields() else {
            return Err(invalid(
                "acoustic state requires exactly pressure, particle-velocity, intensity and port-intensity fields",
            ));
        };
        validate_portable_field(pressure.storage())?;
        validate_portable_field(velocity.storage())?;
        validate_portable_field(intensity.storage())?;
        validate_portable_field(port.storage())?;
        let expected_plan = crate::execution::acoustic::native_acoustic_field_plan(
            self.ordered_node_ids.len(),
            self.ordered_element_ids.len(),
        )?;
        if self.recovery_request.plan != expected_plan {
            return Err(invalid(
                "acoustic state recovery plan differs from the canonical native inventory",
            ));
        }
        self.recovery_request.plan.validate_definitions(
            self.fields
                .fields()
                .iter()
                .map(|field| field.definition().clone()),
        )?;
        let unique_nonzero = |ids: &[u64]| {
            !ids.is_empty()
                && !ids.contains(&0)
                && ids.iter().collect::<std::collections::BTreeSet<_>>().len() == ids.len()
        };
        let sample = RecoveredFieldSample::FrequencyHertz(self.frequency.canonical_value());
        if !self.frequency.canonical_value().is_finite()
            || self.frequency.canonical_value() <= 0.0
            || self.phasor_convention != AcousticPhasorConvention::PositiveAngularFrequency
            || !unique_nonzero(&self.ordered_node_ids)
            || !unique_nonzero(&self.ordered_element_ids)
            || self.fields.topology_digest != topology_digest
            || self.fields.source != FieldRecoverySource::SolverResult(source_result)
            || self.recovery_request.producer != producer
            || self.recovery_request.source != FieldRecoverySource::SolverResult(source_result)
            || !matches!(
                self.recovery_request.topology,
                FieldTopologyIdentity::VolumeMesh { mesh_digest }
                    if Digest::from_bytes(*mesh_digest.bytes()) == topology_digest
            )
            || self.recovery_request.identity() != self.fields.recovery_request
            || self.fields.entity_counts.nodes != self.ordered_node_ids.len()
            || self.fields.entity_counts.volume_elements != self.ordered_element_ids.len()
            || pressure.sample() != sample
            || velocity.sample() != sample
            || !matches_native(
                pressure,
                ObservationQuantity::AcousticPressure,
                FieldLocation::Node,
                RecoveredFieldEntityDomain::Nodes,
                ObservationValueShape::Scalar,
            )
            || !matches_native(
                velocity,
                ObservationQuantity::AcousticParticleVelocity,
                FieldLocation::Element,
                RecoveredFieldEntityDomain::VolumeElements,
                ObservationValueShape::Vector3,
            )
        {
            return Err(invalid(
                "acoustic state fields, ordering, frequency, topology or source do not agree",
            ));
        }
        Ok(())
    }
}

fn matches_native(
    field: &crate::results::field::RecoveredField,
    quantity: ObservationQuantity,
    location: FieldLocation,
    domain: RecoveredFieldEntityDomain,
    shape: ObservationValueShape,
) -> bool {
    let definition = field.definition();
    definition.quantity() == quantity
        && definition.location() == location
        && definition.entity_domain() == domain
        && definition.value_shape() == &shape
        && definition.numeric_representation()
            == RecoveredFieldNumericRepresentation::ComplexInterleaved
        && definition.derivation() == RecoveredFieldDerivation::Native
}

impl CanonicalSemanticState {
    /// Publish the exact native state from one completed acoustic frequency solve.
    pub fn from_acoustic_result(
        workload: &AcousticHarmonicWorkload,
        mesh: &SolverMeshArtifact,
        result: &AcousticHarmonicSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        mesh.validate_canonical()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        result.validate_for(workload, mesh)?;
        let requests = result.state_publication_requests()?;
        let [request] = requests.as_slice() else {
            return Err(invalid(
                "acoustic state requires one unambiguous persisted authored declaration",
            ));
        };
        Self::from_acoustic_publication(request, result)
    }

    /// Detached publication consumes authenticated completed fields, never another solve.
    pub fn from_acoustic_publication(
        request: &SemanticStatePublicationRequest,
        result: &AcousticHarmonicSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_publication_request(request)?;
        let recovery_request = result.native_field_request_from_result()?;
        let fields = result.recover_native_fields_from_result(&recovery_request)?;
        let payload = CanonicalAcousticStatePayload {
            frequency: result.frequency(),
            phasor_convention: AcousticPhasorConvention::PositiveAngularFrequency,
            ordered_node_ids: result.ordered_node_ids().to_vec(),
            ordered_element_ids: result.ordered_element_ids().to_vec(),
            recovery_request,
            fields,
        };
        payload.validate_for(
            request.topology_digest,
            request.source_result,
            request.producer,
        )?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::Acoustic {
                state: Box::new(payload),
            },
        })
    }
}

fn invalid(reason: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(reason.into())
}
