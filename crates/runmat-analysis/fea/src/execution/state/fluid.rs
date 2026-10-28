//! Complete original steady finite-volume state, not a restart/resumption claim.
//! U [m/s], p [Pa], and owner-oriented q [m³/s] remain in the immutable original
//! canonical result bytes; exact source authority/layout is retained alongside them.
//! Publication does not solve, interpolate, or duplicate the numeric field arrays.
use super::*;
use crate::{
    execution::{FluidSteadySolverResult, FluidSteadyWorkload},
    results::field::FiniteVolumeFieldTopology,
};
use runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate;
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidStatePayload {
    pub workload: Box<FluidSteadyWorkload>,
    pub mesh: Box<SolverMeshArtifact>,
    pub result_bytes: Vec<u8>,
    pub topology: FiniteVolumeFieldTopology,
}

/// Borrowed exact payload shape, shared by cold/source residency and state wire profiling.
#[derive(Serialize)]
pub(super) struct FluidStatePayloadProjection<'a> {
    pub workload: &'a FluidSteadyWorkload,
    pub mesh: &'a SolverMeshArtifact,
    pub result_bytes: &'a [u8],
    pub topology: &'a FiniteVolumeFieldTopology,
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
fn add(a: u64, b: u64) -> AnalysisExecutionResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("fluid state requested heap overflow"))
}

/// Source-bound validation of the retained original result: the verification phase of the
/// original source (re-preparation and residual re-evaluation, no linear solve) plus the
/// directly decoded original result beside its canonical bytes. The owned state payload is
/// not charged here: the shared publication phase charges it once, as the state entry's
/// direct materialization inside output construction. This is requested heap, not
/// scheduler/RSS evidence.
pub fn fluid_state_publication_workspace_bytes(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    result_bytes: &[u8],
) -> AnalysisExecutionResult<u64> {
    let projection =
        crate::planning::resource::derive_fluid_steady_resource_projection(workload, mesh)
            .map_err(invalid)?;
    let decoded_original = canonical_payload_typed_resident_bytes_estimate(result_bytes)?;
    let required = add(
        projection.verification_phase_resident_bytes_upper_bound.0,
        decoded_original,
    )?;
    if required > workload.resident_ceiling()? || required > isize::MAX as u64 {
        return Err(invalid(
            "fluid complete state requested heap exceeds original authored/numerical ceiling",
        ));
    }
    Ok(required)
}

impl CanonicalFluidStatePayload {
    /// Source-bound cold decode and original-equation validation; never invokes solve.
    pub fn original_result(&self) -> AnalysisExecutionResult<FluidSteadySolverResult> {
        fluid_state_publication_workspace_bytes(&self.workload, &self.mesh, &self.result_bytes)?;
        FluidSteadySolverResult::from_canonical_bytes_for(
            &self.result_bytes,
            Digest::sha256(&self.result_bytes),
            &self.workload,
            &self.mesh,
        )
    }
    pub fn validate_for(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
        producer: StepIdentity,
    ) -> AnalysisExecutionResult<()> {
        let result = self.original_result()?;
        let original_topology = result.native_field_topology(&self.workload, &self.mesh)?;
        if result.identity()? != source_result
            || result.producer_identity() != producer
            || self.topology != original_topology
            || self.topology.identity().discretization_digest() != topology_digest
        {
            return Err(invalid(
                "fluid state differs from original producer/result/owner-oriented layout",
            ));
        }
        Ok(())
    }
}

pub fn fluid_state_publication_requests(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    result: &FluidSteadySolverResult,
) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
    result.validate_for(workload, mesh)?;
    let topology = result.native_field_topology(workload, mesh)?;
    requests_on(workload, result, &topology)
}

/// The declared publication requests of a result whose `validate_for` succeeded, on its native
/// field topology.
fn requests_on(
    workload: &FluidSteadyWorkload,
    result: &FluidSteadySolverResult,
    topology: &FiniteVolumeFieldTopology,
) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
    workload.model.sequence.steps[0]
        .state_outputs
        .iter()
        .cloned()
        .map(|declaration| {
            SemanticStatePublicationRequest::new(
                SemanticStatePublicationSource::FluidPrimaryFields,
                result.producer_identity(),
                declaration,
                topology.identity().discretization_digest(),
                result.identity()?,
            )
        })
        .collect()
}

pub(super) fn validate_request(
    request: &SemanticStatePublicationRequest,
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    result: &FluidSteadySolverResult,
) -> AnalysisExecutionResult<()> {
    request.validate()?;
    if !fluid_state_publication_requests(workload, mesh, result)?.contains(request) {
        return Err(invalid(
            "fluid publication differs from original complete authored declaration",
        ));
    }
    Ok(())
}

/// The size certificate of exactly one declared fluid state publication, bound to the exact
/// solver payload. Each state publication profiles only its own state, never the others.
pub fn fluid_state_publication_profile(
    request: &SemanticStatePublicationRequest,
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    result: &FluidSteadySolverResult,
    source_payload_digest: Digest,
) -> AnalysisExecutionResult<StatePublicationProfiles> {
    let bytes = result.canonical_bytes()?;
    if source_payload_digest != Digest::sha256(&bytes) {
        return Err(invalid(
            "fluid state profile source payload differs from original immutable result",
        ));
    }
    request.validate()?;
    result.validate_for(workload, mesh)?;
    let topology = result.native_field_topology(workload, mesh)?;
    if !requests_on(workload, result, &topology)?.contains(request) {
        return Err(invalid(
            "fluid publication differs from original complete authored declaration",
        ));
    }
    let profile = request.encoding_profile(
        SemanticStatePublicationProfileSource::FluidPrimaryFields {
            workload,
            mesh,
            result,
        },
    )?;
    StatePublicationProfiles::new(
        result.producer_identity(),
        result.identity()?,
        topology.identity().discretization_digest(),
        source_payload_digest,
        vec![StatePublicationProfileEntry::new(request.clone(), profile)?],
    )
}

impl CanonicalSemanticState {
    pub fn from_fluid_publication(
        request: &SemanticStatePublicationRequest,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
        result: &FluidSteadySolverResult,
    ) -> AnalysisExecutionResult<Self> {
        validate_request(request, workload, mesh, result)?;
        let result_bytes = result.canonical_bytes()?;
        let topology = result.native_field_topology(workload, mesh)?;
        fluid_state_publication_workspace_bytes(workload, mesh, &result_bytes)?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::Fluid {
                state: Box::new(CanonicalFluidStatePayload {
                    workload: Box::new(workload.clone()),
                    mesh: Box::new(mesh.clone()),
                    result_bytes,
                    topology,
                }),
            },
        })
    }
}

#[cfg(test)]
mod tests;
