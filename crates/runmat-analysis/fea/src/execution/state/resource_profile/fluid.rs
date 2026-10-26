use super::super::fluid::FluidStatePayloadProjection;
use super::*;
use crate::execution::{FluidSteadySolverResult, FluidSteadyWorkload};
use runmat_meshing_core::SolverMeshArtifact;
pub(super) fn profile(
    request: &SemanticStatePublicationRequest,
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    result: &FluidSteadySolverResult,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    super::super::fluid::validate_request(request, workload, mesh, result)?;
    let result_bytes = result.canonical_bytes()?;
    let topology = result.native_field_topology(workload, mesh)?;
    super::super::fluid::fluid_state_publication_workspace_bytes(workload, mesh, &result_bytes)?;
    // The state is owned typed storage built and encoded by the direct codec (no value tree):
    // charge its direct materialization, so the embedded result bytes cost their typed decode
    // rather than value-tree nodes.
    profile_state_payload_direct(
        request,
        &Tagged {
            kind: "fluid",
            key: "state",
            value: &FluidStatePayloadProjection {
                workload,
                mesh,
                result_bytes: &result_bytes,
                topology: &topology,
            },
        },
    )
}
