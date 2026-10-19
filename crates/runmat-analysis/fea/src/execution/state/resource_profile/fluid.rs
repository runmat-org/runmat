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
    super::super::fluid::fluid_state_publication_workspace_bytes(
        workload,
        mesh,
        &result_bytes,
        &topology,
    )?;
    profile_state_payload(
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
