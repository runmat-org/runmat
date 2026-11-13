use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::NonlinearFieldRecoveryResourceDemand;
pub(super) fn build(
    input: FeaResourcePlanningInput,
    demand: &NonlinearFieldRecoveryResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.construction_workspace_bytes.0 == 0
        || demand.recovery_workspace_bytes.0 == 0
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "nonlinear field recovery needs complete host construction workspace".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "nonlinear retained input transport",
    )?);
    // Conservative original recovery includes physical basis and retained original result buffers; no equilibrium solve.
    let workspace = ByteCount(add(
        demand.construction_workspace_bytes.0,
        add(
            demand.recovery_workspace_bytes.0,
            demand.inputs.decoded_resident_bytes_admission_budget.0,
            "nonlinear validation canonical identity materialization",
        )?,
        "nonlinear source validation and retained field construction",
    )?);
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            workspace,
            retained,
            &[&demand.output],
        ),
    )
}
