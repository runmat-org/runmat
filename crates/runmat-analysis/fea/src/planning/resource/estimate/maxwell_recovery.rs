use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::MaxwellFieldRecoveryResourceDemand;
pub(super) fn build(
    input: FeaResourcePlanningInput,
    demand: &MaxwellFieldRecoveryResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.construction_workspace_bytes.0 == 0
        || demand.source_validation_workspace_bytes.0 == 0
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "maxwell field recovery needs complete host construction workspace".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "maxwell retained input transport",
    )?);
    // Source validation creates projection/property uniqueness sets, but no numerical solve.
    let workspace = ByteCount(add(
        demand.construction_workspace_bytes.0,
        add(
            demand.source_validation_workspace_bytes.0,
            demand.inputs.decoded_resident_bytes_admission_budget.0,
            "maxwell validation canonical identity materialization",
        )?,
        "maxwell source validation and retained field construction",
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
