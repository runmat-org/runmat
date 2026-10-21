use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::AcousticFieldRecoveryResourceDemand;
pub(super) fn build(
    input: FeaResourcePlanningInput,
    demand: &AcousticFieldRecoveryResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.construction_workspace_bytes.0 == 0
        || demand.source_validation_workspace_bytes.0 == 0
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "acoustic field recovery needs complete host construction workspace".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "acoustic retained input transport",
    )?);
    let workspace = ByteCount(add(
        demand.construction_workspace_bytes.0,
        add(
            demand.source_validation_workspace_bytes.0,
            demand.inputs.decoded_resident_bytes_admission_budget.0,
            "acoustic validation canonical identity materialization",
        )?,
        "acoustic source validation and retained field construction",
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
