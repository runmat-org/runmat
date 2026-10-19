use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
use super::*;
use crate::planning::resource::StructuralFieldRecoveryResourceDemand;
pub(super) fn build(
    input: FeaResourcePlanningInput,
    demand: &StructuralFieldRecoveryResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.constraint_compilation_peak_bytes.0 == 0
        || demand.construction_workspace_bytes.0 == 0
        || demand.constraint_plan_digest == Digest::sha256([])
        || !input.worker.accelerators.is_empty()
    {
        return Err(ResourcePlanningError::Invalid(
            "original structural field recovery requires complete host/compiler demand".into(),
        ));
    }
    let retained = ByteCount(add(
        demand.inputs.transport_closure_bytes.0,
        implicit_resource_plan_input_facts()?
            .transport_closure_bytes
            .0,
        "structural recovery retained input",
    )?);
    // Original compilation is released before field projection/partition construction.
    let workspace = ByteCount(
        demand
            .constraint_compilation_peak_bytes
            .0
            .max(demand.construction_workspace_bytes.0),
    );
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
