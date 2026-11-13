use super::*;
use crate::planning::resource::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedFluidSteadyResourceFacts {
    pub local_execution: LocalExecutionPlan,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}

pub fn fluid_steady_result_payload_bytes_upper_bound(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<u64, ResourcePlanningError> {
    Ok(derive_fluid_steady_resource_projection(workload, mesh)?
        .result_payload_bytes_upper_bound
        .0)
}

pub(super) fn worker(
    workload: &FluidSteadyWorkload,
    local_execution: LocalExecutionPlan,
) -> Result<WorkerResourcePolicy, ResourcePlanningError> {
    Ok(WorkerResourcePolicy {
        cpu_millicores: local_execution.checked_cpu_millicores().map_err(invalid)?,
        maximum_wall_milliseconds: workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_wall_time_milliseconds,
        fixed_memory_bytes: ByteCount(16 * 1024 * 1024),
        fixed_scratch_bytes: ByteCount(8 * 1024 * 1024),
        required_capabilities: Default::default(),
        accelerators: Vec::new(),
    })
}
pub(super) fn ceiling(
    workload: &FluidSteadyWorkload,
) -> Result<runmat_analysis_core::AccuracyResourceCeiling, ResourcePlanningError> {
    let mut ceiling = workload.model.sequence.steps[0].accuracy.resources;
    ceiling.set_resident_byte_budget(workload.resident_ceiling().map_err(invalid)?);
    Ok(ceiling)
}

pub fn build_fluid_steady_resource_plan(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    prepared: PreparedFluidSteadyResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    build_fluid_steady_resource_plan_with(workload, mesh, |_| Ok(prepared))
}

/// Derive the resource projection of `workload` on `mesh` once, let the caller size its
/// prepared facts from it, and build the plan on that same projection.
pub fn build_fluid_steady_resource_plan_with<E: From<ResourcePlanningError>>(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    prepare: impl FnOnce(&FluidSteadyResourceProjection) -> Result<PreparedFluidSteadyResourceFacts, E>,
) -> Result<FeaResourcePlan, E> {
    let projection = derive_fluid_steady_resource_projection(workload, mesh)?;
    let prepared = prepare(&projection)?;
    Ok(build_on_projection(workload, projection, prepared)?)
}

fn build_on_projection(
    workload: &FluidSteadyWorkload,
    projection: FluidSteadyResourceProjection,
    prepared: PreparedFluidSteadyResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let receipt_profile = crate::execution::fluid_receipt_encoding_profile().map_err(invalid)?;
    validate_fluid_steady_prepared_facts(
        &projection,
        &PreparedPhaseResourceFacts {
            inputs: prepared.inputs.clone(),
            output: prepared.result_output.clone(),
            output_set: prepared.output_set.clone(),
        },
    )?;
    if prepared.receipt_output.encoded_payload_bytes_upper_bound.0
        != crate::execution::FLUID_RECEIPT_MAX_ENCODED_BYTES
        || prepared.receipt_output.transport_closure_bytes_upper_bound
            < prepared.receipt_output.encoded_payload_bytes_upper_bound
        || prepared
            .receipt_output
            .construction_resident_bytes_admission_budget
            < prepared.receipt_output.transport_closure_bytes_upper_bound
    {
        return Err(invalid(
            "steady fluid receipt output differs from original fixed schema",
        ));
    }
    if prepared
        .receipt_output
        .construction_resident_bytes_admission_budget
        .0
        < receipt_profile.materialization_resident_bytes_estimate
    {
        return Err(invalid(
            "steady fluid receipt materialization is underfunded",
        ));
    }
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::FluidSteadySolve(projection.workload),
        demand: FeaResourceDemand::FluidSteadySolve(FluidSteadySolveResourceDemand {
            projection,
            inputs: prepared.inputs,
            result_output: prepared.result_output,
            receipt_output: prepared.receipt_output,
        }),
        output_set: Some(prepared.output_set),
        worker: worker(workload, prepared.local_execution)?,
        accuracy_ceiling: ceiling(workload)?,
    })
}
