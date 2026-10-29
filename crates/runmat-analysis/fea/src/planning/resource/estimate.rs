mod accuracy;
mod acoustic;
mod common;
mod electrical_static;
mod fluid;
mod electrical_recovery;
mod magnetic_recovery;
pub(super) mod magnetic_static;
pub(super) mod maxwell_harmonic;
mod maxwell_recovery;
mod acoustic_response;
mod acoustic_recovery;
mod nonlinear_recovery;
mod result_publication;
mod state_publication;
mod structural_recovery;
use common::validate_common;
mod constraint_compilation;
mod demand_validation;
mod host_phase;
mod operator;
use operator::build_operator_resource_plan;
mod operator_validation;
use operator_validation::validate_operator_demand;
mod implicit_input;
mod phase_policy;
mod post_process;
mod solve_storage;
mod sparse_validation;
mod thermal;

use demand_validation::{
    validate_buckling_demand, validate_modal_demand, validate_random_response_interval_demand,
    validate_transient_block_demand,
};
use implicit_input::implicit_resource_plan_input_facts;
use solve_storage::solve_storage;

use post_process::{
    build_frequency_result_assembly_resource_plan, build_observation_assembly_resource_plan,
    build_observation_history_resource_plan, build_observation_partition_resource_plan,
    build_observation_reduction_resource_plan, build_random_response_assembly_resource_plan,
    build_response_spectrum_assembly_resource_plan, build_transient_result_assembly_resource_plan,
};

use super::contract::{
    AccuracyResourceDemand, ByteCount, Count, FeaResourceDemand, FeaResourcePhase, FeaResourcePlan,
    FeaResourcePlanIdentity, FeaResourcePlanningInput, ObservationAssemblyResourceDemand,
    ObservationHistoryCheckpointResourceDemand, ObservationPartitionResourceDemand,
    ObservationReductionResourceDemand, ObservationResourceFacts, PartitioningPolicy,
    PreparedInputResourceFacts, PreparedOutputResourceFacts, ResourceBreakdown,
    ResourcePlanningError, SparseResourceFacts, StructuralFrequencyResultAssemblyResourceDemand,
    StructuralModalResponseSpectrumAssemblyResourceDemand, StructuralOperatorResourceDemand,
    StructuralRandomResponseAssemblyResourceDemand,
    StructuralTransientResultAssemblyResourceDemand, RESIDENT_MEMORY_RESOURCE,
    RESOURCE_ESTIMATION_MODEL_REVISION, RESOURCE_PLAN_MAX_ENCODED_BYTES,
};
use runmat_execution::{resource::ResourceRequest, Digest};

pub fn build_resource_plan(
    input: FeaResourcePlanningInput,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let plan = build_resource_plan_unlogged(input)?;
    tracing::debug!(
        "fea resource plan {:?}: request memory {} scratch {}; breakdown {:?}",
        plan.input.phase,
        plan.request.memory_bytes,
        plan.request.scratch_bytes,
        plan.breakdown
    );
    Ok(plan)
}

fn build_resource_plan_unlogged(
    input: FeaResourcePlanningInput,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    validate_common(&input)?;
    match (&input.phase, &input.demand) {
        (
            FeaResourcePhase::AcousticFrequencyAssembly(workload),
            FeaResourceDemand::AcousticFrequencyAssembly(demand),
        ) if *workload == demand.workload => {
            if demand.entry_count.0 == 0 {
                return Err(ResourcePlanningError::Invalid(
                    "acoustic assembly requires complete entries".into(),
                ));
            }
            host_phase::build_host_phase_resource_plan(
                input.clone(),
                host_phase::HostPhaseAccounting::host_only(
                    demand.topology_digest,
                    &demand.inputs,
                    demand.workspace_bytes_upper_bound,
                    ByteCount(0),
                    &[&demand.result_output, &demand.publication_output],
                ),
            )
        }
        (
            FeaResourcePhase::FluidTransientBlock(workload),
            FeaResourceDemand::FluidTransientBlock(demand),
        ) if *workload == demand.projection.workload => fluid::transient(input.clone(), demand),
        (
            FeaResourcePhase::FluidSteadySolve(workload),
            FeaResourceDemand::FluidSteadySolve(demand),
        ) if *workload == demand.projection.workload => fluid::solve(input.clone(), demand),
        (
            FeaResourcePhase::FluidFieldRecovery(request),
            FeaResourceDemand::FluidFieldRecovery(demand),
        ) if *request == demand.request => fluid::recovery(input.clone(), demand),
        (
            FeaResourcePhase::MaxwellHarmonic(workload),
            FeaResourceDemand::MaxwellHarmonic(demand),
        ) if *workload == demand.workload => {
            maxwell_harmonic::build_maxwell_harmonic_plan(input.clone(), demand)
        }
        (
            FeaResourcePhase::StructuralFieldRecovery(request),
            FeaResourceDemand::StructuralFieldRecovery(demand),
        ) if *request == demand.request => structural_recovery::build(input.clone(), demand),
        (
            FeaResourcePhase::NonlinearFieldRecovery(request),
            FeaResourceDemand::NonlinearFieldRecovery(demand),
        ) if *request == demand.request => nonlinear_recovery::build(input.clone(), demand),
        (
            FeaResourcePhase::MagneticFieldRecovery(request),
            FeaResourceDemand::MagneticFieldRecovery(demand),
        ) if *request == demand.request => magnetic_recovery::build(input.clone(), demand),
        (
            FeaResourcePhase::ElectricalFieldRecovery(request),
            FeaResourceDemand::ElectricalFieldRecovery(demand),
        ) if *request == demand.request => electrical_recovery::build(input.clone(), demand),
        (
            FeaResourcePhase::AcousticResponse(workload),
            FeaResourceDemand::AcousticResponse(demand),
        ) if *workload == demand.workload => acoustic_response::build(input.clone(), demand),
        (
            FeaResourcePhase::StatisticalFieldRecovery(request),
            FeaResourceDemand::StatisticalFieldRecovery(demand),
        ) if *request == demand.request => magnetic_recovery::build(input.clone(), demand),
        (
            FeaResourcePhase::MaxwellFieldRecovery(request),
            FeaResourceDemand::MaxwellFieldRecovery(demand),
        ) if *request == demand.request => maxwell_recovery::build(input.clone(), demand),
        (
            FeaResourcePhase::AcousticFieldRecovery(request),
            FeaResourceDemand::AcousticFieldRecovery(demand),
        ) if *request == demand.request => acoustic_recovery::build(input.clone(), demand),
        (FeaResourcePhase::Magnetostatic(workload), FeaResourceDemand::Magnetostatic(demand))
            if *workload == demand.workload =>
        {
            magnetic_static::build_magnetostatic_plan(input.clone(), demand)
        }
        (
            FeaResourcePhase::ElectricalStatic(workload),
            FeaResourceDemand::ElectricalStatic(demand),
        ) if *workload == demand.workload => {
            electrical_static::build_electrical_static_plan(input.clone(), demand)
        }
        (
            FeaResourcePhase::AcousticHarmonic(workload),
            FeaResourceDemand::AcousticHarmonic(demand),
        ) if *workload == demand.workload => acoustic::build_acoustic_plan(input.clone(), demand),
        (
            FeaResourcePhase::ResultPublication(request),
            FeaResourceDemand::ResultPublication(demand),
        ) if *request == demand.request => {
            result_publication::build_result_publication_plan(input.clone(), demand)
        }
        (
            FeaResourcePhase::SemanticStatePublication(request),
            FeaResourceDemand::SemanticStatePublication(demand),
        ) if *request == demand.request => {
            state_publication::build_state_publication_plan(input.clone(), demand)
        }
        (
            FeaResourcePhase::AccuracyEvaluation(request),
            FeaResourceDemand::AccuracyEvaluation(demand),
        ) if *request == demand.request => accuracy::build_accuracy_plan(input.clone(), demand),
        (FeaResourcePhase::ThermalSolve(workload), FeaResourceDemand::ThermalSolve(demand))
            if *workload == demand.workload =>
        {
            thermal::build_thermal_plan(input.clone(), demand)
        }
        (
            FeaResourcePhase::StructuralNonlinearRecovery,
            FeaResourceDemand::StructuralNonlinearRecovery(demand),
        ) => {
            if demand.integration_identity == Digest::sha256([])
                || demand.point_count.0 == 0
                || demand.retained_output_increments.0 == 0
                || demand.physical_evaluation_limit.0 != 2
                || demand.additional_persistent_bytes.0 == 0
            {
                return Err(ResourcePlanningError::Invalid(
                    "nonlinear recovery physical/storage authority is incomplete".into(),
                ));
            }
            operator::build_operator_resource_plan_with_persistent_storage(
                input.clone(),
                &demand.operator,
                None,
                demand.additional_persistent_bytes,
            )
        }
        (
            FeaResourcePhase::StructuralNonlinearPipeline,
            FeaResourceDemand::StructuralNonlinearPipeline(demand),
        ) => {
            if demand.integration_identity == Digest::sha256([])
                || demand.point_count.0 == 0
                || demand.maximum_accepted_increments.0 == 0
                || demand.maximum_retained_output_increments.0 == 0
                || demand.maximum_retained_output_increments > demand.maximum_accepted_increments
                || demand.tangent_evaluation_limit.0 == 0
                || demand.linear_solve_limit.0 == 0
                || demand.additional_persistent_bytes.0 == 0
            {
                return Err(ResourcePlanningError::Invalid(
                    "nonlinear point/work/storage authority is incomplete".into(),
                ));
            }
            operator::build_operator_resource_plan_with_persistent_storage(
                input.clone(),
                &demand.operator,
                demand.block_transport.as_ref().map(|block| &block.inputs),
                demand.additional_persistent_bytes,
            )
        }
        (
            FeaResourcePhase::StructuralLinearPipeline,
            FeaResourceDemand::StructuralLinearPipeline(demand),
        ) => build_operator_resource_plan(input.clone(), &demand.operator, demand.inputs.as_ref()),
        (
            FeaResourcePhase::StructuralModalExtraction(workload),
            FeaResourceDemand::StructuralModalExtraction(demand),
        ) if *workload == demand.workload => {
            validate_modal_demand(demand)?;
            build_operator_resource_plan(input.clone(), &demand.operator, Some(&demand.inputs))
        }
        (
            FeaResourcePhase::StructuralBuckling(workload),
            FeaResourceDemand::StructuralBuckling(demand),
        ) if *workload == demand.workload => {
            validate_buckling_demand(demand)?;
            build_operator_resource_plan(input.clone(), &demand.operator, Some(&demand.inputs))
        }
        (
            FeaResourcePhase::StructuralTransientScopePreparation(workload_digest),
            FeaResourceDemand::StructuralTransientScopePreparation(demand),
        ) if *workload_digest == demand.source_workload_digest => {
            build_operator_resource_plan(input.clone(), &demand.operator, Some(&demand.inputs))
        }
        (
            FeaResourcePhase::StructuralTransientBlock(request),
            FeaResourceDemand::StructuralTransientBlock(demand),
        ) if *request == demand.request => {
            validate_transient_block_demand(demand)?;
            build_operator_resource_plan(input.clone(), &demand.operator, Some(&demand.inputs))
        }
        (
            FeaResourcePhase::StructuralTransientFieldRecovery(result),
            FeaResourceDemand::StructuralTransientFieldRecovery(demand),
        ) if *result == demand.result => {
            if demand.sample_count.0 == 0
                || demand.physical_dof_count != demand.operator.sparse.rows
                || demand.decoded_sample_bytes_upper_bound.0 == 0
                || demand.retained_field_bytes_upper_bound.0 == 0
                || demand.operator.phase_resources.retained_bytes
                    < demand.retained_field_bytes_upper_bound
            {
                return Err(ResourcePlanningError::Invalid(
                    "transient recovery dimensions or retained-field bound are inconsistent".into(),
                ));
            }
            build_operator_resource_plan(input.clone(), &demand.operator, Some(&demand.inputs))
        }
        (
            FeaResourcePhase::StructuralFrequencyEntry(_),
            FeaResourceDemand::StructuralFrequencyEntry(demand),
        )
        | (
            FeaResourcePhase::StructuralFrequencyFieldRecovery(_),
            FeaResourceDemand::StructuralFrequencyFieldRecovery(demand),
        ) => build_operator_resource_plan(input.clone(), &demand.operator, Some(&demand.inputs)),
        (
            FeaResourcePhase::StructuralObservationPartition(_),
            FeaResourceDemand::StructuralObservationPartition(demand),
        ) => build_observation_partition_resource_plan(input.clone(), demand),
        (
            FeaResourcePhase::StructuralObservationReduction(_),
            FeaResourceDemand::StructuralObservationReduction(demand),
        ) => build_observation_reduction_resource_plan(input.clone(), demand),
        (
            FeaResourcePhase::StructuralObservationHistoryCheckpoint(_),
            FeaResourceDemand::StructuralObservationHistoryCheckpoint(demand),
        ) => build_observation_history_resource_plan(input.clone(), demand),
        (
            FeaResourcePhase::ObservationHistoryFinalization(request),
            FeaResourceDemand::ObservationHistoryFinalization(demand),
        ) if *request == demand.request => {
            post_process::build_observation_history_finalization_resource_plan(
                input.clone(),
                demand,
            )
        }
        (
            FeaResourcePhase::StructuralObservationAssembly(_),
            FeaResourceDemand::StructuralObservationAssembly(demand),
        ) => build_observation_assembly_resource_plan(input.clone(), demand),
        (
            FeaResourcePhase::StructuralFrequencyResultAssembly(_),
            FeaResourceDemand::StructuralFrequencyResultAssembly(demand),
        ) => build_frequency_result_assembly_resource_plan(input.clone(), demand),
        (
            FeaResourcePhase::StructuralTransientResultAssembly(_),
            FeaResourceDemand::StructuralTransientResultAssembly(demand),
        ) => build_transient_result_assembly_resource_plan(input.clone(), demand),
        (
            FeaResourcePhase::StructuralModalResponseSpectrumDirection(_),
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand),
        ) => build_operator_resource_plan(input.clone(), &demand.operator, Some(&demand.inputs)),
        (
            FeaResourcePhase::StructuralModalResponseSpectrumAssembly(_),
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(demand),
        ) => build_response_spectrum_assembly_resource_plan(input.clone(), demand),
        (
            FeaResourcePhase::StructuralRandomResponseInterval(_),
            FeaResourceDemand::StructuralRandomResponseInterval(demand),
        ) => {
            validate_random_response_interval_demand(demand)?;
            build_operator_resource_plan(input.clone(), &demand.operator, Some(&demand.inputs))
        }
        (
            FeaResourcePhase::StructuralRandomResponseAssembly(_),
            FeaResourceDemand::StructuralRandomResponseAssembly(demand),
        ) => build_random_response_assembly_resource_plan(input.clone(), demand),
        _ => Err(ResourcePlanningError::Invalid(
            "resource phase and typed demand variant differ".into(),
        )),
    }
}

fn prepared_output_is_valid(output: &PreparedOutputResourceFacts) -> bool {
    output.encoded_payload_bytes_upper_bound.0 > 0
        && output.transport_closure_bytes_upper_bound >= output.encoded_payload_bytes_upper_bound
        && output.construction_resident_bytes_admission_budget
            >= output.encoded_payload_bytes_upper_bound
}

fn operator_peak_resident_bytes(
    persistent_numerical: u64,
    decoded_inputs: u64,
    decoded_resource_plan: u64,
    assembly_phase: u64,
    solve_phase: u64,
    recovery_phase: u64,
) -> Result<u64, ResourcePlanningError> {
    add(
        sum(
            &[persistent_numerical, decoded_inputs, decoded_resource_plan],
            "persistent input and numerical memory",
        )?,
        assembly_phase.max(solve_phase).max(recovery_phase),
        "numerical peak memory",
    )
}

struct PartitionFacts {
    partitionable_memory: u64,
    fixed_memory: u64,
    partitionable_scratch: u64,
    fixed_scratch: u64,
    replicated_memory: u64,
    halo_memory: u64,
    replicated_scratch: u64,
    halo_scratch: u64,
}

fn partition_facts(
    input: &FeaResourcePlanningInput,
    partitioning: &PartitioningPolicy,
    numerical_memory: u64,
    total_memory: u64,
    phase_scratch: u64,
    total_scratch: u64,
) -> Result<PartitionFacts, ResourcePlanningError> {
    match partitioning {
        PartitioningPolicy::Monolithic => {
            if total_memory > input.accuracy_ceiling.resident_byte_budget() {
                return Err(ResourcePlanningError::AccuracyCeiling {
                    resource: RESIDENT_MEMORY_RESOURCE,
                    required: total_memory,
                    maximum: input.accuracy_ceiling.resident_byte_budget(),
                });
            }
            Ok(PartitionFacts {
                partitionable_memory: 0,
                fixed_memory: total_memory,
                partitionable_scratch: 0,
                fixed_scratch: total_scratch,
                replicated_memory: 0,
                halo_memory: 0,
                replicated_scratch: 0,
                halo_scratch: 0,
            })
        }
        PartitioningPolicy::OperatorDomain {
            replicated_memory_bytes_per_partition,
            halo_memory_bytes_per_partition,
            replicated_scratch_bytes_per_partition,
            halo_scratch_bytes_per_partition,
            ..
        } => Ok(PartitionFacts {
            partitionable_memory: numerical_memory,
            fixed_memory: sum(
                &[
                    input.worker.fixed_memory_bytes.0,
                    replicated_memory_bytes_per_partition.0,
                    halo_memory_bytes_per_partition.0,
                ],
                "partition fixed memory",
            )?,
            partitionable_scratch: phase_scratch,
            fixed_scratch: sum(
                &[
                    input.worker.fixed_scratch_bytes.0,
                    replicated_scratch_bytes_per_partition.0,
                    halo_scratch_bytes_per_partition.0,
                ],
                "partition fixed scratch",
            )?,
            replicated_memory: replicated_memory_bytes_per_partition.0,
            halo_memory: halo_memory_bytes_per_partition.0,
            replicated_scratch: replicated_scratch_bytes_per_partition.0,
            halo_scratch: halo_scratch_bytes_per_partition.0,
        }),
    }
}

mod accuracy_policy;
use accuracy_policy::validate_accuracy_demand;

fn bytes(count: u64, width: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    mul(count, width, label)
}
fn add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
fn mul(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
fn sum(values: &[u64], label: &'static str) -> Result<u64, ResourcePlanningError> {
    values
        .iter()
        .try_fold(0, |total, value| add(total, *value, label))
}

#[cfg(test)]
mod lifetime_tests {
    use super::*;

    #[test]
    fn decoded_inputs_are_additive_when_the_solve_phase_dominates() {
        let baseline = operator_peak_resident_bytes(100, 20, 5, 10, 1_000, 30).unwrap();
        let larger = operator_peak_resident_bytes(100, 21, 5, 10, 1_000, 30).unwrap();
        assert_eq!(baseline, 1_125);
        assert_eq!(larger, baseline + 1);
    }
}
