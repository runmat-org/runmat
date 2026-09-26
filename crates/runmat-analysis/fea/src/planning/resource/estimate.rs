mod accuracy;
mod common;
mod result_publication;
mod state_publication;
use common::validate_common;
mod demand_validation;
mod host_phase;
mod operator;
use operator::build_operator_resource_plan;
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
use phase_policy::requires_no_solve_policy;
use solve_storage::solve_storage;
use sparse_validation::validate_sparse;

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
    ResourcePlanningError, SolveResourcePolicy, SparseResourceFacts,
    StructuralFrequencyResultAssemblyResourceDemand,
    StructuralModalResponseSpectrumAssemblyResourceDemand, StructuralOperatorResourceDemand,
    StructuralRandomResponseAssemblyResourceDemand,
    StructuralTransientResultAssemblyResourceDemand, RESOURCE_ESTIMATION_MODEL_REVISION,
    RESOURCE_PLAN_MAX_ENCODED_BYTES,
};
use runmat_execution::{resource::ResourceRequest, Digest};

pub fn build_resource_plan(
    input: FeaResourcePlanningInput,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    validate_common(&input)?;
    match (&input.phase, &input.demand) {
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
            FeaResourcePhase::StructuralLinearPipeline,
            FeaResourceDemand::StructuralLinearPipeline(demand),
        ) => build_operator_resource_plan(input.clone(), &demand.operator, None),
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
            if total_memory > input.accuracy_ceiling.maximum_resident_bytes {
                return Err(ResourcePlanningError::AccuracyCeiling {
                    resource: "resident memory bytes",
                    required: total_memory,
                    maximum: input.accuracy_ceiling.maximum_resident_bytes,
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

fn validate_operator_demand(
    input: &FeaResourcePlanningInput,
    demand: &StructuralOperatorResourceDemand,
) -> Result<(), ResourcePlanningError> {
    if demand.topology_digest == Digest::sha256([])
        || demand.numerical_scope_digest == Digest::sha256([])
    {
        return Err(ResourcePlanningError::Invalid(
            "operator resource scope and identities must be nonzero".into(),
        ));
    }
    if demand.topology.nodes.0 == 0
        || demand.topology.elements.0 == 0
        || demand.topology.estimated_resident_bytes.0 == 0
        || demand.constraints.estimated_resident_bytes.0 == 0
        || demand.sparse.rows.0 == 0
        || demand.sparse.columns.0 == 0
        || demand.sparse.rows != demand.sparse.columns
    {
        return Err(ResourcePlanningError::Invalid(
            "operator phases require topology, resident bounds, constraints, and a square sparse structure"
                .into(),
        ));
    }
    validate_sparse(&demand.sparse)?;
    let topology_payload_lower_bound = sum(
        &[
            bytes(demand.topology.nodes.0, 24, "topology coordinate payload")?,
            bytes(
                demand.topology.element_node_references.0,
                8,
                "element index payload",
            )?,
            bytes(
                demand.topology.boundary_node_references.0,
                8,
                "boundary index payload",
            )?,
            bytes(
                demand.topology.node_neighbor_references.0,
                8,
                "neighbor index payload",
            )?,
        ],
        "topology structural lower bound",
    )?;
    if demand.topology.estimated_resident_bytes.0 < topology_payload_lower_bound {
        return Err(ResourcePlanningError::Invalid(
            "topology resident bound is below its coordinate/index payload".into(),
        ));
    }
    let constraint_payload_lower_bound = sum(
        &[
            bytes(
                demand.constraints.rows.0,
                8,
                "constraint right-hand-side payload",
            )?,
            bytes(
                demand.constraints.coefficients.0,
                8,
                "constraint coefficient payload",
            )?,
            bytes(
                demand.constraints.reference_points.0,
                24,
                "constraint point payload",
            )?,
        ],
        "constraint structural lower bound",
    )?;
    if demand.constraints.estimated_resident_bytes.0 < constraint_payload_lower_bound {
        return Err(ResourcePlanningError::Invalid(
            "constraint resident bound is below its numerical payload".into(),
        ));
    }
    let contribution_payload_lower_bound = bytes(
        demand.assembly.contribution_nonzero_upper_bound.0,
        24,
        "assembly row, column, and real-f64 contribution payload",
    )?;
    let phase_payload_lower_bound = bytes(
        demand.phase_resources.resident_value_count.0,
        8,
        "phase value payload",
    )?;
    if demand.phase_resources.resident_bytes_upper_bound.0 < phase_payload_lower_bound {
        return Err(ResourcePlanningError::Invalid(
            "phase resident bound is below its scalar payload".into(),
        ));
    }
    let recovery = &demand.field_recovery;
    let recovery_is_empty = recovery.requirement_count.0 == 0
        && recovery.source_entity_count.0 == 0
        && recovery.output_entity_count.0 == 0
        && recovery.topology_incidence_count.0 == 0
        && recovery.arithmetic_operation_count.0 == 0;
    if !recovery_is_empty
        && (recovery.requirement_count.0 == 0
            || recovery.source_entity_count.0 == 0
            || recovery.output_entity_count.0 == 0
            || recovery.arithmetic_operation_count.0 == 0)
    {
        return Err(ResourcePlanningError::Invalid(
            "field recovery resource facts are incomplete".into(),
        ));
    }
    if demand.assembly.contribution_nonzero_upper_bound.0 < demand.sparse.maximum_nonzeros()
        || demand.assembly.contribution_bytes_upper_bound.0 == 0
        || demand.assembly.contribution_bytes_upper_bound.0 < contribution_payload_lower_bound
    {
        return Err(ResourcePlanningError::Invalid(
            "assembly contribution count and byte bounds must cover canonical nonzeros".into(),
        ));
    }
    match demand.solve {
        SolveResourcePolicy::NoSolve { .. } => {}
        SolveResourcePolicy::Iterative { vector_count, .. }
        | SolveResourcePolicy::DirectFactorization { vector_count, .. }
            if vector_count.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "solve policy requires at least one vector".into(),
            ));
        }
        SolveResourcePolicy::RestartedGmres {
            block_dimension,
            restart,
            maximum_iterations,
            workspace_bytes_upper_bound,
        }
        | SolveResourcePolicy::SaddlePointGmres {
            block_dimension,
            restart,
            maximum_iterations,
            workspace_bytes_upper_bound,
            ..
        } if maximum_iterations.0 == 0
            || (block_dimension.0 == 0 && restart.0 != 0)
            || (block_dimension.0 != 0 && restart.0 == 0)
            || restart.0 > block_dimension.0
            || restart.0 > maximum_iterations.0
            || workspace_bytes_upper_bound.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "restarted GMRES dimensions, iteration bounds, and workspace must be valid".into(),
            ));
        }
        SolveResourcePolicy::SaddlePointGmres {
            block_dimension,
            restart,
            maximum_iterations,
            workspace_bytes_upper_bound,
            ..
        } if block_dimension.0
            != add(
                demand.sparse.rows.0,
                demand.constraints.rows.0,
                "saddle dimension",
            )?
            || restart.0 != block_dimension.0.min(32)
            || maximum_iterations.0
                != block_dimension
                    .0
                    .checked_mul(8)
                    .ok_or(ResourcePlanningError::Overflow(
                        "saddle GMRES iteration bound",
                    ))?
                    .max(256)
            || workspace_bytes_upper_bound.0
                < super::preflight::saddle_gmres_workspace_bytes(
                    demand.sparse.rows.0,
                    demand.constraints.rows.0,
                )? =>
        {
            return Err(ResourcePlanningError::Invalid(
                "saddle GMRES dimensions or workspace disagree with structural facts".into(),
            ));
        }
        SolveResourcePolicy::ModalDenseHarmonic {
            full_dimension,
            mode_count,
            workspace_bytes_upper_bound,
        } if full_dimension.0 == 0
            || mode_count.0 == 0
            || mode_count.0 > full_dimension.0
            || workspace_bytes_upper_bound.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "modal harmonic dimensions and workspace must be valid".into(),
            ));
        }
        SolveResourcePolicy::TransientBlock {
            physical_dimension,
            coordinate_dimension,
            workspace_bytes_upper_bound,
        } if physical_dimension != demand.sparse.rows
            || coordinate_dimension.0 == 0
            || coordinate_dimension.0 > physical_dimension.0
            || workspace_bytes_upper_bound.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "transient block workspace and dimensions must be valid".into(),
            ));
        }
        SolveResourcePolicy::SymmetricSubspaceIteration {
            full_dimension,
            reduced_dimension,
            requested_mode_count,
            initial_subspace_dimension,
            maximum_subspace_dimension,
            boundary_probe_width,
            subspace_expansion_count_upper_bound,
            maximum_iterations_per_expansion,
            orthogonalization_passes_per_block,
            dense_projected_value_count_upper_bound,
            workspace_bytes_upper_bound,
        } if full_dimension.0 == 0
            || reduced_dimension.0 == 0
            || reduced_dimension.0 > full_dimension.0
            || requested_mode_count.0 == 0
            || requested_mode_count.0 > reduced_dimension.0
            || initial_subspace_dimension.0 < requested_mode_count.0
            || initial_subspace_dimension.0 > maximum_subspace_dimension.0
            || maximum_subspace_dimension != reduced_dimension
            || boundary_probe_width.0 == 0
            || subspace_expansion_count_upper_bound.0 == 0
            || maximum_iterations_per_expansion.0 == 0
            || orthogonalization_passes_per_block.0 == 0
            || dense_projected_value_count_upper_bound.0
                != maximum_subspace_dimension
                    .0
                    .saturating_mul(maximum_subspace_dimension.0)
            || workspace_bytes_upper_bound.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "modal extraction dimensions, probe bounds, and workspace must be valid".into(),
            ));
        }
        _ => {}
    }
    if requires_no_solve_policy(&input.phase)
        != matches!(demand.solve, SolveResourcePolicy::NoSolve { .. })
    {
        return Err(ResourcePlanningError::Invalid(
            "solve and post-processing phases require their matching solve policy".into(),
        ));
    }
    match (&demand.partitioning, &demand.solve) {
        (
            PartitioningPolicy::OperatorDomain {
                maximum_partitions, ..
            },
            _,
        ) if *maximum_partitions < 2 => {
            return Err(ResourcePlanningError::Invalid(
                "operator/domain partitioning requires at least two partitions".into(),
            ))
        }
        (
            PartitioningPolicy::OperatorDomain { .. },
            SolveResourcePolicy::DirectFactorization { .. },
        ) => {
            return Err(ResourcePlanningError::Invalid(
                "direct factorization lacks a partition-local fill contract".into(),
            ))
        }
        _ => {}
    }
    Ok(())
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
