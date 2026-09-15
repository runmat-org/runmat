mod post_process;

use post_process::{
    build_frequency_result_assembly_resource_plan, build_observation_assembly_resource_plan,
    build_observation_history_resource_plan, build_observation_partition_resource_plan,
    build_observation_reduction_resource_plan,
};

use runmat_execution::{resource::ResourceRequest, Digest};

use super::contract::{
    AccuracyResourceDemand, ByteCount, Count, FeaResourceDemand, FeaResourcePhase, FeaResourcePlan,
    FeaResourcePlanIdentity, FeaResourcePlanningInput, ObservationAssemblyResourceDemand,
    ObservationHistoryCheckpointResourceDemand, ObservationPartitionResourceDemand,
    ObservationReductionResourceDemand, ObservationResourceFacts, PartitioningPolicy,
    PreconditionerStorage, PreparedInputResourceFacts, PreparedOutputResourceFacts,
    ResourceBreakdown, ResourcePlanningError, SolveResourcePolicy, SparseResourceFacts,
    StructuralFrequencyResultAssemblyResourceDemand, StructuralOperatorResourceDemand,
    RESOURCE_ESTIMATION_MODEL_REVISION, RESOURCE_PLAN_MAX_ENCODED_BYTES,
};

fn implicit_resource_plan_input_facts() -> Result<PreparedInputResourceFacts, ResourcePlanningError>
{
    let decoded =
        runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
            RESOURCE_PLAN_MAX_ENCODED_BYTES,
            1,
        )
        .ok_or(ResourcePlanningError::Overflow(
            "resource-plan decoded residency budget",
        ))?;
    let transport = add(
        RESOURCE_PLAN_MAX_ENCODED_BYTES,
        crate::execution::ANALYSIS_MANIFEST_MAX_ENCODED_BYTES,
        "resource-plan transport closure budget",
    )?;
    Ok(PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(RESOURCE_PLAN_MAX_ENCODED_BYTES),
        transport_closure_bytes: ByteCount(transport),
        decoded_resident_bytes_admission_budget: ByteCount(decoded),
    })
}

pub fn build_resource_plan(
    input: FeaResourcePlanningInput,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    validate_common(&input)?;
    match (&input.phase, &input.demand) {
        (
            FeaResourcePhase::StructuralLinearPipeline,
            FeaResourceDemand::StructuralLinearPipeline(demand),
        ) => build_operator_resource_plan(input.clone(), demand, None),
        (
            FeaResourcePhase::StructuralModalExtraction(workload),
            FeaResourceDemand::StructuralModalExtraction(demand),
        ) if *workload == demand.workload => {
            validate_modal_demand(demand)?;
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
            FeaResourcePhase::StructuralObservationAssembly(_),
            FeaResourceDemand::StructuralObservationAssembly(demand),
        ) => build_observation_assembly_resource_plan(input.clone(), demand),
        (
            FeaResourcePhase::StructuralFrequencyResultAssembly(_),
            FeaResourceDemand::StructuralFrequencyResultAssembly(demand),
        ) => build_frequency_result_assembly_resource_plan(input.clone(), demand),
        _ => Err(ResourcePlanningError::Invalid(
            "resource phase and typed demand variant differ".into(),
        )),
    }
}

fn validate_modal_demand(
    demand: &super::contract::StructuralModalResourceDemand,
) -> Result<(), ResourcePlanningError> {
    let SolveResourcePolicy::ModalSubspaceIteration {
        reduced_dimension,
        maximum_subspace_dimension,
        ..
    } = demand.operator.solve
    else {
        return Err(ResourcePlanningError::Invalid(
            "structural modal demand requires a modal-subspace solve policy".into(),
        ));
    };
    if demand.retained_mode_count_upper_bound.0 == 0
        || demand.retained_mode_count_upper_bound.0 > reduced_dimension.0
        || demand.modal_phase_value_count_upper_bound.0
            < demand.modal_output_value_count_upper_bound.0
        || demand.modal_output_value_count_upper_bound.0 == 0
        || maximum_subspace_dimension != reduced_dimension
        || demand.operator.partitioning != PartitioningPolicy::Monolithic
        || !prepared_output_is_valid(&demand.result_output)
        || !prepared_output_is_valid(&demand.receipt_output)
    {
        return Err(ResourcePlanningError::Invalid(
            "structural modal output, subspace, or placement bounds are inconsistent".into(),
        ));
    }
    Ok(())
}

fn prepared_output_is_valid(output: &PreparedOutputResourceFacts) -> bool {
    output.encoded_payload_bytes_upper_bound.0 > 0
        && output.transport_closure_bytes_upper_bound >= output.encoded_payload_bytes_upper_bound
        && output.construction_resident_bytes_admission_budget
            >= output.encoded_payload_bytes_upper_bound
}

fn build_operator_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &StructuralOperatorResourceDemand,
    inputs: Option<&PreparedInputResourceFacts>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    validate_operator_demand(&input, demand)?;
    let resource_plan_input = implicit_resource_plan_input_facts()?;
    let output_set_bytes = input
        .output_set
        .as_ref()
        .map_or(ByteCount(0), |facts| facts.encoded_bytes_upper_bound);
    let topology_bytes = demand.topology.estimated_resident_bytes;
    let operator_count = u64::try_from(demand.sparse.operator_nonzeros.len())
        .map_err(|_| ResourcePlanningError::Overflow("sparse operator count"))?;
    let total_operator_nonzeros = demand
        .sparse
        .operator_nonzeros
        .iter()
        .try_fold(0_u64, |total, count| add(total, count.0, "sparse nonzeros"))?;
    let csr_row_offset_bytes = ByteCount(bytes(
        mul(
            add(demand.sparse.rows.0, 1, "CSR row offsets")?,
            operator_count,
            "CSR row offsets",
        )?,
        8,
        "CSR row offsets",
    )?);
    let csr_column_index_bytes =
        ByteCount(bytes(total_operator_nonzeros, 8, "CSR column indices")?);
    let csr_value_bytes = ByteCount(bytes(total_operator_nonzeros, 8, "CSR values")?);
    let assembly_contribution_bytes = demand.assembly.contribution_bytes_upper_bound;
    let assembly_scratch_bytes = ByteCount(add(
        demand.assembly.fixed_scratch_bytes_per_worker.0,
        mul(
            demand.topology.elements.0,
            demand.assembly.scratch_bytes_per_element.0,
            "assembly element scratch",
        )?,
        "assembly scratch",
    )?);
    let workspace_bytes = match demand.solve.explicit_workspace_bytes() {
        Some(bytes) => bytes,
        None => ByteCount(bytes(
            mul(
                demand.sparse.rows.0,
                demand.solve.vector_count().ok_or_else(|| {
                    ResourcePlanningError::Invalid(
                        "solve policy omits both vectors and explicit workspace".into(),
                    )
                })?,
                "solve vector values",
            )?,
            demand.solve.scalar_bytes(),
            "solve vectors",
        )?),
    };
    let (preconditioner_bytes, factorization_bytes) = solve_storage(demand)?;
    let constraint_bytes = demand.constraints.estimated_resident_bytes;
    let input_decoded_resident_bytes = inputs.map_or(ByteCount(0), |inputs| {
        inputs.decoded_resident_bytes_admission_budget
    });
    let phase_resident_bytes = demand.phase_resources.resident_bytes_upper_bound;
    let persistent_numerical = sum(
        &[
            topology_bytes.0,
            csr_row_offset_bytes.0,
            csr_column_index_bytes.0,
            csr_value_bytes.0,
            constraint_bytes.0,
        ],
        "persistent numerical memory",
    )?;
    let solve_phase = sum(
        &[
            workspace_bytes.0,
            preconditioner_bytes.0,
            factorization_bytes.0,
        ],
        "solve phase memory",
    )?;
    let recovery_phase = sum(
        &[
            phase_resident_bytes.0,
            demand.observations.peak_resident_bytes.0,
            output_set_bytes.0,
        ],
        "recovery phase memory",
    )?;
    let numerical_memory = operator_peak_resident_bytes(
        persistent_numerical,
        input_decoded_resident_bytes.0,
        resource_plan_input
            .decoded_resident_bytes_admission_budget
            .0,
        assembly_contribution_bytes.0,
        solve_phase,
        recovery_phase,
    )?;
    let total_memory = add(
        input.worker.fixed_memory_bytes.0,
        numerical_memory,
        "worker peak memory",
    )?;
    let phase_scratch = assembly_scratch_bytes
        .0
        .max(demand.phase_resources.scratch_bytes.0);
    let total_scratch = add(
        input.worker.fixed_scratch_bytes.0,
        phase_scratch,
        "worker scratch",
    )?;
    let artifact_bytes = sum(
        &[
            demand.observations.stored_bytes.0,
            demand.publication.result_artifact_bytes.0,
            demand.publication.checkpoint_bytes.0,
            demand.publication.artifact_overhead_bytes.0,
            output_set_bytes.0,
        ],
        "published artifacts",
    )?;
    let total_egress = sum(
        &[
            demand.observations.transfer_bytes.0,
            demand.publication.result_egress_bytes.0,
            demand.publication.checkpoint_egress_bytes.0,
            output_set_bytes.0,
        ],
        "result and checkpoint egress",
    )?;

    if input.worker.maximum_wall_milliseconds
        > input.accuracy_ceiling.maximum_wall_time_milliseconds
    {
        return Err(ResourcePlanningError::AccuracyCeiling("wall time"));
    }
    validate_accuracy_demand(&input, &demand.accuracy_demand, &demand.observations)?;
    let partition = partition_facts(
        &input,
        &demand.partitioning,
        numerical_memory,
        total_memory,
        phase_scratch,
        total_scratch,
    )?;
    let request = ResourceRequest {
        cpu_millicores: input.worker.cpu_millicores,
        memory_bytes: total_memory,
        scratch_bytes: total_scratch,
        max_wall_millis: input.worker.maximum_wall_milliseconds,
        max_artifact_bytes: artifact_bytes,
        max_egress_bytes: total_egress,
        max_relay_bytes: add(
            add(
                demand.publication.relay_bytes.0,
                inputs.map_or(0, |inputs| inputs.transport_closure_bytes.0),
                "operator input and publication relay bytes",
            )?,
            resource_plan_input.transport_closure_bytes.0,
            "operator resource-plan relay bytes",
        )?,
        accelerators: input.worker.accelerators.iter().map(Into::into).collect(),
        required_capabilities: input.worker.required_capabilities.clone(),
    };
    request
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let breakdown = ResourceBreakdown {
        input_encoded_payload_bytes: inputs
            .map_or(ByteCount(0), |value| value.encoded_payload_bytes),
        input_transport_closure_bytes: inputs
            .map_or(ByteCount(0), |value| value.transport_closure_bytes),
        input_decoded_resident_bytes,
        resource_plan_transport_closure_bytes: resource_plan_input.transport_closure_bytes,
        resource_plan_decoded_resident_bytes: resource_plan_input
            .decoded_resident_bytes_admission_budget,
        output_set_bytes,
        topology_bytes,
        csr_row_offset_bytes,
        csr_column_index_bytes,
        csr_value_bytes,
        assembly_contribution_bytes,
        assembly_scratch_bytes,
        workspace_bytes,
        preconditioner_bytes,
        factorization_bytes,
        constraint_bytes,
        phase_resident_bytes,
        phase_scratch_bytes: demand.phase_resources.scratch_bytes,
        phase_retained_bytes: demand.phase_resources.retained_bytes,
        result_artifact_bytes: demand.publication.result_artifact_bytes,
        observation_maximum_samples_per_observation: demand
            .observations
            .maximum_samples_per_observation,
        observation_produced_sample_count: demand.observations.produced_sample_count,
        observation_retained_sample_count: demand.observations.retained_sample_count,
        observation_operation_count: demand.observations.operation_count,
        observation_peak_bytes: demand.observations.peak_resident_bytes,
        observation_stored_bytes: demand.observations.stored_bytes,
        observation_transfer_bytes: demand.observations.transfer_bytes,
        result_egress_bytes: demand.publication.result_egress_bytes,
        checkpoint_egress_bytes: demand.publication.checkpoint_egress_bytes,
        total_egress_bytes: ByteCount(total_egress),
        observation_collectives: demand.observations.collective_count,
        checkpoint_bytes: demand.publication.checkpoint_bytes,
        artifact_overhead_bytes: demand.publication.artifact_overhead_bytes,
        worker_overhead_bytes: input.worker.fixed_memory_bytes,
        partition_replicated_memory_bytes: ByteCount(partition.replicated_memory),
        partition_halo_memory_bytes: ByteCount(partition.halo_memory),
        partition_replicated_scratch_bytes: ByteCount(partition.replicated_scratch),
        partition_halo_scratch_bytes: ByteCount(partition.halo_scratch),
        partitionable_memory_bytes: ByteCount(partition.partitionable_memory),
        fixed_memory_bytes: ByteCount(partition.fixed_memory),
        partitionable_scratch_bytes: ByteCount(partition.partitionable_scratch),
        fixed_scratch_bytes: ByteCount(partition.fixed_scratch),
    };
    let mut plan = FeaResourcePlan {
        schema_version: 1,
        identity: FeaResourcePlanIdentity(Digest::sha256([])),
        input,
        breakdown,
        request,
    };
    plan.identity = plan.derive_identity()?;
    plan.validate_identity()?;
    Ok(plan)
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
                return Err(ResourcePlanningError::AccuracyCeiling("resident memory"));
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

fn validate_common(input: &FeaResourcePlanningInput) -> Result<(), ResourcePlanningError> {
    if input.estimation_model_revision != RESOURCE_ESTIMATION_MODEL_REVISION {
        return Err(ResourcePlanningError::Invalid(
            "unsupported resource estimation model revision".into(),
        ));
    }
    if input.worker.cpu_millicores == 0 || input.worker.maximum_wall_milliseconds == 0 {
        return Err(ResourcePlanningError::Invalid(
            "worker CPU and wall-time bounds must be nonzero".into(),
        ));
    }
    if input.demand.governs_prepared_outputs()
        != input
            .output_set
            .as_ref()
            .is_some_and(|facts| facts.encoded_bytes_upper_bound.0 > 0)
    {
        return Err(ResourcePlanningError::Invalid(
            "independently prepared outputs require one bounded output set".into(),
        ));
    }
    if input.worker.maximum_wall_milliseconds
        > input.accuracy_ceiling.maximum_wall_time_milliseconds
    {
        return Err(ResourcePlanningError::AccuracyCeiling("wall time"));
    }
    let mut accelerator_classes = std::collections::BTreeSet::new();
    for accelerator in &input.worker.accelerators {
        if accelerator.class.trim().is_empty()
            || accelerator.count == 0
            || accelerator.memory_bytes_each.0 == 0
            || !accelerator_classes.insert(accelerator.class.as_str())
        {
            return Err(ResourcePlanningError::Invalid(
                "accelerator class, count, and memory must be nonzero and classes must be unique"
                    .into(),
            ));
        }
    }
    Ok(())
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
        SolveResourcePolicy::ModalSubspaceIteration {
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
    let recovery_phase = matches!(
        input.phase,
        FeaResourcePhase::StructuralFrequencyFieldRecovery(_)
    );
    if recovery_phase != matches!(demand.solve, SolveResourcePolicy::NoSolve { .. }) {
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

fn validate_sparse(facts: &SparseResourceFacts) -> Result<(), ResourcePlanningError> {
    if facts.rows.0 == 0
        || facts.columns.0 == 0
        || facts.operator_nonzeros.is_empty()
        || facts.operator_nonzeros.iter().any(|count| count.0 == 0)
    {
        return Err(ResourcePlanningError::Invalid(
            "sparse dimensions and every operator nonzero count must be nonzero".into(),
        ));
    }
    let dense = mul(facts.rows.0, facts.columns.0, "sparse dense bound")?;
    if facts.operator_nonzeros.iter().any(|count| count.0 > dense) {
        return Err(ResourcePlanningError::Invalid(
            "sparse nonzeros exceed matrix dimensions".into(),
        ));
    }
    Ok(())
}

fn validate_accuracy_demand(
    input: &FeaResourcePlanningInput,
    demand: &super::contract::AccuracyResourceDemand,
    observations: &super::contract::ObservationResourceFacts,
) -> Result<(), ResourcePlanningError> {
    let refinements = u64::try_from(input.accuracy_ceiling.maximum_refinement_cycles)
        .map_err(|_| ResourcePlanningError::Overflow("accuracy refinement ceiling"))?;
    let tightenings = u64::try_from(input.accuracy_ceiling.maximum_solver_tightenings)
        .map_err(|_| ResourcePlanningError::Overflow("accuracy tightening ceiling"))?;
    let samples = u64::try_from(input.accuracy_ceiling.maximum_samples)
        .map_err(|_| ResourcePlanningError::Overflow("accuracy sample ceiling"))?;
    if demand.refinement_cycles.0 > refinements {
        return Err(ResourcePlanningError::AccuracyCeiling("refinement cycles"));
    }
    if demand.solver_tightenings.0 > tightenings {
        return Err(ResourcePlanningError::AccuracyCeiling("solver tightenings"));
    }
    if demand.samples.0 > samples || observations.maximum_samples_per_observation.0 > samples {
        return Err(ResourcePlanningError::AccuracyCeiling("samples"));
    }
    Ok(())
}

fn solve_storage(
    demand: &StructuralOperatorResourceDemand,
) -> Result<(ByteCount, ByteCount), ResourcePlanningError> {
    match demand.solve {
        SolveResourcePolicy::Iterative {
            ref preconditioner,
            ref admission,
            ..
        } => {
            let value = match preconditioner {
                PreconditionerStorage::None => 0,
                PreconditionerStorage::Diagonal => bytes(
                    demand.sparse.rows.0,
                    demand.solve.scalar_bytes(),
                    "diagonal preconditioner",
                )?,
                PreconditionerStorage::SparseCsr {
                    nonzero_upper_bound,
                    resident_bytes_upper_bound,
                } => {
                    let payload_lower_bound = sum(
                        &[
                            bytes(
                                add(demand.sparse.rows.0, 1, "preconditioner row offsets")?,
                                8,
                                "preconditioner row offsets",
                            )?,
                            bytes(nonzero_upper_bound.0, 8, "preconditioner column indices")?,
                            bytes(
                                nonzero_upper_bound.0,
                                demand.solve.scalar_bytes(),
                                "preconditioner values",
                            )?,
                        ],
                        "sparse preconditioner CSR payload",
                    )?;
                    if nonzero_upper_bound.0 == 0
                        || resident_bytes_upper_bound.0 < payload_lower_bound
                    {
                        return Err(ResourcePlanningError::Invalid(
                            "sparse preconditioner count and resident bound must be nonzero".into(),
                        ));
                    }
                    resident_bytes_upper_bound.0
                }
            };
            let factor_payload_lower_bound = sum(
                &[
                    bytes(
                        admission.factor_nonzero_upper_bound.0,
                        demand.solve.scalar_bytes(),
                        "solver-admission factor values",
                    )?,
                    bytes(
                        admission.factor_nonzero_upper_bound.0,
                        8,
                        "solver-admission factor indices",
                    )?,
                    bytes(
                        add(admission.dimension.0, 1, "solver-admission rows")?,
                        8,
                        "solver-admission row offsets",
                    )?,
                ],
                "solver-admission factor payload",
            )?;
            if admission.dimension.0 == 0
                || admission.dimension.0 > demand.sparse.rows.0
                || admission.factor_nonzero_upper_bound.0 < admission.dimension.0
                || admission.resident_bytes_upper_bound.0 < factor_payload_lower_bound
            {
                return Err(ResourcePlanningError::Invalid(
                    "solver-admission fill, work, and resident bounds must cover the symbolic factor"
                        .into(),
                ));
            }
            Ok((ByteCount(value), admission.resident_bytes_upper_bound))
        }
        SolveResourcePolicy::DirectFactorization {
            factor_nonzero_upper_bound,
            factor_bytes_upper_bound,
            ..
        } => {
            let payload_lower_bound = sum(
                &[
                    bytes(
                        add(demand.sparse.rows.0, 1, "factor row offsets")?,
                        8,
                        "factor row offsets",
                    )?,
                    bytes(factor_nonzero_upper_bound.0, 8, "factor column indices")?,
                    bytes(
                        factor_nonzero_upper_bound.0,
                        demand.solve.scalar_bytes(),
                        "factor values",
                    )?,
                ],
                "factorization sparse payload",
            )?;
            if factor_nonzero_upper_bound.0 == 0 || factor_bytes_upper_bound.0 < payload_lower_bound
            {
                return Err(ResourcePlanningError::Invalid(
                    "factorization count and resident bound must be nonzero".into(),
                ));
            }
            Ok((ByteCount(0), factor_bytes_upper_bound))
        }
        SolveResourcePolicy::NoSolve { .. }
        | SolveResourcePolicy::RestartedGmres { .. }
        | SolveResourcePolicy::ModalDenseHarmonic { .. }
        | SolveResourcePolicy::ModalSubspaceIteration { .. } => Ok((ByteCount(0), ByteCount(0))),
    }
}

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
