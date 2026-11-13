//! Operator-phase accounting, including independently reserved producer metadata.
use super::*;

pub(super) fn build_operator_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &StructuralOperatorResourceDemand,
    inputs: Option<&PreparedInputResourceFacts>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    build_operator_resource_plan_with_persistent_storage(input, demand, inputs, ByteCount(0))
}

pub(super) fn build_operator_resource_plan_with_persistent_storage(
    input: FeaResourcePlanningInput,
    demand: &StructuralOperatorResourceDemand,
    inputs: Option<&PreparedInputResourceFacts>,
    additional_persistent_bytes: ByteCount,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    validate_operator_demand(&input, demand)?;
    if matches!(
        input.demand,
        FeaResourceDemand::StructuralLinearPipeline(_)
            | FeaResourceDemand::StructuralNonlinearPipeline(_)
    ) && inputs.is_some_and(|facts| {
        facts.encoded_payload_bytes.0 == 0
                // Transport counts unique chunks; repeated logical payloads need not
                // increase that closure, unlike their independent materializations.
                || facts.transport_closure_bytes.0 == 0
                || facts.decoded_resident_bytes_admission_budget < facts.encoded_payload_bytes
    }) {
        return Err(ResourcePlanningError::Invalid(
            "operator prepared input transport or materialization is underreserved".into(),
        ));
    }
    if let FeaResourceDemand::StructuralNonlinearPipeline(pipeline) = &input.demand {
        if let Some(block) = &pipeline.block_transport {
            let transport = add(
                block.output.transport_closure_bytes_upper_bound.0,
                block.receipt_output.transport_closure_bytes_upper_bound.0,
                "nonlinear outputs",
            )?;
            let construction = add(
                block.output.construction_resident_bytes_admission_budget.0,
                block
                    .receipt_output
                    .construction_resident_bytes_admission_budget
                    .0,
                "nonlinear output construction",
            )?;
            if block.request != demand.numerical_scope_digest
                || !prepared_output_is_valid(&block.output)
                || !prepared_output_is_valid(&block.receipt_output)
                || block.receipt_output.encoded_payload_bytes_upper_bound.0
                    != crate::execution::NONLINEAR_SOLID_RECEIPT_MAX_ENCODED_BYTES
                || input
                    .output_set
                    .as_ref()
                    .is_none_or(|set| set.encoded_bytes_upper_bound.0 == 0)
                || demand.publication.result_artifact_bytes.0 < transport
                || pipeline.additional_persistent_bytes.0 < construction
            {
                return Err(ResourcePlanningError::Invalid(
                    "nonlinear block transport authority is incomplete".into(),
                ));
            }
        }
    }
    let catalog = match &input.demand {
        FeaResourceDemand::StructuralLinearPipeline(pipeline) => {
            Some(&pipeline.state_profiles_output)
        }
        FeaResourceDemand::StructuralNonlinearPipeline(pipeline) => {
            Some(&pipeline.state_profiles_output)
        }
        _ => None,
    };
    let (catalog_construction, catalog_transport) = if let Some(output) = catalog {
        // The catalog is encoded directly, as every published output.
        let materialization =
            runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(
                output.encoded_payload_bytes_upper_bound.0,
                1,
            )
            .ok_or(ResourcePlanningError::Overflow(
                "static catalog materialization",
            ))?;
        let minimum = add(
            materialization,
            output.transport_closure_bytes_upper_bound.0,
            "static catalog construction",
        )?;
        if !prepared_output_is_valid(output)
            || output.construction_resident_bytes_admission_budget.0 < minimum
        {
            return Err(ResourcePlanningError::Invalid(
                "static state profile catalog output is underreserved".into(),
            ));
        }
        (
            output.construction_resident_bytes_admission_budget.0,
            output.transport_closure_bytes_upper_bound.0,
        )
    } else {
        (0, 0)
    };
    let resource_plan_input =
        super::implicit_input::operator_resource_plan_input_facts(&input.phase)?;
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
            demand.excitation_workspace.resident_bytes_upper_bound.0,
            additional_persistent_bytes.0,
        ],
        "persistent numerical memory",
    )?;
    let solve_phase = sum(
        &[
            workspace_bytes.0,
            preconditioner_bytes.0,
            factorization_bytes.0,
            demand.solve.transpose_index_bytes_upper_bound().0,
        ],
        "solve phase memory",
    )?;
    let recovery_phase = sum(
        &[
            phase_resident_bytes.0,
            catalog_construction,
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
    let input_transport = sum(
        &[
            inputs.map_or(0, |facts| facts.transport_closure_bytes.0),
            resource_plan_input.transport_closure_bytes.0,
        ],
        "retained original input transport",
    )?;
    let compiler_peak = super::constraint_compilation::original_compiler_peak(&input)?;
    let compiler_memory = if compiler_peak == 0 {
        0
    } else {
        sum(
            &[
                compiler_peak,
                input_decoded_resident_bytes.0,
                resource_plan_input
                    .decoded_resident_bytes_admission_budget
                    .0,
            ],
            "original compiler and retained decoded inputs",
        )?
    };
    let task_peak = add(
        numerical_memory.max(compiler_memory),
        input_transport,
        "original inputs and sequential operator phases",
    )?;
    let pool_stacks = mul(
        u64::from(if input.worker.cpu_millicores > 1_000 {
            input.worker.cpu_millicores / 1_000
        } else {
            0
        }),
        runmat_cpu_execution::WORKER_STACK_BYTES,
        "persistent CPU pool stacks",
    )?;
    let total_memory = sum(
        &[input.worker.fixed_memory_bytes.0, task_peak, pool_stacks],
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
            catalog_transport,
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
            catalog_transport,
            demand.publication.checkpoint_egress_bytes.0,
            output_set_bytes.0,
        ],
        "result and checkpoint egress",
    )?;

    if input.worker.maximum_wall_milliseconds
        > input.accuracy_ceiling.maximum_wall_time_milliseconds
    {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "wall time milliseconds",
            required: input.worker.maximum_wall_milliseconds,
            maximum: input.accuracy_ceiling.maximum_wall_time_milliseconds,
        });
    }
    validate_accuracy_demand(&input, &demand.accuracy_demand, &demand.observations)?;
    let retained_decoded = add(
        input_decoded_resident_bytes.0,
        resource_plan_input
            .decoded_resident_bytes_admission_budget
            .0,
        "partition original decoded inputs",
    )?;
    let mut partition = partition_facts(
        &input,
        &demand.partitioning,
        numerical_memory - retained_decoded,
        total_memory,
        phase_scratch,
        total_scratch,
    )?;
    // Compilation uses the complete original inputs before operator partitioning.
    // Reserve it per worker; partitioning cannot divide a full-source construction.
    if matches!(
        demand.partitioning,
        PartitioningPolicy::OperatorDomain { .. }
    ) {
        partition.fixed_memory = sum(
            &[
                partition.fixed_memory,
                compiler_peak,
                retained_decoded,
                input_transport,
            ],
            "partition original compiler and transport",
        )?;
    }
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
        phase_retained_bytes: ByteCount(add(
            demand.phase_resources.retained_bytes.0,
            additional_persistent_bytes.0,
            "nonlinear persistent phase storage",
        )?),
        field_recovery_requirement_count: demand.field_recovery.requirement_count,
        field_recovery_source_entity_count: demand.field_recovery.source_entity_count,
        field_recovery_output_entity_count: demand.field_recovery.output_entity_count,
        field_recovery_topology_incidence_count: demand.field_recovery.topology_incidence_count,
        field_recovery_arithmetic_operation_count: demand.field_recovery.arithmetic_operation_count,
        result_artifact_bytes: ByteCount(add(
            demand.publication.result_artifact_bytes.0,
            catalog_transport,
            "static result and catalog artifacts",
        )?),
        observation_maximum_samples_per_observation: demand
            .observations
            .maximum_samples_per_observation,
        observation_produced_sample_count: demand.observations.produced_sample_count,
        observation_retained_sample_count: demand.observations.retained_sample_count,
        observation_operation_count: demand.observations.operation_count,
        observation_peak_bytes: demand.observations.peak_resident_bytes,
        observation_stored_bytes: demand.observations.stored_bytes,
        observation_transfer_bytes: demand.observations.transfer_bytes,
        result_egress_bytes: ByteCount(add(
            demand.publication.result_egress_bytes.0,
            catalog_transport,
            "static result and catalog egress",
        )?),
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
