//! Shared additive accounting for a monolithic host phase.
use super::*;

pub(super) struct HostPhaseAccounting<'a> {
    pub(super) topology_digest: Digest,
    pub(super) inputs: &'a PreparedInputResourceFacts,
    pub(super) workspace_bytes: ByteCount,
    pub(super) retained_bytes: ByteCount,
    pub(super) outputs: &'a [&'a PreparedOutputResourceFacts],
    pub(super) observations: super::ObservationResourceFacts,
    pub(super) accuracy_demand: super::AccuracyResourceDemand,
}

impl<'a> HostPhaseAccounting<'a> {
    pub(super) fn host_only(
        topology_digest: Digest,
        inputs: &'a PreparedInputResourceFacts,
        workspace_bytes: ByteCount,
        retained_bytes: ByteCount,
        outputs: &'a [&'a PreparedOutputResourceFacts],
    ) -> Self {
        Self {
            topology_digest,
            inputs,
            workspace_bytes,
            retained_bytes,
            outputs,
            observations: super::ObservationResourceFacts {
                maximum_samples_per_observation: super::Count(0),
                produced_sample_count: super::Count(0),
                retained_sample_count: super::Count(0),
                operation_count: super::Count(0),
                peak_resident_bytes: ByteCount(0),
                stored_bytes: ByteCount(0),
                transfer_bytes: ByteCount(0),
                collective_count: super::Count(0),
            },
            accuracy_demand: super::AccuracyResourceDemand {
                refinement_cycles: super::Count(0),
                solver_tightenings: super::Count(0),
                samples: super::Count(0),
            },
        }
    }
}

pub(super) fn build_host_phase_resource_plan(
    input: FeaResourcePlanningInput,
    accounting: HostPhaseAccounting<'_>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let resource_plan_input = implicit_resource_plan_input_facts()?;
    let output_set_bytes = input
        .output_set
        .as_ref()
        .ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "host phase output set resource facts are missing".into(),
            )
        })?
        .encoded_bytes_upper_bound;
    if accounting.topology_digest == Digest::sha256([])
        || accounting.inputs.encoded_payload_bytes.0 == 0
        || accounting.inputs.decoded_resident_bytes_admission_budget.0 == 0
        || accounting.outputs.is_empty()
        || accounting.outputs.iter().any(|output| {
            output.encoded_payload_bytes_upper_bound.0 == 0
                || output.transport_closure_bytes_upper_bound.0
                    < output.encoded_payload_bytes_upper_bound.0
                || output.construction_resident_bytes_admission_budget.0
                    < output.encoded_payload_bytes_upper_bound.0
        })
    {
        return Err(ResourcePlanningError::Invalid(
            "host phase input and output residency facts are invalid".into(),
        ));
    }
    validate_accuracy_demand(
        &input,
        &accounting.accuracy_demand,
        &accounting.observations,
    )?;
    let output_construction = accounting.outputs.iter().try_fold(0_u64, |total, output| {
        add(
            total,
            output.construction_resident_bytes_admission_budget.0,
            "host phase output construction",
        )
    })?;
    let numerical_memory = sum(
        &[
            accounting.inputs.decoded_resident_bytes_admission_budget.0,
            resource_plan_input
                .decoded_resident_bytes_admission_budget
                .0,
            accounting.workspace_bytes.0,
            accounting.retained_bytes.0,
            output_construction,
            output_set_bytes.0,
        ],
        "host phase peak memory",
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
        &[
            input.worker.fixed_memory_bytes.0,
            numerical_memory,
            pool_stacks,
        ],
        "worker peak memory",
    )?;
    let total_scratch = input.worker.fixed_scratch_bytes.0;
    let result_artifact_bytes = accounting.outputs.iter().try_fold(0_u64, |total, output| {
        add(
            total,
            output.transport_closure_bytes_upper_bound.0,
            "published numerical artifacts",
        )
    })?;
    let artifact_bytes = add(
        result_artifact_bytes,
        output_set_bytes.0,
        "published artifacts",
    )?;
    let total_egress = artifact_bytes;
    let partition = partition_facts(
        &input,
        &PartitioningPolicy::Monolithic,
        numerical_memory,
        total_memory,
        0,
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
            accounting.inputs.transport_closure_bytes.0,
            resource_plan_input.transport_closure_bytes.0,
            "host phase input and resource-plan relay bytes",
        )?,
        accelerators: input.worker.accelerators.iter().map(Into::into).collect(),
        required_capabilities: input.worker.required_capabilities.clone(),
    };
    request
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let zero = ByteCount(0);
    let breakdown = ResourceBreakdown {
        input_encoded_payload_bytes: accounting.inputs.encoded_payload_bytes,
        input_transport_closure_bytes: accounting.inputs.transport_closure_bytes,
        input_decoded_resident_bytes: accounting.inputs.decoded_resident_bytes_admission_budget,
        resource_plan_transport_closure_bytes: resource_plan_input.transport_closure_bytes,
        resource_plan_decoded_resident_bytes: resource_plan_input
            .decoded_resident_bytes_admission_budget,
        output_set_bytes,
        topology_bytes: zero,
        csr_row_offset_bytes: zero,
        csr_column_index_bytes: zero,
        csr_value_bytes: zero,
        assembly_contribution_bytes: zero,
        assembly_scratch_bytes: zero,
        workspace_bytes: accounting.workspace_bytes,
        preconditioner_bytes: zero,
        factorization_bytes: zero,
        constraint_bytes: zero,
        phase_resident_bytes: ByteCount(output_construction),
        phase_scratch_bytes: ByteCount(0),
        phase_retained_bytes: accounting.retained_bytes,
        field_recovery_requirement_count: Count(0),
        field_recovery_source_entity_count: Count(0),
        field_recovery_output_entity_count: Count(0),
        field_recovery_topology_incidence_count: Count(0),
        field_recovery_arithmetic_operation_count: Count(0),
        result_artifact_bytes: ByteCount(result_artifact_bytes),
        observation_maximum_samples_per_observation: accounting
            .observations
            .maximum_samples_per_observation,
        observation_produced_sample_count: accounting.observations.produced_sample_count,
        observation_retained_sample_count: accounting.observations.retained_sample_count,
        observation_operation_count: accounting.observations.operation_count,
        observation_peak_bytes: accounting.observations.peak_resident_bytes,
        observation_stored_bytes: accounting.observations.stored_bytes,
        observation_transfer_bytes: accounting.observations.transfer_bytes,
        result_egress_bytes: ByteCount(total_egress),
        checkpoint_egress_bytes: ByteCount(0),
        total_egress_bytes: ByteCount(total_egress),
        observation_collectives: accounting.observations.collective_count,
        checkpoint_bytes: ByteCount(0),
        artifact_overhead_bytes: ByteCount(0),
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
