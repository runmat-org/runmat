use super::*;

pub(super) fn build_observation_partition_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &ObservationPartitionResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.partition_entity_count.0 == 0
        || demand.source_entity_count.0 == 0
        || demand.source_partition_count.0 == 0
        || demand.source_value_count.0 == 0
        || demand.operation_count_upper_bound.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "observation partition resource cardinalities must be nonzero".into(),
        ));
    }
    let observations = super::ObservationResourceFacts {
        maximum_samples_per_observation: super::Count(1),
        produced_sample_count: super::Count(1),
        retained_sample_count: super::Count(0),
        operation_count: demand.operation_count_upper_bound,
        peak_resident_bytes: demand.output.construction_resident_bytes_admission_budget,
        stored_bytes: demand.output.encoded_payload_bytes_upper_bound,
        transfer_bytes: demand.inputs.transport_closure_bytes,
        collective_count: super::Count(0),
    };
    build_post_process_resource_plan(
        input,
        PostProcessAccounting {
            topology_digest: demand.topology_digest,
            inputs: &demand.inputs,
            workspace_bytes: demand.evaluator_workspace_bytes_upper_bound,
            retained_bytes: ByteCount(0),
            outputs: &[&demand.output],
            observations,
            accuracy_demand: super::AccuracyResourceDemand {
                refinement_cycles: super::Count(0),
                solver_tightenings: super::Count(0),
                samples: super::Count(1),
            },
        },
    )
}

pub(super) fn build_observation_reduction_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &ObservationReductionResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.contribution_count.0 == 0
        || demand.contribution_encoded_bytes.0 == 0
        || demand.contribution_value_count.0 == 0
        || demand.output_value_count.0 == 0
        || demand.accumulator_value_count.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "observation reduction resource cardinalities must be nonzero".into(),
        ));
    }
    build_post_process_resource_plan(
        input,
        PostProcessAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            demand.workspace_bytes_upper_bound,
            ByteCount(0),
            &[&demand.output],
        ),
    )
}

pub(super) fn build_observation_history_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &ObservationHistoryCheckpointResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.predecessor_present != (demand.predecessor_sample_count.0 > 0)
        || demand.retained_sample_count.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "history resource predecessor and retained-sample facts are inconsistent".into(),
        ));
    }
    build_post_process_resource_plan(
        input,
        PostProcessAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            demand.transition_workspace_bytes_upper_bound,
            ByteCount(0),
            &[&demand.output],
        ),
    )
}

pub(super) fn build_observation_assembly_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &ObservationAssemblyResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.result_count.0 == 0
        || demand.result_encoded_bytes.0 == 0
        || demand.evidence_count.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "observation assembly resource cardinalities must be nonzero".into(),
        ));
    }
    build_post_process_resource_plan(
        input,
        PostProcessAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            demand.ordering_workspace_bytes_upper_bound,
            ByteCount(0),
            &[&demand.output],
        ),
    )
}

pub(super) fn build_frequency_result_assembly_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &StructuralFrequencyResultAssemblyResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.entry_count.0 == 0
        || demand.entry_encoded_bytes.0 == 0
        || demand.physical_dof_count.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "frequency result assembly resource cardinalities must be nonzero".into(),
        ));
    }
    build_post_process_resource_plan(
        input,
        PostProcessAccounting::host_only(
            demand.scope_digest,
            &demand.inputs,
            demand.ordering_workspace_bytes_upper_bound,
            ByteCount(0),
            &[&demand.result_output, &demand.receipt_output],
        ),
    )
}

pub(super) fn build_response_spectrum_assembly_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &StructuralModalResponseSpectrumAssemblyResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let missing_mass_values = [
        demand.missing_mass_reduced_dof_count.0,
        demand.missing_mass_system_workspace_bytes_upper_bound.0,
        demand.missing_mass_total_iteration_count_upper_bound.0,
    ];
    let missing_mass_is_absent = demand.missing_mass_static_solve_count.0 == 0;
    let missing_mass_tuple_is_valid = if missing_mass_is_absent {
        missing_mass_values.iter().all(|value| *value == 0)
    } else {
        missing_mass_values.iter().all(|value| *value > 0)
    };
    let expected_iteration_count = if missing_mass_is_absent {
        0
    } else {
        let dimension = usize::try_from(demand.missing_mass_reduced_dof_count.0)
            .map_err(|_| ResourcePlanningError::Overflow("missing-mass reduced dimension"))?;
        let per_solve = u64::try_from(
            crate::pipeline::homogeneous_static::homogeneous_static_iteration_limit(dimension),
        )
        .map_err(|_| ResourcePlanningError::Overflow("missing-mass static iteration limit"))?;
        demand
            .missing_mass_static_solve_count
            .0
            .checked_mul(per_solve)
            .ok_or(ResourcePlanningError::Overflow(
                "missing-mass total static iterations",
            ))?
    };
    if demand.direction_count.0 == 0
        || demand.direction_count.0 > 3
        || demand.direction_encoded_bytes.0 == 0
        || demand.physical_dof_count.0 == 0
        || demand.missing_mass_static_solve_count.0 > demand.direction_count.0
        || demand.missing_mass_reduced_dof_count.0 > demand.physical_dof_count.0
        || !missing_mass_tuple_is_valid
        || demand.missing_mass_total_iteration_count_upper_bound.0 != expected_iteration_count
    {
        return Err(ResourcePlanningError::Invalid(
            "response-spectrum assembly resource cardinalities are invalid".into(),
        ));
    }
    build_post_process_resource_plan(
        input,
        PostProcessAccounting::host_only(
            demand.scope_digest,
            &demand.inputs,
            ByteCount(
                demand
                    .ordering_workspace_bytes_upper_bound
                    .0
                    .checked_add(demand.field_recovery_workspace_bytes_upper_bound.0)
                    .and_then(|value| {
                        value.checked_add(demand.missing_mass_system_workspace_bytes_upper_bound.0)
                    })
                    .ok_or(ResourcePlanningError::Overflow(
                        "response-spectrum assembly, field-recovery, and missing-mass workspace",
                    ))?,
            ),
            ByteCount(0),
            &[
                &demand.result_output,
                &demand.receipt_output,
                &demand.field_recovery_output,
            ],
        ),
    )
}

pub(super) fn build_random_response_assembly_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &StructuralRandomResponseAssemblyResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.task_count.0 == 0
        || demand.interval_count.0 < demand.task_count.0
        || demand.interval_encoded_bytes.0 == 0
        || demand.response_value_count.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "random-response assembly resource cardinalities are invalid".into(),
        ));
    }
    build_post_process_resource_plan(
        input,
        PostProcessAccounting::host_only(
            demand.scope_digest,
            &demand.inputs,
            demand.ordering_workspace_bytes_upper_bound,
            ByteCount(0),
            &[&demand.result_output, &demand.receipt_output],
        ),
    )
}

struct PostProcessAccounting<'a> {
    topology_digest: Digest,
    inputs: &'a PreparedInputResourceFacts,
    workspace_bytes: ByteCount,
    retained_bytes: ByteCount,
    outputs: &'a [&'a PreparedOutputResourceFacts],
    observations: super::ObservationResourceFacts,
    accuracy_demand: super::AccuracyResourceDemand,
}

impl<'a> PostProcessAccounting<'a> {
    fn host_only(
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

fn build_post_process_resource_plan(
    input: FeaResourcePlanningInput,
    accounting: PostProcessAccounting<'_>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let resource_plan_input = implicit_resource_plan_input_facts()?;
    let output_set_bytes = input
        .output_set
        .as_ref()
        .ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "post-processing output set resource facts are missing".into(),
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
            "post-processing input and output residency facts are invalid".into(),
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
            "post-processing output construction",
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
        "post-processing peak memory",
    )?;
    let total_memory = add(
        input.worker.fixed_memory_bytes.0,
        numerical_memory,
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
            "post-processing input and resource-plan relay bytes",
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
