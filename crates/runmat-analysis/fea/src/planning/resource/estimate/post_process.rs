use super::host_phase::{build_host_phase_resource_plan, HostPhaseAccounting};
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
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting {
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
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
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
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            ByteCount(history_transition_workspace(
                demand.transition_workspace_bytes_upper_bound.0,
                demand.execution_evidence_encoded_bytes_upper_bound.0,
                demand.execution_evidence_workspace_bytes_upper_bound.0,
                demand.output.encoded_payload_bytes_upper_bound.0,
            )?),
            ByteCount(0),
            &[&demand.output],
        ),
    )
}

fn history_transition_workspace(
    transition: u64,
    evidence_encoded: u64,
    evidence_workspace: u64,
    output: u64,
) -> Result<u64, ResourcePlanningError> {
    if evidence_encoded == 0 || evidence_workspace < evidence_encoded || output < evidence_encoded {
        return Err(ResourcePlanningError::Invalid(
            "history execution evidence exceeds its output or workspace admission".into(),
        ));
    }
    add(
        transition,
        evidence_workspace,
        "history execution evidence transition workspace",
    )
}

pub(super) fn build_observation_history_finalization_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &crate::planning::resource::ObservationHistoryFinalizationResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.retained_sample_count.0 == 0 || demand.retained_value_count.0 == 0 {
        return Err(ResourcePlanningError::Invalid(
            "history finalization requires nonempty retained samples and values".into(),
        ));
    }
    let mut outputs = vec![&demand.history_output];
    if let Some(output) = &demand.series_output {
        outputs.push(output);
    }
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.topology_digest,
            &demand.inputs,
            demand.workspace_bytes_upper_bound,
            ByteCount(0),
            &outputs,
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
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
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
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.scope_digest,
            &demand.inputs,
            demand.ordering_workspace_bytes_upper_bound,
            ByteCount(0),
            &[&demand.result_output, &demand.receipt_output],
        ),
    )
}

pub(super) fn build_transient_result_assembly_resource_plan(
    input: FeaResourcePlanningInput,
    demand: &StructuralTransientResultAssemblyResourceDemand,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if demand.segment_count.0 == 0
        || demand.sample_count.0 == 0
        || demand.physical_dof_count.0 == 0
        || demand.ordering_workspace_bytes_upper_bound.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "transient result assembly resource cardinalities must be nonzero".into(),
        ));
    }
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
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
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
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
                &demand.state_profiles_output,
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
    build_host_phase_resource_plan(
        input,
        HostPhaseAccounting::host_only(
            demand.scope_digest,
            &demand.inputs,
            demand.ordering_workspace_bytes_upper_bound,
            ByteCount(0),
            &[
                &demand.result_output,
                &demand.receipt_output,
                &demand.state_profiles_output,
            ],
        ),
    )
}

#[cfg(test)]
mod history_admission_tests {
    use super::history_transition_workspace;

    #[test]
    fn execution_evidence_is_added_to_transition_workspace() {
        assert_eq!(
            history_transition_workspace(100, 200, 500, 300).unwrap(),
            600
        );
    }

    #[test]
    fn execution_evidence_cannot_use_fixed_overhead_as_its_budget() {
        let evidence_bytes = 128 * 1024;
        assert!(
            history_transition_workspace(100, evidence_bytes, evidence_bytes, 64 * 1024).is_err()
        );
        assert!(
            history_transition_workspace(100, evidence_bytes, 64 * 1024, evidence_bytes).is_err()
        );
        assert!(history_transition_workspace(100, 0, 0, 100).is_err());
    }

    #[test]
    fn execution_evidence_workspace_overflow_is_rejected() {
        assert!(history_transition_workspace(u64::MAX, 1, 1, 1).is_err());
    }
}
