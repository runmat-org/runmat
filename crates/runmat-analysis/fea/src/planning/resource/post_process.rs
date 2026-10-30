use std::collections::BTreeSet;

use runmat_analysis_core::AccuracyResourceCeiling;
use runmat_execution::resource::{AcceleratorRequest, Capability};

use crate::execution::{
    ObservationAssemblyRequest, ObservationEvaluationRequest, ObservationHistoryCheckpointRequest,
    ObservationPartitionEvaluationRequest, ObservationPartitionReductionRequest,
    StructuralFrequencyEntryResult, StructuralFrequencyResultAssemblyRequest,
};
use crate::results::field::RecoveredFieldPartitionSet;
use crate::results::observation::{
    ObservationHistoryCheckpoint, ObservationPartition, ObservationPartitionContribution,
    ObservationPartitionPayload, ObservationPartitionPlan, ObservationResult,
};

use super::{
    build_resource_plan, AcceleratorRequirement, ByteCount, Count, FeaResourceDemand,
    FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput, ObservationAssemblyResourceDemand,
    ObservationHistoryCheckpointResourceDemand, ObservationPartitionPayloadLayout,
    ObservationPartitionResourceDemand, ObservationReductionResourceDemand,
    PreparedInputResourceFacts, PreparedOutputResourceFacts, PreparedOutputSetResourceFacts,
    ResourcePlanningError, StructuralFrequencyResultAssemblyResourceDemand, WorkerResourcePolicy,
    RESOURCE_ESTIMATION_MODEL_REVISION,
};

const POST_PROCESS_CPU_MILLICORES: u32 = 1_000;
const POST_PROCESS_WALL_MILLISECONDS: u64 = 3_600_000;
const POST_PROCESS_FIXED_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const POST_PROCESS_FIXED_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
const DERIVED_OUTPUT_RECORD_OVERHEAD_BYTES: u64 = 64 * 1024;
const CANONICAL_NUMERIC_BYTES_UPPER_BOUND: u64 = 9;

mod history;
pub use history::{
    build_observation_history_checkpoint_resource_plan,
    observation_history_checkpoint_payload_bytes_upper_bound,
};

pub struct ObservationPartitionResourcePlanning<'a> {
    pub source: &'a ObservationEvaluationRequest,
    pub request: &'a ObservationPartitionEvaluationRequest,
    pub partition_plan: &'a ObservationPartitionPlan,
    pub partition: &'a ObservationPartition,
    pub source_fields: &'a RecoveredFieldPartitionSet,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
    pub accelerator: Option<&'a AcceleratorRequest>,
}

/// Pre-execution canonical payload ceiling for one observation-partition contribution.
///
/// The bound is derived only from admitted inputs and typed cardinalities. It deliberately does
/// not evaluate the observation on the driver. An accelerated execution may append one governed
/// provider-evidence record.
pub fn observation_partition_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    source_fields: &RecoveredFieldPartitionSet,
    accelerated: bool,
) -> Result<u64, ResourcePlanningError> {
    let source_value_count = source_fields
        .partitions
        .iter()
        .try_fold(0_u64, |total, item| {
            let values = item.storage.as_host_f64().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "observation resource planning requires host f64 field partitions".into(),
                )
            })?;
            checked_add(
                total,
                count(values.len(), "observation source values")?,
                "observation source values",
            )
        })?;
    checked_add(
        derived_output_payload_bytes_upper_bound(inputs, 1, source_value_count)?,
        if accelerated {
            crate::results::observation::OBSERVATION_ACCELERATION_EVIDENCE_MAX_ENCODED_BYTES
        } else {
            0
        },
        "observation partition payload ceiling",
    )
}

pub fn build_observation_partition_resource_plan(
    planning: ObservationPartitionResourcePlanning<'_>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let ObservationPartitionResourcePlanning {
        source,
        request,
        partition_plan,
        partition,
        source_fields,
        inputs,
        output,
        output_set,
        accelerator,
    } = planning;
    request
        .validate_for(source, partition_plan, partition, source_fields)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let observation = source
        .plan
        .observations
        .iter()
        .find(|observation| observation.spec.observation_id == partition_plan.observation_id)
        .ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "observation partition is absent from its resolved plan".into(),
            )
        })?;
    let component_count = count(
        source_fields
            .definition
            .storage_component_count()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        "observation payload components",
    )?;
    let entity_count = count(
        partition.target.entity_count(),
        "observation partition entities",
    )?;
    let spatial_reduction = observation
        .spec
        .reduction
        .as_ref()
        .map(|reduction| reduction.spatial)
        .unwrap_or(runmat_analysis_core::SpatialReduction::None);
    let (output_payload, payload_value_count) = match spatial_reduction {
        runmat_analysis_core::SpatialReduction::None => (
            ObservationPartitionPayloadLayout::SelectedValues {
                entity_count: Count(entity_count),
                component_count: Count(component_count),
            },
            checked_multiply(entity_count, component_count, "selected observation values")?,
        ),
        runmat_analysis_core::SpatialReduction::Minimum
        | runmat_analysis_core::SpatialReduction::Maximum => (
            ObservationPartitionPayloadLayout::ExtremaCandidates {
                candidate_count_upper_bound: Count(entity_count),
                component_count: Count(component_count),
            },
            checked_multiply(entity_count, component_count, "observation extrema values")?,
        ),
        _ => (
            ObservationPartitionPayloadLayout::WeightedAccumulator {
                component_count: Count(component_count),
            },
            checked_add(
                checked_multiply(component_count, 2, "observation accumulator values")?,
                1,
                "observation accumulator measure",
            )?,
        ),
    };
    let source_value_count = source_fields
        .partitions
        .iter()
        .try_fold(0_u64, |total, item| {
            let values = item.storage.as_host_f64().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "observation resource planning requires host f64 field partitions".into(),
                )
            })?;
            checked_add(
                total,
                count(values.len(), "observation source values")?,
                "observation source values",
            )
        })?;
    let output_payload_upper_bound = observation_partition_payload_bytes_upper_bound(
        &inputs,
        source_fields,
        accelerator.is_some(),
    )?;
    validate_projected_output(&output, output_payload_upper_bound, "observation partition")?;
    let evaluator_workspace = checked_add(
        checked_multiply(source_value_count, 8, "observation source workspace")?,
        checked_multiply(payload_value_count, 16, "observation payload workspace")?,
        "observation evaluator workspace",
    )?;
    let required_accelerator_bytes =
        crate::results::observation::governed_extrema_device_resident_bytes(
            partition.target.entity_count(),
        )
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let worker = post_process_worker_with_accelerator(accelerator, required_accelerator_bytes)?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralObservationPartition(request.identity()),
        demand: FeaResourceDemand::StructuralObservationPartition(
            ObservationPartitionResourceDemand {
                topology_digest: source.plan.topology_digest,
                partition_entity_count: Count(count(
                    partition.target.entity_count(),
                    "observation partition entities",
                )?),
                source_entity_count: Count(count(
                    source_fields.source_indices().count(),
                    "observation source entities",
                )?),
                source_partition_count: Count(count(
                    source_fields.partitions.len(),
                    "observation source partitions",
                )?),
                source_value_count: Count(source_value_count),
                output_payload,
                operation_count_upper_bound: Count(
                    partition_plan.resource_cost.operation_count.max(1),
                ),
                inputs,
                evaluator_workspace_bytes_upper_bound: ByteCount(evaluator_workspace),
                output,
            },
        ),
        output_set: Some(output_set),
        worker,
        accuracy_ceiling: post_process_ceiling(),
    })
}

pub fn build_observation_reduction_resource_plan(
    source: &ObservationEvaluationRequest,
    request: &ObservationPartitionReductionRequest,
    partition_plan: &ObservationPartitionPlan,
    contributions: &[ObservationPartitionContribution],
    inputs: PreparedInputResourceFacts,
    output: PreparedOutputResourceFacts,
    output_set: PreparedOutputSetResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    request
        .validate_for(source, partition_plan, contributions)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let (contribution_encoded_bytes, contribution_value_count) =
        contributions
            .iter()
            .try_fold((0_u64, 0_u64), |(encoded, values), contribution| {
                let bytes = contribution
                    .canonical_bytes()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
                Ok::<_, ResourcePlanningError>((
                    checked_add(encoded, byte_len(&bytes)?, "observation contribution bytes")?,
                    checked_add(
                        values,
                        contribution_payload_value_count(&contribution.payload)?,
                        "observation contribution values",
                    )?,
                ))
            })?;
    let output_entity_count = contributions
        .iter()
        .try_fold(0_u64, |total, contribution| {
            let entities = match &contribution.payload {
                ObservationPartitionPayload::SelectedValues { entities, .. }
                | ObservationPartitionPayload::ExtremaCandidates { entities, .. } => entities.len(),
                ObservationPartitionPayload::WeightedReduction { .. }
                | ObservationPartitionPayload::PairedWeightedReduction { .. } => 1,
            };
            checked_add(
                total,
                count(entities, "observation output entities")?,
                "observation output entities",
            )
        })?;
    let output_value_count = contribution_value_count.max(1);
    let output_payload_upper_bound =
        observation_reduction_payload_bytes_upper_bound(&inputs, contributions)?;
    validate_projected_output(&output, output_payload_upper_bound, "observation reduction")?;
    let accumulator_value_count = reduction_accumulator_value_count(contributions)?;
    let workspace = checked_add(
        contribution_encoded_bytes,
        checked_multiply(
            checked_add(
                accumulator_value_count,
                output_value_count,
                "observation reduction workspace values",
            )?,
            8,
            "observation reduction numerical workspace",
        )?,
        "observation reduction workspace",
    )?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralObservationReduction(request.identity()),
        demand: FeaResourceDemand::StructuralObservationReduction(
            ObservationReductionResourceDemand {
                topology_digest: source.plan.topology_digest,
                contribution_count: Count(count(contributions.len(), "observation contributions")?),
                contribution_encoded_bytes: ByteCount(contribution_encoded_bytes),
                contribution_value_count: Count(contribution_value_count),
                output_entity_count: Count(output_entity_count),
                output_value_count: Count(output_value_count),
                accumulator_value_count: Count(accumulator_value_count),
                inputs,
                workspace_bytes_upper_bound: ByteCount(workspace),
                output,
            },
        ),
        output_set: Some(output_set),
        worker: post_process_worker(),
        accuracy_ceiling: post_process_ceiling(),
    })
}

pub fn observation_reduction_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    contributions: &[ObservationPartitionContribution],
) -> Result<u64, ResourcePlanningError> {
    let value_count = contributions
        .iter()
        .try_fold(0_u64, |total, contribution| {
            checked_add(
                total,
                contribution_payload_value_count(&contribution.payload)?,
                "observation reduction output values",
            )
        })?;
    derived_output_payload_bytes_upper_bound(inputs, 1, value_count.max(1))
}

pub fn build_observation_assembly_resource_plan(
    source: &ObservationEvaluationRequest,
    request: &ObservationAssemblyRequest,
    results: &[ObservationResult],
    inputs: PreparedInputResourceFacts,
    output: PreparedOutputResourceFacts,
    output_set: PreparedOutputSetResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    request
        .validate_for(source, results)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let output_payload_upper_bound =
        observation_artifact_payload_bytes_upper_bound(&inputs, results.len())?;
    validate_projected_output(&output, output_payload_upper_bound, "observation assembly")?;
    let result_encoded_bytes = results.iter().try_fold(0_u64, |total, result| {
        checked_add(
            total,
            byte_len(
                &result
                    .canonical_bytes()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            )?,
            "observation result bytes",
        )
    })?;
    let ordering_workspace = checked_multiply(
        count(results.len(), "observation results")?,
        3 * 32,
        "observation assembly ordering workspace",
    )?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralObservationAssembly(request.identity()),
        demand: FeaResourceDemand::StructuralObservationAssembly(
            ObservationAssemblyResourceDemand {
                topology_digest: source.plan.topology_digest,
                result_count: Count(count(results.len(), "observation results")?),
                result_encoded_bytes: ByteCount(result_encoded_bytes),
                evidence_count: Count(count(
                    request.partition_evidence.len(),
                    "observation evidence",
                )?),
                inputs,
                ordering_workspace_bytes_upper_bound: ByteCount(ordering_workspace),
                output,
            },
        ),
        output_set: Some(output_set),
        worker: post_process_worker(),
        accuracy_ceiling: post_process_ceiling(),
    })
}

pub fn observation_artifact_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    result_count: usize,
) -> Result<u64, ResourcePlanningError> {
    derived_output_payload_bytes_upper_bound(
        inputs,
        checked_add(
            count(result_count, "observation results")?,
            1,
            "observation artifact records",
        )?,
        0,
    )
}

/// Builds the host-only envelope for canonical structural-frequency result fan-in without
/// assembling either output on the driver before admission.
pub fn build_structural_frequency_result_assembly_resource_plan(
    request: &StructuralFrequencyResultAssemblyRequest,
    entries: &[StructuralFrequencyEntryResult],
    inputs: PreparedInputResourceFacts,
    result_output: PreparedOutputResourceFacts,
    receipt_output: PreparedOutputResourceFacts,
    output_set: PreparedOutputSetResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    request
        .validate_for(entries)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let entry_encoded_bytes = entries.iter().try_fold(0_u64, |total, entry| {
        let bytes = entry
            .canonical_bytes()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        checked_add(total, byte_len(&bytes)?, "frequency entry payload bytes")
    })?;
    let entry_count = count(entries.len(), "frequency result entry count")?;
    let output_record_count = checked_add(entry_count, 1, "frequency result output records")?;
    let result_payload_bytes =
        structural_frequency_result_payload_bytes_upper_bound(&inputs, output_record_count)?;
    let receipt_payload_bytes =
        structural_frequency_receipt_payload_bytes_upper_bound(&inputs, output_record_count)?;
    validate_projected_output(
        &result_output,
        result_payload_bytes,
        "frequency result assembly",
    )?;
    validate_projected_output(
        &receipt_output,
        receipt_payload_bytes,
        "frequency receipt assembly",
    )?;
    let ordering_workspace_bytes =
        checked_multiply(entry_count, 3 * 8, "frequency result ordering workspace")?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralFrequencyResultAssembly(request.identity()),
        demand: FeaResourceDemand::StructuralFrequencyResultAssembly(
            StructuralFrequencyResultAssemblyResourceDemand {
                scope_digest: request.scope().identity().digest(),
                entry_count: Count(entry_count),
                entry_encoded_bytes: ByteCount(entry_encoded_bytes),
                physical_dof_count: Count(count(
                    request.scope().physical_dof_count(),
                    "frequency result physical DOFs",
                )?),
                inputs,
                ordering_workspace_bytes_upper_bound: ByteCount(ordering_workspace_bytes),
                result_output,
                receipt_output,
            },
        ),
        output_set: Some(output_set),
        worker: post_process_worker(),
        accuracy_ceiling: post_process_ceiling(),
    })
}

pub fn structural_frequency_result_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    output_record_count: u64,
) -> Result<u64, ResourcePlanningError> {
    derived_output_payload_bytes_upper_bound(inputs, output_record_count, 0)
}

pub fn structural_frequency_receipt_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    output_record_count: u64,
) -> Result<u64, ResourcePlanningError> {
    derived_output_payload_bytes_upper_bound(inputs, output_record_count, 0)
}

fn post_process_worker_with_accelerator(
    accelerator: Option<&AcceleratorRequest>,
    required_device_bytes: u64,
) -> Result<WorkerResourcePolicy, ResourcePlanningError> {
    let mut worker = post_process_worker();
    if let Some(accelerator) = accelerator {
        accelerator
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if accelerator.count != 1 || accelerator.minimum_allocation_bytes < required_device_bytes {
            return Err(ResourcePlanningError::Invalid(
                "observation acceleration requires one sufficient bounded allocation".into(),
            ));
        }
        worker
            .required_capabilities
            .insert(Capability::Accelerator(accelerator.class.clone()));
        worker.accelerators.push(AcceleratorRequirement {
            class: accelerator.class.clone(),
            count: accelerator.count,
            memory_bytes_each: ByteCount(accelerator.minimum_allocation_bytes),
        });
    }
    Ok(worker)
}

fn contribution_payload_value_count(
    payload: &ObservationPartitionPayload,
) -> Result<u64, ResourcePlanningError> {
    let values = match payload {
        ObservationPartitionPayload::SelectedValues { values, .. }
        | ObservationPartitionPayload::ExtremaCandidates { values, .. } => values.len(),
        ObservationPartitionPayload::WeightedReduction {
            linear_sum,
            square_sum,
            ..
        } => linear_sum
            .len()
            .checked_add(square_sum.len())
            .and_then(|value| value.checked_add(1))
            .ok_or(ResourcePlanningError::Overflow(
                "observation contribution values",
            ))?,
        ObservationPartitionPayload::PairedWeightedReduction { .. } => 4,
    };
    count(values, "observation contribution values")
}

fn reduction_accumulator_value_count(
    contributions: &[ObservationPartitionContribution],
) -> Result<u64, ResourcePlanningError> {
    let first = contributions.first().ok_or_else(|| {
        ResourcePlanningError::Invalid("observation reduction requires contributions".into())
    })?;
    match &first.payload {
        ObservationPartitionPayload::SelectedValues { .. } => {
            contributions.iter().try_fold(0_u64, |total, contribution| {
                match &contribution.payload {
                    ObservationPartitionPayload::SelectedValues { values, .. } => checked_add(
                        total,
                        count(values.len(), "selected observation values")?,
                        "selected observation accumulator values",
                    ),
                    _ => Err(ResourcePlanningError::Invalid(
                        "observation reduction payload layouts differ".into(),
                    )),
                }
            })
        }
        ObservationPartitionPayload::ExtremaCandidates { .. } => {
            contributions.iter().try_fold(0_u64, |total, contribution| {
                match &contribution.payload {
                    ObservationPartitionPayload::ExtremaCandidates { values, .. } => checked_add(
                        total,
                        count(values.len(), "observation extrema values")?,
                        "observation extrema accumulator values",
                    ),
                    _ => Err(ResourcePlanningError::Invalid(
                        "observation reduction payload layouts differ".into(),
                    )),
                }
            })
        }
        ObservationPartitionPayload::WeightedReduction {
            linear_sum,
            square_sum,
            ..
        } => {
            if linear_sum.len() != square_sum.len()
                || contributions.iter().any(|contribution| {
                    !matches!(
                        &contribution.payload,
                        ObservationPartitionPayload::WeightedReduction {
                            linear_sum: other_linear,
                            square_sum: other_square,
                            ..
                        } if other_linear.len() == linear_sum.len()
                            && other_square.len() == square_sum.len()
                    )
                })
            {
                return Err(ResourcePlanningError::Invalid(
                    "observation reduction payload layouts differ".into(),
                ));
            }
            // Two compensated component arrays plus one compensated measure. Each compensated
            // scalar retains its running sum and correction.
            checked_add(
                checked_multiply(
                    count(linear_sum.len(), "weighted observation components")?,
                    4,
                    "weighted observation component accumulators",
                )?,
                2,
                "weighted observation measure accumulator",
            )
        }
        ObservationPartitionPayload::PairedWeightedReduction { .. } => {
            if contributions.iter().any(|contribution| {
                !matches!(
                    contribution.payload,
                    ObservationPartitionPayload::PairedWeightedReduction { .. }
                )
            }) {
                return Err(ResourcePlanningError::Invalid(
                    "observation reduction payload layouts differ".into(),
                ));
            }
            // Four compensated scalars (two integrals, two measures): sum and correction each.
            Ok(8)
        }
    }
}

pub(super) fn post_process_worker() -> WorkerResourcePolicy {
    WorkerResourcePolicy {
        cpu_millicores: POST_PROCESS_CPU_MILLICORES,
        maximum_wall_milliseconds: POST_PROCESS_WALL_MILLISECONDS,
        fixed_memory_bytes: ByteCount(POST_PROCESS_FIXED_MEMORY_BYTES),
        fixed_scratch_bytes: ByteCount(POST_PROCESS_FIXED_SCRATCH_BYTES),
        required_capabilities: BTreeSet::new(),
        accelerators: Vec::new(),
    }
}

/// Post-process worker sized by the study's admitted local execution plan. Field recovery
/// kernels run their element loops on this pool (`ordered_map_range`, scoped chunks).
pub(super) fn post_process_worker_on(
    local_execution: super::LocalExecutionPlan,
) -> Result<WorkerResourcePolicy, ResourcePlanningError> {
    Ok(WorkerResourcePolicy {
        cpu_millicores: local_execution
            .checked_cpu_millicores()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        ..post_process_worker()
    })
}

pub(super) fn post_process_ceiling() -> AccuracyResourceCeiling {
    AccuracyResourceCeiling {
        maximum_refinement_cycles: 0,
        maximum_solver_tightenings: 0,
        maximum_samples: 1_000_000,
        maximum_wall_time_milliseconds: POST_PROCESS_WALL_MILLISECONDS,
        // Post-processing phases carry no authored memory ceiling; placement admits them
        // against the machine-capacity authority like every other phase.
        maximum_resident_bytes: None,
    }
}

fn byte_len(bytes: &[u8]) -> Result<u64, ResourcePlanningError> {
    count(bytes.len(), "canonical payload bytes")
}

pub(super) fn derived_output_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    output_record_count: u64,
    generated_numeric_value_count: u64,
) -> Result<u64, ResourcePlanningError> {
    if output_record_count == 0 {
        return Err(ResourcePlanningError::Invalid(
            "derived output requires at least one canonical record".into(),
        ));
    }
    checked_add(
        checked_multiply(
            inputs.encoded_payload_bytes.0,
            2,
            "derived output source metadata",
        )?,
        checked_add(
            checked_multiply(
                output_record_count,
                DERIVED_OUTPUT_RECORD_OVERHEAD_BYTES,
                "derived output record overhead",
            )?,
            checked_multiply(
                generated_numeric_value_count,
                CANONICAL_NUMERIC_BYTES_UPPER_BOUND,
                "derived output numeric values",
            )?,
            "derived output generated payload",
        )?,
        "derived output payload ceiling",
    )
}

pub(super) fn validate_projected_output(
    output: &PreparedOutputResourceFacts,
    payload_bytes_upper_bound: u64,
    domain: &'static str,
) -> Result<(), ResourcePlanningError> {
    if output.encoded_payload_bytes_upper_bound.0 != payload_bytes_upper_bound
        || output.transport_closure_bytes_upper_bound.0 < payload_bytes_upper_bound
        || output.construction_resident_bytes_admission_budget.0 < payload_bytes_upper_bound
    {
        return Err(ResourcePlanningError::Invalid(format!(
            "{domain} output facts differ from the pre-execution payload ceiling"
        )));
    }
    Ok(())
}

fn count(value: usize, label: &'static str) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| ResourcePlanningError::Overflow(label))
}

fn checked_add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn checked_multiply(
    left: u64,
    right: u64,
    label: &'static str,
) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerator_preflight_requires_the_complete_device_buffer_peak() {
        let exact = AcceleratorRequest {
            class: runmat_execution::resource::AcceleratorClass::new("test-gpu").unwrap(),
            count: 1,
            minimum_allocation_bytes: 24,
            provider: None,
            required_features: Default::default(),
        };
        let worker = post_process_worker_with_accelerator(Some(&exact), 24).unwrap();
        assert_eq!(worker.accelerators[0].memory_bytes_each, ByteCount(24));

        let mut short = exact;
        short.minimum_allocation_bytes -= 1;
        assert!(post_process_worker_with_accelerator(Some(&short), 24).is_err());
    }
}
