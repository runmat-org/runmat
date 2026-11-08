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
    ObservationPartitionPayload, ObservationPartitionPayloadCapacity, ObservationPartitionPlan,
    ObservationPartitionTarget, ObservationResult,
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
pub(super) use history::observation_history_checkpoint_payload_bytes;
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
    pub source_workspace: ObservationPartitionSourceWorkspace,
}

/// Worker storage an observation partition's source needs beside its decoded inputs before the
/// evaluator runs: source validation, a projection of the mesh and the observation topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationPartitionSourceWorkspace {
    /// The source family's counted law, in bytes.
    Counted(u64),
    /// The source family has no counted law yet (structural, sampled and volume-mesh sources).
    /// The phase reserves the direct-encoding envelope of twice its encoded inputs, the
    /// reservation these sources carried before the contribution ceiling was derived from its
    /// layout.
    InputEnvelope,
}

impl ObservationPartitionSourceWorkspace {
    /// A thermal source: the shared thermal source-validation law (source authority, then one
    /// projection, then the retained projection with the observation topology).
    pub fn thermal(
        workload: &crate::execution::ThermalWorkload,
        manifest: &crate::execution::ThermalSolverResultManifest,
        plan: &crate::results::observation::ResolvedObservationPlan,
        mesh: &runmat_meshing_core::SolverMeshArtifact,
    ) -> Result<Self, ResourcePlanningError> {
        crate::results::observation::thermal_source_validation_resident_bytes(
            workload,
            manifest,
            plan,
            mesh,
            u64::MAX,
        )
        .map(Self::Counted)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
    }

    fn resident_bytes(
        self,
        inputs: &PreparedInputResourceFacts,
    ) -> Result<u64, ResourcePlanningError> {
        match self {
            Self::Counted(bytes) => Ok(bytes),
            Self::InputEnvelope => {
                runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(
                    checked_multiply(
                        inputs.encoded_payload_bytes.0,
                        2,
                        "observation source envelope",
                    )?,
                    1,
                )
                .ok_or(ResourcePlanningError::Overflow(
                    "observation source envelope",
                ))
            }
        }
    }
}

/// Pre-execution canonical payload ceiling for one observation-partition contribution.
///
/// The ceiling comes from the payload layout the partition can write (its target's entity kind
/// and count, the source field's components) and the contribution's identity metadata. It does
/// not evaluate the observation on the driver, and the partition's inputs (mesh, workload, field
/// values) do not enter it. An accelerated execution may append one governed provider-evidence
/// record.
pub fn observation_partition_payload_bytes_upper_bound(
    source: &ObservationEvaluationRequest,
    request: &ObservationPartitionEvaluationRequest,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    source_fields: &RecoveredFieldPartitionSet,
    accelerated: bool,
) -> Result<u64, ResourcePlanningError> {
    let invalid = |error: &dyn std::fmt::Display| ResourcePlanningError::Invalid(error.to_string());
    let (layout, _) =
        observation_partition_payload_layout(source, partition_plan, partition, source_fields)?;
    let size = |value: Count| {
        usize::try_from(value.0)
            .map_err(|_| ResourcePlanningError::Overflow("observation payload capacity"))
    };
    let capacity = match layout {
        ObservationPartitionPayloadLayout::SelectedValues {
            entity_count,
            component_count,
        } => ObservationPartitionPayloadCapacity::Entities {
            extrema: false,
            entity_count: size(entity_count)?,
            values_per_entity: size(component_count)?,
        },
        ObservationPartitionPayloadLayout::ExtremaCandidates {
            candidate_count_upper_bound,
            component_count,
        } => ObservationPartitionPayloadCapacity::Entities {
            extrema: true,
            entity_count: size(candidate_count_upper_bound)?,
            values_per_entity: size(component_count)?,
        },
        // A connector resultant keeps every selected connector value in its sums.
        ObservationPartitionPayloadLayout::WeightedAccumulator { component_count } => {
            let per_entity = size(component_count)?;
            ObservationPartitionPayloadCapacity::Weighted {
                sum_length: if matches!(
                    partition.target,
                    ObservationPartitionTarget::Connectors { .. }
                ) {
                    partition
                        .target
                        .entity_count()
                        .checked_mul(per_entity)
                        .ok_or(ResourcePlanningError::Overflow(
                            "observation payload capacity",
                        ))?
                } else {
                    per_entity
                },
            }
        }
    };
    let contribution = ObservationPartitionContribution::encoded_bytes_upper_bound(
        partition_plan,
        partition,
        source_fields,
        request.event_attribution,
        capacity,
    )
    .map_err(|error| invalid(&error))?;
    checked_add(
        contribution,
        if accelerated {
            crate::results::observation::OBSERVATION_ACCELERATION_EVIDENCE_MAX_ENCODED_BYTES
        } else {
            0
        },
        "observation partition payload ceiling",
    )
}

/// The payload layout an observation partition can write, and its value count.
fn observation_partition_payload_layout(
    source: &ObservationEvaluationRequest,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    source_fields: &RecoveredFieldPartitionSet,
) -> Result<(ObservationPartitionPayloadLayout, u64), ResourcePlanningError> {
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
    Ok(match spatial_reduction {
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
    })
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
        source_workspace,
    } = planning;
    request
        .validate_for(source, partition_plan, partition, source_fields)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let (output_payload, payload_value_count) =
        observation_partition_payload_layout(source, partition_plan, partition, source_fields)?;
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
        source,
        request,
        partition_plan,
        partition,
        source_fields,
        accelerator.is_some(),
    )?;
    validate_projected_output(&output, output_payload_upper_bound, "observation partition")?;
    // The evaluator holds the selected source values, every target entity of the partition
    // (extrema select from them) and the payload values with one projection copy each. The
    // source's own storage (validation, projection, topology) precedes and spans evaluation.
    let entity_workspace = checked_multiply(
        count(
            partition.target.entity_count(),
            "observation partition entities",
        )?,
        count(
            std::mem::size_of::<crate::results::observation::ObservationEntity>(),
            "observation entity bytes",
        )?,
        "observation entity workspace",
    )?;
    let evaluator_workspace = [
        checked_multiply(source_value_count, 8, "observation source workspace")?,
        checked_multiply(payload_value_count, 16, "observation payload workspace")?,
        entity_workspace,
        source_workspace.resident_bytes(&inputs)?,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        checked_add(total, bytes, "observation evaluator workspace")
    })?;
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

/// Output ceiling of a payload derived from all of its inputs: twice the encoded inputs as
/// copied source metadata, plus record envelopes and generated values. Callers whose output is
/// built from a known part of their inputs use `copied_output_payload_bytes_upper_bound` with
/// that part instead.
pub(super) fn derived_output_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    output_record_count: u64,
    generated_numeric_value_count: u64,
) -> Result<u64, ResourcePlanningError> {
    copied_output_payload_bytes_upper_bound(
        checked_multiply(
            inputs.encoded_payload_bytes.0,
            2,
            "derived output source metadata",
        )?,
        output_record_count,
        generated_numeric_value_count,
    )
}

/// Output ceiling of a payload that copies at most `copied_bytes` of encoded records, plus one
/// record envelope per output record and the widest canonical encoding of each generated value.
pub(super) fn copied_output_payload_bytes_upper_bound(
    copied_bytes: u64,
    output_record_count: u64,
    generated_numeric_value_count: u64,
) -> Result<u64, ResourcePlanningError> {
    if output_record_count == 0 {
        return Err(ResourcePlanningError::Invalid(
            "derived output requires at least one canonical record".into(),
        ));
    }
    checked_add(
        copied_bytes,
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
