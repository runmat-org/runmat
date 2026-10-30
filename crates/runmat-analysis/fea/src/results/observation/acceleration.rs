use std::sync::atomic::{AtomicU64, Ordering};

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, ProviderElementType,
};
use runmat_analysis_core::{ExtremaTiePolicy, SpatialReduction};
use runmat_canonical_codec::encode_contract;
use runmat_execution::{identity::AttemptId, resource::AcceleratorRequest, Digest};
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use crate::{
    planning::resource::FeaResourcePlanIdentity, progress,
    results::field::RecoveredFieldPartitionSet,
};

use super::{
    acceleration_evidence::{
        ObservationAccelerationEvidence, ObservationExtremum, ObservationProviderReduction,
        OBSERVATION_PROVIDER_REDUCTION_SCHEMA_VERSION,
    },
    contract::{shape_component_count, OBSERVATION_LIMITS},
    distributed::validate_observation_partition_input,
    distributed_numeric::select_values,
    ObservationEntity, ObservationError, ObservationEventAttribution, ObservationPartition,
    ObservationPartitionContribution, ObservationPartitionPayload, ObservationPartitionPlan,
    ObservationPartitionTarget, ObservationTopology, ResolvedObservationPlan,
};

static LOCAL_OBSERVATION_ATTEMPT_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationAccelerationEligibility {
    Host,
    GovernedExactF64Extrema,
}

pub struct StructuralObservationAccelerationInput<'a> {
    pub model: &'a runmat_analysis_core::AnalysisModel,
    pub mesh: Option<&'a SolverMeshArtifact>,
    pub field: &'a RecoveredFieldPartitionSet,
    pub event_attribution: Option<ObservationEventAttribution>,
    pub allocation: &'a AcceleratorRequest,
    pub resource_plan: FeaResourcePlanIdentity,
}

pub(super) struct ObservationProviderAccelerationInput<'a> {
    pub field: &'a RecoveredFieldPartitionSet,
    pub event_attribution: Option<ObservationEventAttribution>,
    pub allocation: &'a AcceleratorRequest,
    pub resource_plan: FeaResourcePlanIdentity,
    pub provider: &'a dyn AccelProvider,
}

struct GovernedExtremaInput<'a> {
    observation: &'a super::ResolvedObservation,
    operation: ObservationExtremum,
    indices: &'a [usize],
    entities: Vec<ObservationEntity>,
}

pub fn evaluate_structural_observation_partition_accelerated(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    input: StructuralObservationAccelerationInput<'_>,
) -> Result<ObservationPartitionContribution, ObservationError> {
    let provider = runmat_accelerate_api::provider().ok_or(ObservationError::Unsupported(
        "accelerated observation evaluation requires an active provider",
    ))?;
    let topology = ObservationTopology::derive(input.model, input.mesh)?;
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "accelerated observation topology differs from its resolved plan",
        ));
    }
    evaluate_observation_partition_with_provider(
        plan,
        partition_plan,
        partition,
        ObservationProviderAccelerationInput {
            field: input.field,
            event_attribution: input.event_attribution,
            allocation: input.allocation,
            resource_plan: input.resource_plan,
            provider,
        },
    )
}

pub fn classify_observation_partition_acceleration(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    field: &RecoveredFieldPartitionSet,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationAccelerationEligibility, ObservationError> {
    Ok(
        if governed_extrema_input(plan, partition_plan, partition, field, event_attribution)?
            .is_some()
        {
            ObservationAccelerationEligibility::GovernedExactF64Extrema
        } else {
            ObservationAccelerationEligibility::Host
        },
    )
}

pub(crate) fn evaluate_observation_partition_with_provider(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    execution: ObservationProviderAccelerationInput<'_>,
) -> Result<ObservationPartitionContribution, ObservationError> {
    execution
        .allocation
        .validate()
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    if execution.allocation.count != 1 || execution.allocation.minimum_allocation_bytes == 0 {
        return Err(ObservationError::Invalid(
            "governed observation acceleration requires one bounded accelerator allocation".into(),
        ));
    }
    let governed = governed_extrema_input(
        plan,
        partition_plan,
        partition,
        execution.field,
        execution.event_attribution,
    )?
    .ok_or(ObservationError::Unsupported(
        "observation partition is outside the governed exact-f64 extrema envelope",
    ))?;
    let selected = select_values(execution.field, governed.indices, governed.observation)?;
    if selected.len() != governed.entities.len() || selected.is_empty() {
        return Err(ObservationError::Identity(
            "accelerated extrema require one selected value per entity",
        ));
    }
    let selected_values_identity =
        selected_values_identity(plan, partition_plan, partition, execution.field, &selected)?;
    let shape = [1, selected.len()];
    let input_bytes = checked_bytes(selected.len())?;
    let result_download_bytes = 16_u64;
    let required_device_bytes = governed_extrema_device_resident_bytes(selected.len())?;
    if execution.allocation.minimum_allocation_bytes < required_device_bytes {
        return Err(ObservationError::Limit(
            "governed observation accelerator memory",
        ));
    }
    let capability = execution.provider.capability_snapshot();
    if capability.revision == 0
        || capability.device.device_id != execution.provider.device_id()
        || !capability.concurrency.transactional_results
        || !capability.element_types.contains(&ProviderElementType::F64)
    {
        return Err(ObservationError::Unsupported(
            "active provider cannot execute exact-f64 observation extrema",
        ));
    }
    let attempt_id = observation_attempt_id(partition.identity.digest());
    let started = runmat_time::Instant::now();
    let input = execution
        .provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&selected),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    // Reduce the entity axis (zero-based dimension 1); the provider returns the
    // first canonical index on ties, matching ExtremaTiePolicy::FirstCanonical.
    let reduced = futures::executor::block_on(match governed.operation {
        ObservationExtremum::Minimum => execution.provider.reduce_min_dim(&input, 1),
        ObservationExtremum::Maximum => execution.provider.reduce_max_dim(&input, 1),
    })
    .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    let value = download_scalar(execution.provider, &reduced.values)?;
    let index = download_scalar(execution.provider, &reduced.indices)?;
    let execution_ns = u64::try_from(started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1);
    let index = exact_one_based_index(index, selected.len())?;
    if selected[index] != value {
        return Err(ObservationError::Identity(
            "accelerated extrema value and selected source disagree",
        ));
    }
    let provider_evidence = ObservationProviderReduction {
        schema_version: OBSERVATION_PROVIDER_REDUCTION_SCHEMA_VERSION,
        attempt_id,
        input_identity: selected_values_identity,
        provider_device_id: capability.device.device_id,
        provider_capability_schema_version: capability.schema_version,
        provider_capability_revision: capability.revision,
        operation: governed.operation,
        input_bytes,
        output_bytes: result_download_bytes,
        dispatches: 1,
        execution_ns,
    };
    let evidence = ObservationAccelerationEvidence::new(
        execution.resource_plan,
        selected_values_identity,
        execution.allocation.clone(),
        provider_evidence,
        input_bytes,
        result_download_bytes,
        2,
    )?;
    ObservationPartitionContribution::new(
        partition_plan,
        partition,
        execution.field,
        execution.event_attribution,
        Some(evidence),
        ObservationPartitionPayload::ExtremaCandidates {
            entities: vec![governed.entities[index].clone()],
            values: vec![value],
        },
    )
}

fn governed_extrema_input<'a>(
    plan: &'a ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &'a ObservationPartition,
    field: &RecoveredFieldPartitionSet,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<Option<GovernedExtremaInput<'a>>, ObservationError> {
    let observation = validate_observation_partition_input(
        plan,
        partition_plan,
        partition,
        field,
        event_attribution,
    )?;
    if shape_component_count(&observation.result_value_shape)? != 1
        || observation.spec.complex_representation
            == runmat_analysis_core::ComplexRepresentation::Complex
    {
        return Ok(None);
    }
    let Some(reduction) = observation.spec.reduction.as_ref() else {
        return Ok(None);
    };
    if reduction.extrema_ties != ExtremaTiePolicy::FirstCanonical {
        return Ok(None);
    }
    let operation = match reduction.spatial {
        SpatialReduction::Minimum => ObservationExtremum::Minimum,
        SpatialReduction::Maximum => ObservationExtremum::Maximum,
        _ => return Ok(None),
    };
    let (indices, entities) = match &partition.target {
        ObservationPartitionTarget::Nodes {
            indices,
            identities,
        } => (
            indices.as_slice(),
            identities
                .iter()
                .copied()
                .map(ObservationEntity::Node)
                .collect(),
        ),
        ObservationPartitionTarget::Elements {
            domain,
            indices,
            identities,
        } => (
            indices.as_slice(),
            identities
                .iter()
                .copied()
                .map(|identity| super::contract::observation_element_entity(*domain, identity))
                .collect(),
        ),
        _ => return Ok(None),
    };
    Ok(Some(GovernedExtremaInput {
        observation,
        operation,
        indices,
        entities,
    }))
}

fn download_scalar(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<f64, ObservationError> {
    let downloaded = futures::executor::block_on(provider.download_numeric(handle))
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    let HostNumericDataOwned::F64(values) = downloaded.data else {
        return Err(ObservationError::Identity(
            "accelerated extrema changed the admitted f64 representation",
        ));
    };
    if downloaded.storage != GpuTensorStorage::Real
        || downloaded.shape != [1, 1]
        || values.len() != 1
        || !values[0].is_finite()
    {
        return Err(ObservationError::Identity(
            "accelerated extrema returned an invalid scalar",
        ));
    }
    Ok(values[0])
}

fn exact_one_based_index(value: f64, length: usize) -> Result<usize, ObservationError> {
    if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > length as f64 {
        return Err(ObservationError::Identity(
            "accelerated extrema returned an invalid canonical index",
        ));
    }
    Ok(value as usize - 1)
}

fn selected_values_identity(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    field: &RecoveredFieldPartitionSet,
    values: &[f64],
) -> Result<Digest, ObservationError> {
    encode_contract(
        super::contract::OBSERVATION_PREFIX,
        "observation-accelerated-selected-values",
        &(
            plan.identity,
            partition_plan.identity,
            partition.identity,
            field.identity,
            values,
        ),
        OBSERVATION_LIMITS,
    )
    .map(Digest::sha256)
    .map_err(ObservationError::from)
}

fn checked_bytes(length: usize) -> Result<u64, ObservationError> {
    u64::try_from(length)
        .ok()
        .and_then(|length| length.checked_mul(8))
        .ok_or(ObservationError::Limit("accelerated observation bytes"))
}

pub(crate) fn governed_extrema_device_resident_bytes(
    selected_entity_count: usize,
) -> Result<u64, ObservationError> {
    checked_bytes(selected_entity_count)?
        .checked_add(16)
        .ok_or(ObservationError::Limit(
            "governed observation accelerator memory",
        ))
}

fn observation_attempt_id(partition: Digest) -> AttemptId {
    let sequence = LOCAL_OBSERVATION_ATTEMPT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let parent = progress::attempt_id();
    AttemptId::derive(&[
        b"fea-observation-provider-attempt",
        parent.as_ref().map_or(&[][..], |value| value.bytes()),
        partition.bytes(),
        &sequence.to_be_bytes(),
    ])
}
