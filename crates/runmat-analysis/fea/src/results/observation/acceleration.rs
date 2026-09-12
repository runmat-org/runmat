use std::sync::atomic::{AtomicU64, Ordering};

use runmat_accelerate_api::{
    AccelProvider, DenseReductionExecutionEvidence, DenseReductionOperation, DenseReductionRequest,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, ProviderCancellationToken, ProviderCostQuery, ProviderElementType,
    ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderRepresentation, ProviderResidency, ProviderStorage, ProviderWorkload,
    DENSE_REDUCTION_EVIDENCE_SCHEMA_VERSION, GOVERNED_DENSE_REDUCTION_OPERATION,
};
use runmat_analysis_core::{ExtremaTiePolicy, SpatialReduction};
use runmat_canonical_codec::encode_contract;
use runmat_execution::{identity::AttemptId, CandidatePreparationState, Digest};
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use crate::{progress, results::field::RecoveredFieldPartitionSet};

use super::{
    contract::{shape_component_count, OBSERVATION_LIMITS},
    distributed::validate_observation_partition_input,
    distributed_numeric::select_values,
    ObservationEntity, ObservationError, ObservationEventAttribution, ObservationPartition,
    ObservationPartitionContribution, ObservationPartitionPayload, ObservationPartitionPlan,
    ObservationPartitionTarget, ResolvedObservationPlan, StructuralObservationTopology,
};

static LOCAL_OBSERVATION_ATTEMPT_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationAccelerationEligibility {
    Host,
    GovernedExactF64Extrema,
}

struct GovernedExtremaInput<'a> {
    observation: &'a super::ResolvedObservation,
    operation: DenseReductionOperation,
    indices: &'a [usize],
    entities: Vec<ObservationEntity>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationAccelerationEvidenceIdentity(Digest);

impl ObservationAccelerationEvidenceIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationAccelerationEvidenceWire")]
pub struct ObservationAccelerationEvidence {
    pub schema_version: u16,
    pub identity: ObservationAccelerationEvidenceIdentity,
    pub selected_values_identity: Digest,
    pub provider: DenseReductionExecutionEvidence,
    pub input_upload_bytes: u64,
    pub result_download_bytes: u64,
    pub terminal_synchronizations: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationAccelerationEvidenceWire {
    schema_version: u16,
    identity: ObservationAccelerationEvidenceIdentity,
    selected_values_identity: Digest,
    provider: DenseReductionExecutionEvidence,
    input_upload_bytes: u64,
    result_download_bytes: u64,
    terminal_synchronizations: u32,
}

impl TryFrom<ObservationAccelerationEvidenceWire> for ObservationAccelerationEvidence {
    type Error = ObservationError;

    fn try_from(wire: ObservationAccelerationEvidenceWire) -> Result<Self, Self::Error> {
        let evidence = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            selected_values_identity: wire.selected_values_identity,
            provider: wire.provider,
            input_upload_bytes: wire.input_upload_bytes,
            result_download_bytes: wire.result_download_bytes,
            terminal_synchronizations: wire.terminal_synchronizations,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl ObservationAccelerationEvidence {
    pub(super) fn new(
        selected_values_identity: Digest,
        provider: DenseReductionExecutionEvidence,
        input_upload_bytes: u64,
        result_download_bytes: u64,
        terminal_synchronizations: u32,
    ) -> Result<Self, ObservationError> {
        let mut evidence = Self {
            schema_version: 1,
            identity: ObservationAccelerationEvidenceIdentity(Digest::sha256([])),
            selected_values_identity,
            provider,
            input_upload_bytes,
            result_download_bytes,
            terminal_synchronizations,
        };
        evidence.identity = evidence.derive_identity()?;
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        let operation_is_extrema = matches!(
            self.provider.operation,
            DenseReductionOperation::Minimum { .. } | DenseReductionOperation::Maximum { .. }
        );
        if self.schema_version != 1
            || self.provider.schema_version != DENSE_REDUCTION_EVIDENCE_SCHEMA_VERSION
            || self.provider.input_identity != self.selected_values_identity
            || !operation_is_extrema
            || self.provider.used_host_fallback
            || self.provider.cancellation_observed
            || self.provider.cancellation_checks == 0
            || self.provider.dispatches == 0
            || self.provider.host_upload_bytes != 0
            || self.provider.host_download_bytes != 0
            || self.provider.host_synchronizations != 0
            || self.input_upload_bytes != self.provider.input_bytes
            || self.result_download_bytes != self.provider.output_bytes
            || self.terminal_synchronizations != 2
            || self.identity != self.derive_identity()?
        {
            return Err(ObservationError::Invalid(
                "observation acceleration evidence is inconsistent".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ObservationAccelerationEvidenceIdentity, ObservationError> {
        let bytes = encode_contract(
            super::contract::OBSERVATION_PREFIX,
            "observation-acceleration-evidence-identity",
            &(
                self.schema_version,
                self.selected_values_identity,
                &self.provider,
                self.input_upload_bytes,
                self.result_download_bytes,
                self.terminal_synchronizations,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ObservationAccelerationEvidenceIdentity(Digest::sha256(
            bytes,
        )))
    }
}

pub fn evaluate_structural_observation_partition_accelerated(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    model: &runmat_analysis_core::AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    field: &RecoveredFieldPartitionSet,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationPartitionContribution, ObservationError> {
    let provider = runmat_accelerate_api::provider().ok_or(ObservationError::Unsupported(
        "accelerated observation evaluation requires an active provider",
    ))?;
    let topology = StructuralObservationTopology::derive(model, mesh)?;
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "accelerated observation topology differs from its resolved plan",
        ));
    }
    evaluate_observation_partition_with_provider(
        plan,
        partition_plan,
        partition,
        field,
        event_attribution,
        provider,
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
    field: &RecoveredFieldPartitionSet,
    event_attribution: Option<ObservationEventAttribution>,
    provider: &dyn AccelProvider,
) -> Result<ObservationPartitionContribution, ObservationError> {
    let governed =
        governed_extrema_input(plan, partition_plan, partition, field, event_attribution)?.ok_or(
            ObservationError::Unsupported(
                "observation partition is outside the governed exact-f64 extrema envelope",
            ),
        )?;
    let selected = select_values(field, governed.indices, governed.observation)?;
    if selected.len() != governed.entities.len() || selected.is_empty() {
        return Err(ObservationError::Identity(
            "accelerated extrema require one selected value per entity",
        ));
    }
    let selected_values_identity =
        selected_values_identity(plan, partition_plan, partition, field, &selected)?;
    let shape = [1, selected.len()];
    let input_bytes = checked_bytes(selected.len())?;
    let result_download_bytes = 16_u64;
    let feasibility = dense_reduction_feasibility(&shape)?;
    let capability = provider.capability_snapshot();
    if !provider.query_feasibility(&feasibility).is_supported()
        || capability.revision == 0
        || capability.device.device_id != provider.device_id()
        || !capability.concurrency.transactional_results
        || !capability.element_types.contains(&ProviderElementType::F64)
        || !capability.supports_operation(
            &ProviderOperationIdentity::new(GOVERNED_DENSE_REDUCTION_OPERATION),
            ProviderOperationFamily::Reduction,
        )
    {
        return Err(ObservationError::Unsupported(
            "active provider cannot execute exact-f64 governed observation extrema",
        ));
    }
    let cost = provider
        .estimate_cost(&ProviderCostQuery {
            operation: feasibility,
            preparation: CandidatePreparationState::Cold,
            required_upload_bytes: input_bytes,
            required_download_bytes: result_download_bytes,
            downstream_materialization: true,
        })
        .ok_or(ObservationError::Unsupported(
            "active provider cannot estimate governed observation extrema",
        ))?;
    if cost.cost.checked_total_ns().is_none() {
        return Err(ObservationError::Invalid(
            "provider observation cost estimate overflows".into(),
        ));
    }

    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&selected),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    let mut handles = ProviderHandles::new(provider);
    handles.track(&input);
    let cancellation =
        ProviderCancellationToken::from_shared(progress::cancellation_token().shared_flag());
    let request = DenseReductionRequest {
        input: &input,
        operation: governed.operation,
        attempt_id: observation_attempt_id(partition.identity.digest()),
        input_identity: selected_values_identity,
        provider_capability_schema_version: capability.schema_version,
        provider_capability_revision: capability.revision,
        cancellation: &cancellation,
    };
    let result = futures::executor::block_on(provider.execute_dense_reduction(&request))
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    handles.track(&result.values);
    if let Some(indices) = &result.indices {
        handles.track(indices);
    }
    result
        .validate(request, &capability)
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    let value = download_scalar(provider, &result.values)?;
    let index = download_scalar(
        provider,
        result.indices.as_ref().ok_or(ObservationError::Identity(
            "accelerated extrema omitted its canonical index",
        ))?,
    )?;
    let index = exact_one_based_index(index, selected.len())?;
    if selected[index] != value {
        return Err(ObservationError::Identity(
            "accelerated extrema value and selected source disagree",
        ));
    }
    let evidence = ObservationAccelerationEvidence::new(
        selected_values_identity,
        result.evidence,
        input_bytes,
        result_download_bytes,
        2,
    )?;
    ObservationPartitionContribution::new(
        partition_plan,
        partition,
        field,
        event_attribution,
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
        SpatialReduction::Minimum => DenseReductionOperation::Minimum { dimension: 1 },
        SpatialReduction::Maximum => DenseReductionOperation::Maximum { dimension: 1 },
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
            indices,
            identities,
        } => (
            indices.as_slice(),
            identities
                .iter()
                .copied()
                .map(ObservationEntity::Element)
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

fn dense_reduction_feasibility(
    shape: &[usize; 2],
) -> Result<ProviderFeasibilityQuery, ObservationError> {
    let shape = shape
        .iter()
        .map(|extent| u64::try_from(*extent).map_err(|_| ObservationError::Limit("provider shape")))
        .collect::<Result<Vec<_>, _>>()?;
    let representation = |residency| ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.clone(),
        residency,
    };
    let scalar_output = ProviderRepresentation {
        shape: vec![1, 1],
        ..representation(ProviderResidency::Device)
    };
    Ok(ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(GOVERNED_DENSE_REDUCTION_OPERATION),
        family: ProviderOperationFamily::Reduction,
        inputs: vec![representation(ProviderResidency::Device)],
        outputs: vec![scalar_output.clone(), scalar_output],
        workload: ProviderWorkload {
            elements: shape.iter().copied().try_fold(1_u64, u64::checked_mul),
            flops: None,
            batch: Some(1),
        },
    })
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

struct ProviderHandles<'a> {
    provider: &'a dyn AccelProvider,
    handles: Vec<GpuTensorHandle>,
}

impl<'a> ProviderHandles<'a> {
    fn new(provider: &'a dyn AccelProvider) -> Self {
        Self {
            provider,
            handles: Vec::with_capacity(3),
        }
    }

    fn track(&mut self, handle: &GpuTensorHandle) {
        self.handles.push(handle.clone());
    }
}

impl Drop for ProviderHandles<'_> {
    fn drop(&mut self) {
        for handle in self.handles.iter().rev() {
            let _ = self.provider.free(handle);
        }
    }
}
