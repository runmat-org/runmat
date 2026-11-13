use runmat_analysis_core::ObservationId;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::field::RecoveredFieldSample;
use crate::results::observation::{
    BoundedObservationHistory, ObservationHistoryCheckpoint, ObservationHistoryCheckpointIdentity,
    ObservationHistoryEntry, ObservationHistoryStream, ObservationPartitionEvidenceIdentity,
    ObservationResult, ObservationResultIdentity, ResolvedObservationPlan,
    ResolvedObservationPlanIdentity,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ObservationHistoryCheckpointRequestIdentity,
};

const HISTORY_EXECUTION_PREFIX: &[u8] = b"runmat-analysis-history-execution/v2\0";
const HISTORY_EXECUTION_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "ObservationHistoryCheckpointRequestWire"
)]
pub struct ObservationHistoryCheckpointRequest {
    pub schema_version: u16,
    pub identity: ObservationHistoryCheckpointRequestIdentity,
    pub plan: ResolvedObservationPlan,
    pub producer: super::StepIdentity,
    pub observation_id: ObservationId,
    pub predecessor: Option<ObservationHistoryCheckpointIdentity>,
    pub sample_ordinal: u64,
    pub sample_coordinate: RecoveredFieldSample,
    pub sample: ObservationResultIdentity,
    pub sample_partition_evidence: ObservationPartitionEvidenceIdentity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationHistoryCheckpointRequestWire {
    schema_version: u16,
    identity: ObservationHistoryCheckpointRequestIdentity,
    plan: ResolvedObservationPlan,
    producer: super::StepIdentity,
    observation_id: ObservationId,
    predecessor: Option<ObservationHistoryCheckpointIdentity>,
    sample_ordinal: u64,
    sample_coordinate: RecoveredFieldSample,
    sample: ObservationResultIdentity,
    sample_partition_evidence: ObservationPartitionEvidenceIdentity,
}

impl TryFrom<ObservationHistoryCheckpointRequestWire> for ObservationHistoryCheckpointRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ObservationHistoryCheckpointRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            plan: wire.plan,
            producer: wire.producer,
            observation_id: wire.observation_id,
            predecessor: wire.predecessor,
            sample_ordinal: wire.sample_ordinal,
            sample_coordinate: wire.sample_coordinate,
            sample: wire.sample,
            sample_partition_evidence: wire.sample_partition_evidence,
        };
        request.validate()?;
        Ok(request)
    }
}

impl ObservationHistoryCheckpointRequest {
    /// Reconstruct one scheduler request from an independently validated complete history.
    /// Borrow the plan identity rather than cloning the resolved plan for every prefix.
    pub(crate) fn published_sample_identity(
        plan: ResolvedObservationPlanIdentity,
        producer: super::StepIdentity,
        observation_id: &ObservationId,
        predecessor: Option<ObservationHistoryCheckpointIdentity>,
        sample_ordinal: u64,
        sample: &ObservationHistoryEntry,
        evidence: ObservationPartitionEvidenceIdentity,
    ) -> AnalysisExecutionResult<ObservationHistoryCheckpointRequestIdentity> {
        derive_request_identity(
            plan,
            producer,
            observation_id,
            predecessor,
            sample_ordinal,
            (sample.coordinate, sample.result, evidence),
        )
    }

    pub fn new(
        plan: ResolvedObservationPlan,
        producer: super::StepIdentity,
        observation_id: ObservationId,
        predecessor: Option<ObservationHistoryCheckpointIdentity>,
        sample_ordinal: u64,
        result: &ObservationResult,
    ) -> AnalysisExecutionResult<Self> {
        result
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let sample_coordinate = result.sample;
        let sample = result.logical_identity;
        let sample_partition_evidence = result.partition_evidence.identity;
        let identity = derive_request_identity(
            plan.identity,
            producer,
            &observation_id,
            predecessor,
            sample_ordinal,
            (sample_coordinate, sample, sample_partition_evidence),
        )?;
        let request = Self {
            schema_version: 2,
            identity,
            plan,
            producer,
            observation_id,
            predecessor,
            sample_ordinal,
            sample_coordinate,
            sample,
            sample_partition_evidence,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let observation = self
            .plan
            .observations
            .iter()
            .find(|observation| observation.spec.observation_id == self.observation_id)
            .ok_or_else(|| {
                AnalysisExecutionError::Invalid(
                    "history checkpoint request names an unresolved observation".into(),
                )
            })?;
        if self.schema_version != 2
            || matches!(
                observation.spec.sampling,
                runmat_analysis_core::SamplingPolicy::Final
            )
            || self.identity
                != derive_request_identity(
                    self.plan.identity,
                    self.producer,
                    &self.observation_id,
                    self.predecessor,
                    self.sample_ordinal,
                    (
                        self.sample_coordinate,
                        self.sample,
                        self.sample_partition_evidence,
                    ),
                )?
        {
            return Err(AnalysisExecutionError::Identity(
                "history checkpoint request identity or observation is invalid",
            ));
        }
        Ok(())
    }

    pub fn advance(
        &self,
        previous: Option<ObservationHistoryCheckpoint>,
        result: ObservationResult,
    ) -> AnalysisExecutionResult<ObservationHistoryCheckpoint> {
        self.validate_advance_inputs(previous.as_ref(), &result)?;
        let mut history = match (self.predecessor, previous) {
            (None, None) => BoundedObservationHistory::new(&self.plan, &self.observation_id),
            (Some(expected), Some(previous)) if previous.manifest().identity == expected => {
                BoundedObservationHistory::resume(&self.plan, &self.observation_id, previous)
            }
            _ => {
                return Err(AnalysisExecutionError::Identity(
                    "history checkpoint predecessor differs from its request",
                ))
            }
        }
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        history
            .push(result.sample, result)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        history
            .checkpoint(self.predecessor)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))
    }

    /// Validates one history transition without cloning or constructing its retained output.
    pub fn validate_advance_inputs(
        &self,
        previous: Option<&ObservationHistoryCheckpoint>,
        result: &ObservationResult,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        result
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if result.logical_identity != self.sample
            || result.sample != self.sample_coordinate
            || result.partition_evidence.identity != self.sample_partition_evidence
        {
            return Err(AnalysisExecutionError::Identity(
                "history checkpoint sample, coordinate, or execution evidence differs from its request",
            ));
        }
        let observation = self
            .plan
            .observations
            .iter()
            .find(|observation| observation.spec.observation_id == self.observation_id)
            .ok_or_else(|| {
                AnalysisExecutionError::Invalid(
                    "history checkpoint request names an unresolved observation".into(),
                )
            })?;
        let expected_stream = ObservationHistoryStream::new(
            self.plan.identity,
            self.plan.topology_digest,
            self.plan.independent_domain.clone(),
            observation,
            observation.sample_bounds.maximum_produced_samples,
            observation.sample_bounds.maximum_retained_samples,
        )
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let expected_ordinal =
            previous.map_or(0, |checkpoint| checkpoint.manifest().observed_sample_count);
        match (self.predecessor, previous) {
            (None, None) => {}
            (Some(expected), Some(previous)) => {
                previous
                    .validate()
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                if previous.manifest().identity != expected
                    || previous.manifest().stream != expected_stream
                {
                    return Err(AnalysisExecutionError::Identity(
                        "history checkpoint predecessor differs from its resolved stream",
                    ));
                }
            }
            _ => {
                return Err(AnalysisExecutionError::Identity(
                    "history checkpoint predecessor differs from its request",
                ))
            }
        }
        if self.sample_ordinal != expected_ordinal {
            return Err(AnalysisExecutionError::Identity(
                "history checkpoint sample ordinal does not follow its predecessor",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            HISTORY_EXECUTION_PREFIX,
            "history-checkpoint-request",
            self,
            HISTORY_EXECUTION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            HISTORY_EXECUTION_PREFIX,
            "history-checkpoint-request",
            bytes,
            HISTORY_EXECUTION_LIMITS,
        )?)
    }
}

fn derive_request_identity(
    plan: ResolvedObservationPlanIdentity,
    producer: super::StepIdentity,
    observation_id: &ObservationId,
    predecessor: Option<ObservationHistoryCheckpointIdentity>,
    sample_ordinal: u64,
    sample: (
        RecoveredFieldSample,
        ObservationResultIdentity,
        ObservationPartitionEvidenceIdentity,
    ),
) -> AnalysisExecutionResult<ObservationHistoryCheckpointRequestIdentity> {
    ObservationHistoryCheckpointRequestIdentity::derive(
        "history-checkpoint-request",
        &(
            plan,
            producer,
            observation_id,
            predecessor,
            sample_ordinal,
            sample,
        ),
    )
}
