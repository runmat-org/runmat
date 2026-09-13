use runmat_analysis_core::ObservationId;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::field::RecoveredFieldSample;
use crate::results::observation::{
    BoundedObservationHistory, ObservationHistoryCheckpoint, ObservationHistoryCheckpointIdentity,
    ObservationResult, ObservationResultIdentity, ResolvedObservationPlan,
    ResolvedObservationPlanIdentity,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ObservationHistoryCheckpointRequestIdentity,
};

const HISTORY_EXECUTION_PREFIX: &[u8] = b"runmat-analysis-history-execution/v1\0";
const HISTORY_EXECUTION_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

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
        };
        request.validate()?;
        Ok(request)
    }
}

impl ObservationHistoryCheckpointRequest {
    pub fn new(
        plan: ResolvedObservationPlan,
        producer: super::StepIdentity,
        observation_id: ObservationId,
        predecessor: Option<ObservationHistoryCheckpointIdentity>,
        sample_ordinal: u64,
        sample_coordinate: RecoveredFieldSample,
        sample: ObservationResultIdentity,
    ) -> AnalysisExecutionResult<Self> {
        let identity = derive_request_identity(
            plan.identity,
            producer,
            &observation_id,
            predecessor,
            sample_ordinal,
            sample_coordinate,
            sample,
        )?;
        let request = Self {
            schema_version: 1,
            identity,
            plan,
            producer,
            observation_id,
            predecessor,
            sample_ordinal,
            sample_coordinate,
            sample,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.plan
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
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
        if self.schema_version != 1
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
                    self.sample_coordinate,
                    self.sample,
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
        self.validate()?;
        result
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if result.logical_identity != self.sample || result.sample != self.sample_coordinate {
            return Err(AnalysisExecutionError::Identity(
                "history checkpoint sample identity or coordinate differs from its request",
            ));
        }
        let expected_ordinal = previous
            .as_ref()
            .map_or(0, |checkpoint| checkpoint.manifest().observed_sample_count);
        if self.sample_ordinal != expected_ordinal {
            return Err(AnalysisExecutionError::Identity(
                "history checkpoint sample ordinal does not follow its predecessor",
            ));
        }
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
    sample_coordinate: RecoveredFieldSample,
    sample: ObservationResultIdentity,
) -> AnalysisExecutionResult<ObservationHistoryCheckpointRequestIdentity> {
    ObservationHistoryCheckpointRequestIdentity::derive(
        "history-checkpoint-request",
        &(
            plan,
            producer,
            observation_id,
            predecessor,
            sample_ordinal,
            sample_coordinate,
            sample,
        ),
    )
}
