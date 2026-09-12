use runmat_analysis_core::ObservationId;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::observation::{
    BoundedObservationHistory, ObservationHistoryCheckpoint, ObservationHistoryCheckpointIdentity,
    ObservationResult, ObservationResultIdentity, ResolvedObservationPlan,
    ResolvedObservationPlanIdentity,
};

use super::{AnalysisExecutionError, AnalysisExecutionResult, HistoryCheckpointRequestIdentity};

const HISTORY_EXECUTION_PREFIX: &[u8] = b"runmat-analysis-history-execution/v1\0";
const HISTORY_EXECUTION_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "HistoryCheckpointRequestWire")]
pub struct HistoryCheckpointRequest {
    pub schema_version: u16,
    pub identity: HistoryCheckpointRequestIdentity,
    pub plan: ResolvedObservationPlan,
    pub observation_id: ObservationId,
    pub predecessor: Option<ObservationHistoryCheckpointIdentity>,
    pub sample: ObservationResultIdentity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryCheckpointRequestWire {
    schema_version: u16,
    identity: HistoryCheckpointRequestIdentity,
    plan: ResolvedObservationPlan,
    observation_id: ObservationId,
    predecessor: Option<ObservationHistoryCheckpointIdentity>,
    sample: ObservationResultIdentity,
}

impl TryFrom<HistoryCheckpointRequestWire> for HistoryCheckpointRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: HistoryCheckpointRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            plan: wire.plan,
            observation_id: wire.observation_id,
            predecessor: wire.predecessor,
            sample: wire.sample,
        };
        request.validate()?;
        Ok(request)
    }
}

impl HistoryCheckpointRequest {
    pub fn new(
        plan: ResolvedObservationPlan,
        observation_id: ObservationId,
        predecessor: Option<ObservationHistoryCheckpointIdentity>,
        sample: ObservationResultIdentity,
    ) -> AnalysisExecutionResult<Self> {
        let identity =
            derive_request_identity(plan.identity, &observation_id, predecessor, sample)?;
        let request = Self {
            schema_version: 1,
            identity,
            plan,
            observation_id,
            predecessor,
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
                    &self.observation_id,
                    self.predecessor,
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
        if result.logical_identity != self.sample {
            return Err(AnalysisExecutionError::Identity(
                "history checkpoint sample differs from its request",
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
    observation_id: &ObservationId,
    predecessor: Option<ObservationHistoryCheckpointIdentity>,
    sample: ObservationResultIdentity,
) -> AnalysisExecutionResult<HistoryCheckpointRequestIdentity> {
    HistoryCheckpointRequestIdentity::derive(
        "history-checkpoint-request",
        &(plan, observation_id, predecessor, sample),
    )
}
