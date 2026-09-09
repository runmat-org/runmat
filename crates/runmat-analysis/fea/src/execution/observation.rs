use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::results::field::{RecoveredFieldArtifact, RecoveredFieldArtifactIdentity};
use crate::results::observation::ResolvedObservationPlan;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, FieldRecoveryRequestIdentity,
    ObservationEvaluationRequestIdentity, SolverResultIdentity, StepIdentity,
};

const OBSERVATION_EVALUATION_PREFIX: &[u8] = b"runmat-analysis-observation-evaluation/v1\0";
const OBSERVATION_EVALUATION_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationEvaluationRequestWire")]
pub struct ObservationEvaluationRequest {
    pub schema_version: u16,
    identity: ObservationEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub field_recovery_request: FieldRecoveryRequestIdentity,
    pub source_fields: RecoveredFieldArtifactIdentity,
    pub source_result: SolverResultIdentity,
    pub mesh_digest: Digest,
    pub plan: ResolvedObservationPlan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationEvaluationRequestWire {
    schema_version: u16,
    identity: ObservationEvaluationRequestIdentity,
    producer: StepIdentity,
    field_recovery_request: FieldRecoveryRequestIdentity,
    source_fields: RecoveredFieldArtifactIdentity,
    source_result: SolverResultIdentity,
    mesh_digest: Digest,
    plan: ResolvedObservationPlan,
}

impl TryFrom<ObservationEvaluationRequestWire> for ObservationEvaluationRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ObservationEvaluationRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            field_recovery_request: wire.field_recovery_request,
            source_fields: wire.source_fields,
            source_result: wire.source_result,
            mesh_digest: wire.mesh_digest,
            plan: wire.plan,
        };
        request.validate()?;
        Ok(request)
    }
}

impl ObservationEvaluationRequest {
    pub fn new(
        producer: StepIdentity,
        source: &RecoveredFieldArtifact,
        mesh_digest: Digest,
        plan: ResolvedObservationPlan,
    ) -> AnalysisExecutionResult<Self> {
        source
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        plan.validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if source.topology_digest != plan.topology_digest {
            return Err(AnalysisExecutionError::Identity(
                "observation plan topology differs from its recovered fields",
            ));
        }
        let request = Self {
            schema_version: 1,
            identity: ObservationEvaluationRequestIdentity::derive(
                "observation-evaluation-request",
                &(
                    producer,
                    source.recovery_request,
                    source.identity(),
                    source.source_result,
                    mesh_digest,
                    plan.identity,
                    plan.topology_digest,
                ),
            )?,
            producer,
            field_recovery_request: source.recovery_request,
            source_fields: source.identity(),
            source_result: source.source_result,
            mesh_digest,
            plan,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn identity(&self) -> ObservationEvaluationRequestIdentity {
        self.identity
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.plan
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.schema_version != 1 || self.plan.observations.is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "observation evaluation requires a supported schema and nonempty plan".into(),
            ));
        }
        let expected = ObservationEvaluationRequestIdentity::derive(
            "observation-evaluation-request",
            &(
                self.producer,
                self.field_recovery_request,
                self.source_fields,
                self.source_result,
                self.mesh_digest,
                self.plan.identity,
                self.plan.topology_digest,
            ),
        )?;
        if self.identity != expected {
            return Err(AnalysisExecutionError::Identity(
                "observation evaluation request identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_EVALUATION_PREFIX,
            "observation-evaluation-request",
            self,
            OBSERVATION_EVALUATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let request = decode_contract(
            OBSERVATION_EVALUATION_PREFIX,
            "observation-evaluation-request",
            bytes,
            OBSERVATION_EVALUATION_LIMITS,
        )?;
        Ok(request)
    }
}
