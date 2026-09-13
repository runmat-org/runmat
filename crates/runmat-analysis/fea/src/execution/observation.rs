use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;
use crate::results::field::{
    RecoveredFieldArtifactIdentity, RecoveredFieldSelection, RecoveredFieldSelectionIdentity,
};
use crate::results::observation::ResolvedObservationPlan;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, FieldRecoveryRequestIdentity,
    FieldRecoverySource, ObservationEvaluationRequestIdentity, StepIdentity,
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
    pub source_selection: RecoveredFieldSelectionIdentity,
    pub source: FieldRecoverySource,
    pub topology: CanonicalStructuralTopologyIdentity,
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
    source_selection: RecoveredFieldSelectionIdentity,
    source: FieldRecoverySource,
    topology: CanonicalStructuralTopologyIdentity,
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
            source_selection: wire.source_selection,
            source: wire.source,
            topology: wire.topology,
            plan: wire.plan,
        };
        request.validate()?;
        Ok(request)
    }
}

impl ObservationEvaluationRequest {
    pub fn new(
        producer: StepIdentity,
        source: &RecoveredFieldSelection,
        topology: CanonicalStructuralTopologyIdentity,
        plan: ResolvedObservationPlan,
    ) -> AnalysisExecutionResult<Self> {
        source
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        plan.validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if source.source.topology_digest != plan.topology_digest {
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
                    source.source.recovery_request,
                    source.source.identity,
                    source.identity,
                    source.source.source,
                    topology,
                    plan.identity,
                    plan.topology_digest,
                ),
            )?,
            producer,
            field_recovery_request: source.source.recovery_request,
            source_fields: source.source.identity,
            source_selection: source.identity,
            source: source.source.source,
            topology,
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
                self.source_selection,
                self.source,
                self.topology,
                self.plan.identity,
                self.plan.topology_digest,
            ),
        )?;
        if !self.topology.has_valid_identity()
            || self.plan.topology_digest != self.topology.discretization_digest()
            || self.identity != expected
        {
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
