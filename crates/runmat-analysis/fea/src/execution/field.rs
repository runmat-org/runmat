use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, FieldRecoveryRequestIdentity,
    SolverResultIdentity, StepIdentity,
};

const FIELD_RECOVERY_PREFIX: &[u8] = b"runmat-analysis-field-recovery/v1\0";
const FIELD_RECOVERY_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 65_536, 4096, 32);

/// Immutable request for family-owned recovery from one exact solver result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "FieldRecoveryRequestWire")]
pub struct FieldRecoveryRequest {
    pub schema_version: u16,
    identity: FieldRecoveryRequestIdentity,
    pub producer: StepIdentity,
    pub source_result: SolverResultIdentity,
    pub mesh_digest: Digest,
    pub projection_digest: Digest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldRecoveryRequestWire {
    schema_version: u16,
    identity: FieldRecoveryRequestIdentity,
    producer: StepIdentity,
    source_result: SolverResultIdentity,
    mesh_digest: Digest,
    projection_digest: Digest,
}

impl TryFrom<FieldRecoveryRequestWire> for FieldRecoveryRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: FieldRecoveryRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source_result: wire.source_result,
            mesh_digest: wire.mesh_digest,
            projection_digest: wire.projection_digest,
        };
        request.validate()?;
        Ok(request)
    }
}

impl FieldRecoveryRequest {
    pub fn new(
        producer: StepIdentity,
        source_result: SolverResultIdentity,
        mesh_digest: Digest,
        projection_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        let request = Self {
            schema_version: 1,
            identity: FieldRecoveryRequestIdentity::derive(
                "field-recovery-request",
                &(producer, source_result, mesh_digest, projection_digest),
            )?,
            producer,
            source_result,
            mesh_digest,
            projection_digest,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn identity(&self) -> FieldRecoveryRequestIdentity {
        self.identity
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let expected = FieldRecoveryRequestIdentity::derive(
            "field-recovery-request",
            &(
                self.producer,
                self.source_result,
                self.mesh_digest,
                self.projection_digest,
            ),
        )?;
        if self.schema_version != 1 || self.identity != expected {
            return Err(AnalysisExecutionError::Identity(
                "field recovery request identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            FIELD_RECOVERY_PREFIX,
            "field-recovery-request",
            self,
            FIELD_RECOVERY_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let request = decode_contract(
            FIELD_RECOVERY_PREFIX,
            "field-recovery-request",
            bytes,
            FIELD_RECOVERY_LIMITS,
        )?;
        Ok(request)
    }
}
