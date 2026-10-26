//! Accepted algebraic closure bound to the exact canonical acoustic result.
//! This is not spatial convergence, independent physics evidence, or field accuracy.
use super::*;
use crate::execution::{NumericalReceiptIdentity, SolverResultIdentity, StepIdentity};

const PREFIX: &[u8] = b"runmat-acoustic-numerical-receipt/v2\0";
pub const ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES: u64 = 64 * 1024;
const LIMITS: CanonicalLimits =
    CanonicalLimits::new(ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES as usize, 4096, 4096, 32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticNumericalReceipt {
    pub producer: StepIdentity,
    pub topology_digest: Digest,
    pub frequency_hz: f64,
    pub relative_residual: f64,
    pub power_relative_error: f64,
    pub solver_relative_tolerance: f64,
    pub power_balance_relative_tolerance: f64,
    /// Factorization nonsingularity certificate of the reduced operator.
    pub certificate: crate::physics::acoustic::AcousticSolveCertificate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ReceiptWire")]
pub struct AcousticNumericalReceiptArtifact {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: AcousticNumericalReceipt,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: AcousticNumericalReceipt,
}
impl TryFrom<ReceiptWire> for AcousticNumericalReceiptArtifact {
    type Error = AnalysisExecutionError;
    fn try_from(w: ReceiptWire) -> AnalysisExecutionResult<Self> {
        let value = Self {
            schema_version: w.schema_version,
            identity: w.identity,
            source_workload: w.source_workload,
            source_result: w.source_result,
            receipt: w.receipt,
        };
        value.validate()?;
        Ok(value)
    }
}
impl AcousticNumericalReceiptArtifact {
    pub fn new(
        workload: &AcousticHarmonicWorkload,
        result: &AcousticHarmonicSolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_for(workload, mesh)?;
        let receipt = AcousticNumericalReceipt {
            producer: result.producer(),
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            frequency_hz: workload.frequency.canonical_value(),
            relative_residual: result.relative_residual(),
            power_relative_error: result.power_relative_error(),
            solver_relative_tolerance: workload.solver.relative_tolerance,
            power_balance_relative_tolerance: workload.power_balance_relative_tolerance,
            certificate: result.certificate(),
        };
        let source_workload = workload.identity()?;
        let source_result = result.source_result_identity()?;
        let artifact = Self {
            schema_version: 2,
            identity: receipt_identity(source_workload, source_result, &receipt)?,
            source_workload,
            source_result,
            receipt,
        };
        artifact.validate()?;
        Ok(artifact)
    }
    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }
    pub const fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.source_workload
    }
    pub const fn source_result(&self) -> SolverResultIdentity {
        self.source_result
    }
    pub const fn receipt(&self) -> AcousticNumericalReceipt {
        self.receipt
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let r = self.receipt;
        r.certificate
            .validate(r.certificate.limit, usize::MAX)
            .map_err(invalid)?;
        if self.schema_version != 2
            || !r.frequency_hz.is_finite()
            || r.frequency_hz <= 0.0
            || !r.solver_relative_tolerance.is_finite()
            || r.solver_relative_tolerance <= 0.0
            || r.solver_relative_tolerance >= 1.0
            || !r.power_balance_relative_tolerance.is_finite()
            || r.power_balance_relative_tolerance <= 0.0
            || r.power_balance_relative_tolerance >= 1.0
            || !r.relative_residual.is_finite()
            || r.relative_residual < 0.0
            || r.relative_residual > r.solver_relative_tolerance
            || !r.power_relative_error.is_finite()
            || r.power_relative_error < 0.0
            || r.power_relative_error > r.power_balance_relative_tolerance
            || self.identity != receipt_identity(self.source_workload, self.source_result, &r)?
        {
            return Err(invalid(
                "invalid acoustic numerical receipt evidence or identity",
            ));
        }
        Ok(())
    }
    pub fn validate_against(
        &self,
        workload: &AcousticHarmonicWorkload,
        result: &AcousticHarmonicSolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if *self != Self::new(workload, result, mesh)? {
            return Err(invalid(
                "acoustic receipt differs from exact workload, result or mesh",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "acoustic-numerical-receipt",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            PREFIX,
            "acoustic-numerical-receipt",
            bytes,
            LIMITS,
        )?)
    }
}
fn receipt_identity(
    workload: AnalysisWorkloadIdentity,
    result: SolverResultIdentity,
    receipt: &AcousticNumericalReceipt,
) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
    let bytes = encode_contract(
        PREFIX,
        "acoustic-numerical-receipt-content",
        &(workload, receipt),
        LIMITS,
    )?;
    NumericalReceiptIdentity::new(result, Digest::sha256(bytes))
}

#[cfg(test)]
mod tests;
