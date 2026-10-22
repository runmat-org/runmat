//! Original physical acceptance receipt, not an independent discretization certificate.
use super::*;
use crate::execution::{NumericalReceiptIdentity, SolverResultIdentity};
use runmat_meshing_core::ElementOrder;

const PREFIX: &[u8] = b"runmat-acoustic-response-receipt/v1\0";
pub const ACOUSTIC_RESPONSE_RECEIPT_MAX_ENCODED_BYTES: u64 = 64 * 1024;
const RECEIPT_LIMITS: CanonicalLimits = CanonicalLimits::new(
    ACOUSTIC_RESPONSE_RECEIPT_MAX_ENCODED_BYTES as usize,
    4096,
    4096,
    32,
);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticResponseNumericalReceipt {
    pub producer: StepIdentity,
    pub procedure: AnalysisProcedure,
    pub topology_digest: Digest,
    pub element_order: ElementOrder,
    pub nodes: usize,
    pub samples: usize,
    /// Largest accepted physical residual: eigenpair residual for cavity modes, effective
    /// operator residual over every step for a transient.
    pub maximum_relative_residual: f64,
    pub solver_relative_tolerance: f64,
    pub linear_iterations: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ReceiptWire")]
pub struct AcousticResponseNumericalReceiptArtifact {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: AcousticResponseNumericalReceipt,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    receipt: AcousticResponseNumericalReceipt,
}

impl TryFrom<ReceiptWire> for AcousticResponseNumericalReceiptArtifact {
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

impl AcousticResponseNumericalReceiptArtifact {
    pub fn new(
        workload: &AcousticResponseWorkload,
        result: &AcousticResponseSolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_for(workload, mesh)?;
        let (maximum_relative_residual, linear_iterations) = match result.diagnostics() {
            AcousticResponseDiagnostics::Modal {
                relative_residuals,
                linear_iterations,
                ..
            } => (
                relative_residuals.iter().copied().fold(0.0, f64::max),
                *linear_iterations,
            ),
            AcousticResponseDiagnostics::Transient {
                maximum_linear_relative_residual,
                linear_iterations,
                ..
            } => (*maximum_linear_relative_residual, *linear_iterations),
        };
        let receipt = AcousticResponseNumericalReceipt {
            producer: result.producer(),
            procedure: workload.solver.procedure(),
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            element_order: mesh.resolved_request.element_order,
            nodes: mesh.topology.nodes.len(),
            samples: result.samples().len(),
            maximum_relative_residual,
            solver_relative_tolerance: workload.solver.relative_tolerance(),
            linear_iterations,
        };
        let source_workload = workload.identity()?;
        let source_result = result.source_result_identity()?;
        let value = Self {
            schema_version: 1,
            identity: receipt_identity(source_workload, source_result, &receipt)?,
            source_workload,
            source_result,
            receipt,
        };
        value.validate()?;
        Ok(value)
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
    pub const fn receipt(&self) -> AcousticResponseNumericalReceipt {
        self.receipt
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let r = self.receipt;
        let t = r.solver_relative_tolerance;
        if self.schema_version != 1
            || r.nodes == 0
            || r.samples == 0
            || !matches!(
                r.procedure,
                AnalysisProcedure::ModalExtraction | AnalysisProcedure::TransientResponse
            )
            || !(t.is_finite() && t > 0.0 && t < 1.0)
            || !(r.maximum_relative_residual.is_finite()
                && r.maximum_relative_residual >= 0.0
                && r.maximum_relative_residual <= t)
            || self.identity != receipt_identity(self.source_workload, self.source_result, &r)?
        {
            return Err(invalid(
                "acoustic response receipt authority, cardinality or residual differs",
            ));
        }
        Ok(())
    }
    pub fn validate_against(
        &self,
        workload: &AcousticResponseWorkload,
        result: &AcousticResponseSolverResult,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if self != &Self::new(workload, result, mesh)? {
            return Err(invalid(
                "acoustic response receipt differs from original workload/result/mesh",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "acoustic-response-receipt",
            self,
            RECEIPT_LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self =
            decode_contract(PREFIX, "acoustic-response-receipt", bytes, RECEIPT_LIMITS)?;
        value.validate()?;
        Ok(value)
    }
}

fn receipt_identity(
    workload: AnalysisWorkloadIdentity,
    result: SolverResultIdentity,
    receipt: &AcousticResponseNumericalReceipt,
) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
    NumericalReceiptIdentity::new(
        result,
        Digest::sha256(encode_contract(
            PREFIX,
            "acoustic-response-receipt-content",
            &(workload, receipt),
            RECEIPT_LIMITS,
        )?),
    )
}
