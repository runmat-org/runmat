use runmat_analysis_core::{AnalysisProcedure, PhysicsFamily, SolutionSpace};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    admission::AnalysisRunIdentity,
    contracts::{ComputeBackend, SolverMethod, SolverPreconditioner},
    pipeline::structural_linearization::StructuralLinearizationIdentity,
    solve::backend::kind::LinearAlgebraBackendKind,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    NumericalReceiptIdentity, SolverResultIdentity, StateIdentity, StructuralBucklingWorkload,
    StructuralBucklingWorkloadIdentity,
};

const BUCKLING_RESULT_PREFIX: &[u8] = b"runmat-structural-buckling-result/v1\0";
const BUCKLING_RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralBucklingMode {
    pub load_factor: f64,
    pub residual_norm: f64,
    pub dof_shape: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralBucklingSolverResultWire")]
pub struct StructuralBucklingSolverResult {
    schema_version: u16,
    workload: StructuralBucklingWorkloadIdentity,
    topology_digest: Digest,
    source_state: StateIdentity,
    linearization: StructuralLinearizationIdentity,
    run: AnalysisRunIdentity,
    requested_backend: ComputeBackend,
    solver_backend: LinearAlgebraBackendKind,
    solver_method: SolverMethod,
    preconditioner: SolverPreconditioner,
    iteration_count: usize,
    modes: Vec<StructuralBucklingMode>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralBucklingSolverResultWire {
    schema_version: u16,
    workload: StructuralBucklingWorkloadIdentity,
    topology_digest: Digest,
    source_state: StateIdentity,
    linearization: StructuralLinearizationIdentity,
    run: AnalysisRunIdentity,
    requested_backend: ComputeBackend,
    solver_backend: LinearAlgebraBackendKind,
    solver_method: SolverMethod,
    preconditioner: SolverPreconditioner,
    iteration_count: usize,
    modes: Vec<StructuralBucklingMode>,
}

impl TryFrom<StructuralBucklingSolverResultWire> for StructuralBucklingSolverResult {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralBucklingSolverResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            workload: wire.workload,
            topology_digest: wire.topology_digest,
            source_state: wire.source_state,
            linearization: wire.linearization,
            run: wire.run,
            requested_backend: wire.requested_backend,
            solver_backend: wire.solver_backend,
            solver_method: wire.solver_method,
            preconditioner: wire.preconditioner,
            iteration_count: wire.iteration_count,
            modes: wire.modes,
        };
        result.validate()?;
        Ok(result)
    }
}

pub struct StructuralBucklingSolverResultInput {
    pub topology_digest: Digest,
    pub source_state: StateIdentity,
    pub linearization: StructuralLinearizationIdentity,
    pub run: AnalysisRunIdentity,
    pub solver_backend: LinearAlgebraBackendKind,
    pub solver_method: SolverMethod,
    pub preconditioner: SolverPreconditioner,
    pub iteration_count: usize,
    pub modes: Vec<StructuralBucklingMode>,
}

impl StructuralBucklingSolverResult {
    pub fn new(
        workload: &StructuralBucklingWorkload,
        input: StructuralBucklingSolverResultInput,
    ) -> AnalysisExecutionResult<Self> {
        let result = Self {
            schema_version: 1,
            workload: workload.identity(),
            topology_digest: input.topology_digest,
            source_state: input.source_state,
            linearization: input.linearization,
            run: input.run,
            requested_backend: workload.backend(),
            solver_backend: input.solver_backend,
            solver_method: input.solver_method,
            preconditioner: input.preconditioner,
            iteration_count: input.iteration_count,
            modes: input.modes,
        };
        result.validate_for(workload)?;
        Ok(result)
    }

    pub const fn workload(&self) -> StructuralBucklingWorkloadIdentity {
        self.workload
    }
    pub const fn topology_digest(&self) -> Digest {
        self.topology_digest
    }
    pub const fn source_state(&self) -> StateIdentity {
        self.source_state
    }
    pub const fn linearization(&self) -> StructuralLinearizationIdentity {
        self.linearization
    }
    pub fn run(&self) -> &AnalysisRunIdentity {
        &self.run
    }
    pub const fn requested_backend(&self) -> ComputeBackend {
        self.requested_backend
    }
    pub const fn solver_backend(&self) -> LinearAlgebraBackendKind {
        self.solver_backend
    }
    pub const fn solver_method(&self) -> SolverMethod {
        self.solver_method
    }
    pub const fn preconditioner(&self) -> SolverPreconditioner {
        self.preconditioner
    }
    pub const fn iteration_count(&self) -> usize {
        self.iteration_count
    }
    pub fn modes(&self) -> &[StructuralBucklingMode] {
        &self.modes
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.run.family() != PhysicsFamily::Structural
            || self.run.procedure() != AnalysisProcedure::EigenvalueBuckling
            || self.run.solution_space() != SolutionSpace::Direct
            || self.requested_backend != ComputeBackend::Cpu
            || self.solver_backend != LinearAlgebraBackendKind::CpuReference
            || self.iteration_count == 0
            || self.modes.is_empty()
            || self.modes.iter().any(|mode| {
                !mode.load_factor.is_finite()
                    || mode.load_factor <= 0.0
                    || !mode.residual_norm.is_finite()
                    || mode.residual_norm < 0.0
                    || mode.dof_shape.is_empty()
                    || mode.dof_shape.iter().any(|value| !value.is_finite())
            })
            || self
                .modes
                .windows(2)
                .any(|pair| pair[0].load_factor > pair[1].load_factor)
        {
            return Err(invalid(
                "structural buckling result is numerically inconsistent",
            ));
        }
        let shape_len = self.modes[0].dof_shape.len();
        if self
            .modes
            .iter()
            .any(|mode| mode.dof_shape.len() != shape_len)
        {
            return Err(invalid(
                "structural buckling mode shapes have inconsistent dimensions",
            ));
        }
        Ok(())
    }

    pub fn validate_for(
        &self,
        workload: &StructuralBucklingWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        workload.validate()?;
        if self.workload != workload.identity()
            || self.source_state != workload.prestress_transition().source_state
            || self.topology_digest != workload.prestress_transition().target_topology_digest
            || self.requested_backend != workload.backend()
            || self.modes.len() != workload.solver().requested_mode_count()
            || self
                .modes
                .iter()
                .any(|mode| mode.residual_norm > workload.solver().relative_tolerance())
        {
            return Err(AnalysisExecutionError::Identity(
                "structural buckling result differs from its workload or preload state",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            BUCKLING_RESULT_PREFIX,
            "structural-buckling-result",
            self,
            BUCKLING_RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            BUCKLING_RESULT_PREFIX,
            "structural-buckling-result",
            bytes,
            BUCKLING_RESULT_LIMITS,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralBucklingNumericalReceiptWire"
)]
pub struct StructuralBucklingNumericalReceipt {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    source_payload_digest: Digest,
    buckling_workload: StructuralBucklingWorkloadIdentity,
    source_state: StateIdentity,
    linearization: StructuralLinearizationIdentity,
    maximum_residual_norm: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralBucklingNumericalReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    source_payload_digest: Digest,
    buckling_workload: StructuralBucklingWorkloadIdentity,
    source_state: StateIdentity,
    linearization: StructuralLinearizationIdentity,
    maximum_residual_norm: f64,
}

impl TryFrom<StructuralBucklingNumericalReceiptWire> for StructuralBucklingNumericalReceipt {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralBucklingNumericalReceiptWire) -> Result<Self, Self::Error> {
        let receipt = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_workload: wire.source_workload,
            source_result: wire.source_result,
            source_payload_digest: wire.source_payload_digest,
            buckling_workload: wire.buckling_workload,
            source_state: wire.source_state,
            linearization: wire.linearization,
            maximum_residual_norm: wire.maximum_residual_norm,
        };
        receipt.validate()?;
        Ok(receipt)
    }
}

impl StructuralBucklingNumericalReceipt {
    pub fn new(
        source_workload: AnalysisWorkloadIdentity,
        result: &StructuralBucklingSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        let source_payload_digest = Digest::sha256(result.canonical_bytes()?);
        let source_result = SolverResultIdentity::new(source_workload, source_payload_digest)?;
        let maximum_residual_norm = result
            .modes
            .iter()
            .map(|mode| mode.residual_norm)
            .fold(0.0_f64, f64::max);
        let mut receipt = Self {
            schema_version: 1,
            identity: NumericalReceiptIdentity::new(source_result, Digest::sha256([]))?,
            source_workload,
            source_result,
            source_payload_digest,
            buckling_workload: result.workload,
            source_state: result.source_state,
            linearization: result.linearization,
            maximum_residual_norm,
        };
        receipt.identity = receipt.derive_identity()?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }
    pub const fn source_result(&self) -> SolverResultIdentity {
        self.source_result
    }
    pub const fn maximum_residual_norm(&self) -> f64 {
        self.maximum_residual_norm
    }

    pub fn validate_against(
        &self,
        source_result: SolverResultIdentity,
        result: &StructuralBucklingSolverResult,
    ) -> AnalysisExecutionResult<()> {
        let expected = Self::new(self.source_workload, result)?;
        if self != &expected || self.source_result != source_result {
            return Err(AnalysisExecutionError::Identity(
                "structural buckling receipt differs from its source result",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            BUCKLING_RESULT_PREFIX,
            "structural-buckling-receipt",
            self,
            BUCKLING_RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let receipt: Self = decode_contract(
            BUCKLING_RESULT_PREFIX,
            "structural-buckling-receipt",
            bytes,
            BUCKLING_RESULT_LIMITS,
        )?;
        receipt.validate()?;
        Ok(receipt)
    }

    fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || !self.maximum_residual_norm.is_finite()
            || self.maximum_residual_norm < 0.0
            || self.source_result
                != SolverResultIdentity::new(self.source_workload, self.source_payload_digest)?
            || self.identity != self.derive_identity()?
        {
            return Err(invalid("structural buckling receipt is invalid"));
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
        let evidence_digest = Digest::sha256(encode_contract(
            BUCKLING_RESULT_PREFIX,
            "structural-buckling-receipt-evidence",
            &(
                self.buckling_workload,
                self.source_state,
                self.linearization,
                self.maximum_residual_norm,
            ),
            BUCKLING_RESULT_LIMITS,
        )?);
        NumericalReceiptIdentity::new(self.source_result, evidence_digest)
    }
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}
