use runmat_analysis_core::{AnalysisField, AnalysisProcedure, PhysicsFamily, SolutionSpace};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{contracts::FeaRunResult, ModalBasisArtifact, ModalBasisIdentity};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    NumericalReceiptIdentity, SolverResultIdentity, StructuralModalWorkload,
    StructuralModalWorkloadIdentity,
};

const MODAL_RESULT_PREFIX: &[u8] = b"runmat-structural-modal-result/v1\0";
const MODAL_RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Portable result of one modal-extraction workload.
///
/// The reusable basis is authoritative. Result-facing mode fields can be projected or recovered
/// from that basis and the separately transported mesh; storing every recovered mode here would
/// multiply the dominant payload without adding numerical authority.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralModalSolverResultWire")]
pub struct StructuralModalSolverResult {
    schema_version: u16,
    workload: StructuralModalWorkloadIdentity,
    topology_digest: Digest,
    run: FeaRunResult,
    basis: ModalBasisArtifact,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalSolverResultWire {
    schema_version: u16,
    workload: StructuralModalWorkloadIdentity,
    topology_digest: Digest,
    run: FeaRunResult,
    basis: ModalBasisArtifact,
}

impl TryFrom<StructuralModalSolverResultWire> for StructuralModalSolverResult {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralModalSolverResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            workload: wire.workload,
            topology_digest: wire.topology_digest,
            run: wire.run,
            basis: wire.basis,
        };
        result.validate()?;
        Ok(result)
    }
}

impl StructuralModalSolverResult {
    pub fn new(
        workload: &StructuralModalWorkload,
        topology_digest: Digest,
        run: FeaRunResult,
        basis: ModalBasisArtifact,
    ) -> AnalysisExecutionResult<Self> {
        workload.validate()?;
        let result = Self {
            schema_version: 1,
            workload: workload.identity(),
            topology_digest,
            run,
            basis,
        };
        result.validate_for(workload)?;
        Ok(result)
    }

    pub const fn workload(&self) -> StructuralModalWorkloadIdentity {
        self.workload
    }

    pub const fn topology_digest(&self) -> Digest {
        self.topology_digest
    }

    pub const fn basis_identity(&self) -> ModalBasisIdentity {
        self.basis.identity()
    }

    pub fn basis(&self) -> &crate::ModalBasisArtifact {
        &self.basis
    }

    pub fn run(&self) -> &FeaRunResult {
        &self.run
    }

    pub fn into_parts(self) -> (FeaRunResult, ModalBasisArtifact) {
        (self.run, self.basis)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let mode_count = self.basis.modes().len();
        if self.schema_version != 1
            || mode_count == 0
            || self.basis.source_run() != &self.run.identity
            || self.run.identity.family() != PhysicsFamily::Structural
            || !matches!(
                self.run.identity.procedure(),
                AnalysisProcedure::ModalExtraction | AnalysisProcedure::PrestressedModalExtraction
            )
            || self.run.identity.solution_space() != SolutionSpace::Direct
            || self.run.fields_are_empty()
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural modal result metadata or mode inventory is inconsistent".into(),
            ));
        }
        self.basis
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        validate_fields(&self.run.fields)?;
        Ok(())
    }

    pub fn validate_for(&self, workload: &StructuralModalWorkload) -> AnalysisExecutionResult<()> {
        self.validate()?;
        workload.validate()?;
        let step = workload.procedure_step()?;
        let expected_residual_directions =
            crate::pipeline::modal_residual::canonical_residual_vector_directions(
                workload.residual_vector_directions(),
            )
            .map_err(AnalysisExecutionError::Invalid)?;
        let residual_directions_match = expected_residual_directions.len()
            == self.basis.residual_vectors().len()
            && expected_residual_directions
                .iter()
                .zip(self.basis.residual_vectors())
                .all(|(expected, actual)| {
                    expected
                        .iter()
                        .zip(actual.direction())
                        .all(|(left, right)| {
                            (left - right).abs()
                                <= 128.0 * f64::EPSILON * left.abs().max(right.abs()).max(1.0)
                        })
                });
        if self.workload != workload.identity()
            || self.run.requested_backend != workload.backend()
            || self.run.identity.family() != step.family
            || self.run.identity.procedure() != step.procedure
            || self.run.identity.solution_space() != step.solution_space
            || self.basis.truncation().policy() != workload.truncation()
            || !residual_directions_match
        {
            return Err(AnalysisExecutionError::Identity(
                "structural modal result differs from its source workload",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            MODAL_RESULT_PREFIX,
            "structural-modal-result",
            self,
            MODAL_RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            MODAL_RESULT_PREFIX,
            "structural-modal-result",
            bytes,
            MODAL_RESULT_LIMITS,
        )?)
    }
}

/// Compact source-bound evidence for modal state publication and accuracy admission.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralModalNumericalReceiptWire")]
pub struct StructuralModalNumericalReceipt {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_payload_digest: Digest,
    source_result: SolverResultIdentity,
    modal_workload: StructuralModalWorkloadIdentity,
    topology_digest: Digest,
    basis: ModalBasisIdentity,
    operator_digest: Digest,
    retained_mode_count: usize,
    available_mode_count: usize,
    boundary_probe_count: usize,
    maximum_residual_norm: f64,
    residual_vector_count: usize,
    maximum_residual_vector_static_equilibrium_ratio: f64,
    maximum_residual_vector_mass_orthogonality_ratio: f64,
    complete_boundary_eigenspace: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalNumericalReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_payload_digest: Digest,
    source_result: SolverResultIdentity,
    modal_workload: StructuralModalWorkloadIdentity,
    topology_digest: Digest,
    basis: ModalBasisIdentity,
    operator_digest: Digest,
    retained_mode_count: usize,
    available_mode_count: usize,
    boundary_probe_count: usize,
    maximum_residual_norm: f64,
    residual_vector_count: usize,
    maximum_residual_vector_static_equilibrium_ratio: f64,
    maximum_residual_vector_mass_orthogonality_ratio: f64,
    complete_boundary_eigenspace: bool,
}

impl TryFrom<StructuralModalNumericalReceiptWire> for StructuralModalNumericalReceipt {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralModalNumericalReceiptWire) -> Result<Self, Self::Error> {
        let receipt = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_workload: wire.source_workload,
            source_payload_digest: wire.source_payload_digest,
            source_result: wire.source_result,
            modal_workload: wire.modal_workload,
            topology_digest: wire.topology_digest,
            basis: wire.basis,
            operator_digest: wire.operator_digest,
            retained_mode_count: wire.retained_mode_count,
            available_mode_count: wire.available_mode_count,
            boundary_probe_count: wire.boundary_probe_count,
            maximum_residual_norm: wire.maximum_residual_norm,
            residual_vector_count: wire.residual_vector_count,
            maximum_residual_vector_static_equilibrium_ratio: wire
                .maximum_residual_vector_static_equilibrium_ratio,
            maximum_residual_vector_mass_orthogonality_ratio: wire
                .maximum_residual_vector_mass_orthogonality_ratio,
            complete_boundary_eigenspace: wire.complete_boundary_eigenspace,
        };
        receipt.validate()?;
        Ok(receipt)
    }
}

impl StructuralModalNumericalReceipt {
    pub fn new(
        source_workload: AnalysisWorkloadIdentity,
        result: &StructuralModalSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        let source_payload_digest = Digest::sha256(result.canonical_bytes()?);
        let source_result = SolverResultIdentity::new(source_workload, source_payload_digest)?;
        let basis = result.basis();
        let mut receipt = Self {
            schema_version: 1,
            identity: NumericalReceiptIdentity::new(source_result, Digest::sha256([]))?,
            source_workload,
            source_payload_digest,
            source_result,
            modal_workload: result.workload(),
            topology_digest: result.topology_digest(),
            basis: basis.identity(),
            operator_digest: basis.operator_digest(),
            retained_mode_count: basis.truncation().retained_mode_count(),
            available_mode_count: basis.truncation().available_mode_count(),
            boundary_probe_count: basis.truncation().boundary_probe_count(),
            maximum_residual_norm: basis
                .modes()
                .iter()
                .map(|mode| mode.residual_norm())
                .fold(0.0_f64, f64::max),
            residual_vector_count: basis.residual_vectors().len(),
            maximum_residual_vector_static_equilibrium_ratio: basis
                .residual_vectors()
                .iter()
                .map(|residual| residual.static_equilibrium_residual_ratio())
                .fold(0.0_f64, f64::max),
            maximum_residual_vector_mass_orthogonality_ratio: basis
                .residual_vectors()
                .iter()
                .map(|residual| residual.maximum_mass_orthogonality_ratio())
                .fold(0.0_f64, f64::max),
            complete_boundary_eigenspace: basis.truncation().complete_boundary_eigenspace(),
        };
        receipt.identity = NumericalReceiptIdentity::new(source_result, receipt.content_digest()?)?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }

    pub const fn source_result(&self) -> SolverResultIdentity {
        self.source_result
    }

    pub const fn basis(&self) -> ModalBasisIdentity {
        self.basis
    }

    pub const fn residual_vector_count(&self) -> usize {
        self.residual_vector_count
    }

    pub const fn maximum_residual_vector_static_equilibrium_ratio(&self) -> f64 {
        self.maximum_residual_vector_static_equilibrium_ratio
    }

    pub const fn maximum_residual_vector_mass_orthogonality_ratio(&self) -> f64 {
        self.maximum_residual_vector_mass_orthogonality_ratio
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let expected_source =
            SolverResultIdentity::new(self.source_workload, self.source_payload_digest)?;
        let expected_identity =
            NumericalReceiptIdentity::new(self.source_result, self.content_digest()?)?;
        if self.schema_version != 1
            || self.source_result != expected_source
            || self.identity != expected_identity
            || self.retained_mode_count == 0
            || self.available_mode_count < self.retained_mode_count
            || self.residual_vector_count > 3
            || !self.maximum_residual_norm.is_finite()
            || self.maximum_residual_norm < 0.0
            || !self
                .maximum_residual_vector_static_equilibrium_ratio
                .is_finite()
            || !(0.0..=1.0e-8).contains(&self.maximum_residual_vector_static_equilibrium_ratio)
            || !self
                .maximum_residual_vector_mass_orthogonality_ratio
                .is_finite()
            || !(0.0..=1.0e-8).contains(&self.maximum_residual_vector_mass_orthogonality_ratio)
            || (self.residual_vector_count == 0
                && (self.maximum_residual_vector_static_equilibrium_ratio != 0.0
                    || self.maximum_residual_vector_mass_orthogonality_ratio != 0.0))
        {
            return Err(AnalysisExecutionError::Identity(
                "structural modal numerical receipt identity or evidence does not converge",
            ));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        source_result: SolverResultIdentity,
        result: &StructuralModalSolverResult,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let expected = Self::new(self.source_workload, result)?;
        if self.source_result != source_result || self != &expected {
            return Err(AnalysisExecutionError::Identity(
                "structural modal numerical receipt differs from its solver result",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            MODAL_RESULT_PREFIX,
            "structural-modal-numerical-receipt",
            self,
            MODAL_RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            MODAL_RESULT_PREFIX,
            "structural-modal-numerical-receipt",
            bytes,
            MODAL_RESULT_LIMITS,
        )?)
    }

    fn content_digest(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(encode_contract(
            MODAL_RESULT_PREFIX,
            "structural-modal-numerical-receipt-content",
            &(
                self.modal_workload,
                self.topology_digest,
                self.basis,
                self.operator_digest,
                self.retained_mode_count,
                self.available_mode_count,
                self.boundary_probe_count,
                self.maximum_residual_norm,
                self.residual_vector_count,
                self.maximum_residual_vector_static_equilibrium_ratio,
                self.maximum_residual_vector_mass_orthogonality_ratio,
                self.complete_boundary_eigenspace,
            ),
            MODAL_RESULT_LIMITS,
        )?))
    }
}

fn validate_fields(fields: &[AnalysisField]) -> AnalysisExecutionResult<()> {
    let mut identities = std::collections::BTreeSet::new();
    for field in fields {
        let values = field.as_host_f64().ok_or_else(|| {
            AnalysisExecutionError::Invalid(
                "portable modal results cannot contain device-local field references".into(),
            )
        })?;
        let expected = field
            .shape
            .iter()
            .try_fold(1_usize, |count, extent| count.checked_mul(*extent));
        if field.field_id.trim().is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "portable modal field has an empty identity".into(),
            ));
        }
        if !identities.insert(field.field_id.as_str()) {
            return Err(AnalysisExecutionError::Invalid(format!(
                "portable modal result repeats field `{}`",
                field.field_id
            )));
        }
        if expected != Some(values.len()) || values.is_empty() {
            return Err(AnalysisExecutionError::Invalid(format!(
                "portable modal field `{}` shape {:?} does not match its {} values",
                field.field_id,
                field.shape,
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(AnalysisExecutionError::Invalid(format!(
                "portable modal field `{}` contains a non-finite value",
                field.field_id
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
