use std::collections::BTreeSet;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    contracts::{SolverMethod, SolverPreconditioner},
    execution::ModalProjectionEvidence,
    frequency::FrequencyResponsePlanEntryIdentity,
    solve::{accelerator::AcceleratorSolveEvidence, backend::kind::LinearAlgebraBackendKind},
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, NumericalReceiptIdentity,
    SolverResultIdentity, StateIdentity, StructuralFrequencyPhysicalSample,
    StructuralFrequencyPhysicalSampleIdentity, StructuralFrequencyScope,
    StructuralFrequencySolverResult,
};

const FREQUENCY_RECEIPT_PREFIX: &[u8] = b"runmat-structural-frequency-receipt/v1\0";
const FREQUENCY_RECEIPT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Numerical coordinate space used to produce a method-neutral physical response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralFrequencySolutionSpace {
    Direct,
    ModalReduced {
        basis_state: StateIdentity,
        projection: ModalProjectionEvidence,
    },
}

impl StructuralFrequencySolutionSpace {
    pub(super) fn validate(&self, scope: &StructuralFrequencyScope) -> AnalysisExecutionResult<()> {
        match self {
            Self::Direct => Ok(()),
            Self::ModalReduced { projection, .. } => projection
                .validate_for(scope.operator_digest(), scope.physical_dof_count())
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string())),
        }
    }

    pub(crate) fn expected_method(&self) -> SolverMethod {
        match self {
            Self::Direct => SolverMethod::StructuralDirectHarmonicGmres,
            Self::ModalReduced { .. } => SolverMethod::StructuralModalHarmonicDenseDirect,
        }
    }

    pub(crate) fn expected_preconditioner(&self) -> SolverPreconditioner {
        match self {
            Self::Direct => SolverPreconditioner::Jacobi,
            Self::ModalReduced { .. } => SolverPreconditioner::None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralFrequencyEntrySolverEvidenceWire"
)]
pub struct StructuralFrequencyEntrySolverEvidence {
    entry: FrequencyResponsePlanEntryIdentity,
    sample: StructuralFrequencyPhysicalSampleIdentity,
    solver_method: SolverMethod,
    solver_backend: LinearAlgebraBackendKind,
    preconditioner: SolverPreconditioner,
    iterations: u32,
    solver_residual_norm: f64,
    solver_relative_residual: f64,
    physical_residual_norm: f64,
    physical_relative_balance_error: f64,
    accelerator_evidence: Vec<AcceleratorSolveEvidence>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyEntrySolverEvidenceWire {
    entry: FrequencyResponsePlanEntryIdentity,
    sample: StructuralFrequencyPhysicalSampleIdentity,
    solver_method: SolverMethod,
    solver_backend: LinearAlgebraBackendKind,
    preconditioner: SolverPreconditioner,
    iterations: u32,
    solver_residual_norm: f64,
    solver_relative_residual: f64,
    physical_residual_norm: f64,
    physical_relative_balance_error: f64,
    accelerator_evidence: Vec<AcceleratorSolveEvidence>,
}

impl TryFrom<StructuralFrequencyEntrySolverEvidenceWire>
    for StructuralFrequencyEntrySolverEvidence
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencyEntrySolverEvidenceWire) -> Result<Self, Self::Error> {
        let evidence = Self {
            entry: wire.entry,
            sample: wire.sample,
            solver_method: wire.solver_method,
            solver_backend: wire.solver_backend,
            preconditioner: wire.preconditioner,
            iterations: wire.iterations,
            solver_residual_norm: wire.solver_residual_norm,
            solver_relative_residual: wire.solver_relative_residual,
            physical_residual_norm: wire.physical_residual_norm,
            physical_relative_balance_error: wire.physical_relative_balance_error,
            accelerator_evidence: wire.accelerator_evidence,
        };
        evidence.validate_intrinsic()?;
        Ok(evidence)
    }
}

pub struct StructuralFrequencySolverEvidenceInput {
    pub solver_method: SolverMethod,
    pub solver_backend: LinearAlgebraBackendKind,
    pub preconditioner: SolverPreconditioner,
    pub iterations: u32,
    pub solver_residual_norm: f64,
    pub solver_relative_residual: f64,
    pub accelerator_evidence: Vec<AcceleratorSolveEvidence>,
}

impl StructuralFrequencyEntrySolverEvidence {
    pub fn new(
        sample: &StructuralFrequencyPhysicalSample,
        input: StructuralFrequencySolverEvidenceInput,
    ) -> AnalysisExecutionResult<Self> {
        let evidence = Self {
            entry: sample.entry().identity(),
            sample: sample.identity(),
            solver_method: input.solver_method,
            solver_backend: input.solver_backend,
            preconditioner: input.preconditioner,
            iterations: input.iterations,
            solver_residual_norm: input.solver_residual_norm,
            solver_relative_residual: input.solver_relative_residual,
            physical_residual_norm: sample.equilibrium_residual_norm()?,
            physical_relative_balance_error: sample.relative_balance_error()?,
            accelerator_evidence: input.accelerator_evidence,
        };
        evidence.validate_intrinsic()?;
        Ok(evidence)
    }

    pub const fn entry(&self) -> FrequencyResponsePlanEntryIdentity {
        self.entry
    }

    pub const fn sample(&self) -> StructuralFrequencyPhysicalSampleIdentity {
        self.sample
    }

    pub const fn solver_method(&self) -> SolverMethod {
        self.solver_method
    }

    pub const fn solver_backend(&self) -> LinearAlgebraBackendKind {
        self.solver_backend
    }

    pub const fn preconditioner(&self) -> SolverPreconditioner {
        self.preconditioner
    }

    pub const fn solver_relative_residual(&self) -> f64 {
        self.solver_relative_residual
    }

    pub const fn physical_relative_balance_error(&self) -> f64 {
        self.physical_relative_balance_error
    }

    pub(super) fn validate_for_sample(
        &self,
        sample: &StructuralFrequencyPhysicalSample,
        solution_space: &StructuralFrequencySolutionSpace,
    ) -> AnalysisExecutionResult<()> {
        self.validate_intrinsic()?;
        if self.entry != sample.entry().identity()
            || self.sample != sample.identity()
            || self.physical_residual_norm.to_bits()
                != sample.equilibrium_residual_norm()?.to_bits()
            || self.physical_relative_balance_error.to_bits()
                != sample.relative_balance_error()?.to_bits()
        {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency solver evidence differs from its physical sample or solution space",
            ));
        }
        self.validate_solution_semantics(solution_space)
    }

    pub(super) fn validate_for_manifest(
        &self,
        entry: FrequencyResponsePlanEntryIdentity,
        sample: StructuralFrequencyPhysicalSampleIdentity,
        solution_space: &StructuralFrequencySolutionSpace,
    ) -> AnalysisExecutionResult<()> {
        self.validate_intrinsic()?;
        if self.entry != entry || self.sample != sample {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency solver evidence differs from its entry manifest",
            ));
        }
        self.validate_solution_semantics(solution_space)
    }

    fn validate_intrinsic(&self) -> AnalysisExecutionResult<()> {
        if !self.solver_residual_norm.is_finite()
            || self.solver_residual_norm < 0.0
            || !self.solver_relative_residual.is_finite()
            || self.solver_relative_residual < 0.0
            || !self.physical_residual_norm.is_finite()
            || self.physical_residual_norm < 0.0
            || !self.physical_relative_balance_error.is_finite()
            || self.physical_relative_balance_error < 0.0
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency solver evidence contains an invalid residual".into(),
            ));
        }
        let expected_preconditioner = match self.solver_method {
            SolverMethod::StructuralDirectHarmonicGmres => SolverPreconditioner::Jacobi,
            SolverMethod::StructuralModalHarmonicDenseDirect => SolverPreconditioner::None,
            _ => {
                return Err(AnalysisExecutionError::Invalid(
                    "unexpected structural harmonic method".into(),
                ))
            }
        };
        if self.preconditioner != expected_preconditioner {
            return Err(AnalysisExecutionError::Invalid(
                "structural harmonic evidence does not name the applied preconditioner".into(),
            ));
        }
        match self.solver_backend {
            LinearAlgebraBackendKind::CpuReference if !self.accelerator_evidence.is_empty() => {
                Err(AnalysisExecutionError::Invalid(
                    "CPU structural frequency evidence contains accelerator records".into(),
                ))
            }
            LinearAlgebraBackendKind::Accelerator
                if self.accelerator_evidence.is_empty()
                    || self.accelerator_evidence.iter().any(|evidence| {
                        evidence.validate().is_err() || !evidence.satisfies_tolerance()
                    }) =>
            {
                Err(AnalysisExecutionError::Invalid(
                    "accelerated structural frequency evidence lacks valid completed provider records"
                        .into(),
                ))
            }
            _ => Ok(()),
        }
    }

    fn validate_solution_semantics(
        &self,
        solution_space: &StructuralFrequencySolutionSpace,
    ) -> AnalysisExecutionResult<()> {
        if self.solver_method != solution_space.expected_method() {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency solver method differs from its solution space".into(),
            ));
        }
        if matches!(
            solution_space,
            StructuralFrequencySolutionSpace::ModalReduced { .. }
        ) && (self.solver_backend != LinearAlgebraBackendKind::CpuReference
            || self.preconditioner != SolverPreconditioner::None)
        {
            return Err(AnalysisExecutionError::Invalid(
                "modal harmonic dense-direct evidence requires the CPU reference backend without a preconditioner"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Largest full-space relative balance error of a modal-reduced sweep and the sample (in plan
/// order) where it occurs. See [`StructuralFrequencyNumericalReceipt::modal_truncation_balance`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalTruncationBalance {
    pub maximum_relative_balance_error: f64,
    pub sample_index: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralFrequencyNumericalReceiptWire"
)]
pub struct StructuralFrequencyNumericalReceipt {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_result: SolverResultIdentity,
    scope: StructuralFrequencyScope,
    solution_space: StructuralFrequencySolutionSpace,
    samples: Vec<StructuralFrequencyEntrySolverEvidence>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyNumericalReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_result: SolverResultIdentity,
    scope: StructuralFrequencyScope,
    solution_space: StructuralFrequencySolutionSpace,
    samples: Vec<StructuralFrequencyEntrySolverEvidence>,
}

impl TryFrom<StructuralFrequencyNumericalReceiptWire> for StructuralFrequencyNumericalReceipt {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencyNumericalReceiptWire) -> Result<Self, Self::Error> {
        let receipt = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_result: wire.source_result,
            scope: wire.scope,
            solution_space: wire.solution_space,
            samples: wire.samples,
        };
        receipt.validate_identity()?;
        Ok(receipt)
    }
}

impl StructuralFrequencyNumericalReceipt {
    pub fn new(
        result: &StructuralFrequencySolverResult,
        solution_space: StructuralFrequencySolutionSpace,
        samples: Vec<StructuralFrequencyEntrySolverEvidence>,
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        solution_space.validate(result.manifest().scope())?;
        let mut receipt = Self {
            schema_version: 1,
            identity: NumericalReceiptIdentity::new(
                result.manifest().identity(),
                Digest::sha256([]),
            )?,
            source_result: result.manifest().identity(),
            scope: result.manifest().scope().clone(),
            solution_space,
            samples,
        };
        receipt.validate_against_samples(result)?;
        receipt.identity = receipt.derive_identity()?;
        receipt.validate_for_result(result)?;
        Ok(receipt)
    }

    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }

    pub const fn source_result(&self) -> SolverResultIdentity {
        self.source_result
    }

    pub fn scope(&self) -> &StructuralFrequencyScope {
        &self.scope
    }

    pub fn solution_space(&self) -> &StructuralFrequencySolutionSpace {
        &self.solution_space
    }

    pub fn samples(&self) -> &[StructuralFrequencyEntrySolverEvidence] {
        &self.samples
    }

    /// Full-space balance of a modal-reduced sweep, named for what it measures: the error of
    /// truncating the response to the reduced space (retained modes plus any residual vectors and
    /// static corrections). A modal-reduced solve is accepted on its reduced-space residual; this
    /// measure is published beside it. `None` for a direct solve, whose balance error is its
    /// acceptance criterion.
    pub fn modal_truncation_balance(&self) -> Option<ModalTruncationBalance> {
        if !matches!(
            self.solution_space,
            StructuralFrequencySolutionSpace::ModalReduced { .. }
        ) {
            return None;
        }
        self.samples
            .iter()
            .enumerate()
            .map(|(sample_index, sample)| ModalTruncationBalance {
                maximum_relative_balance_error: sample.physical_relative_balance_error,
                sample_index,
            })
            .max_by(|left, right| {
                left.maximum_relative_balance_error
                    .total_cmp(&right.maximum_relative_balance_error)
            })
    }

    pub fn validate_for_result(
        &self,
        result: &StructuralFrequencySolverResult,
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        result.validate()?;
        if self.source_result != result.manifest().identity()
            || self.scope != *result.manifest().scope()
        {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency numerical receipt names another physical result or scope",
            ));
        }
        self.validate_against_samples(result)
    }

    /// Validates receipt lineage without loading physical displacement and reaction vectors.
    ///
    /// Publication and query paths use the result manifest as their bounded semantic authority;
    /// worker-side solve admission continues to use [`Self::validate_for_result`].
    pub fn validate_for_manifest(
        &self,
        manifest: &super::StructuralFrequencySolverResultManifest,
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        manifest.validate()?;
        if self.source_result != manifest.identity() || self.scope != *manifest.scope() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency numerical receipt names another result manifest or scope",
            ));
        }
        if self.samples.len() != manifest.samples().len() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency numerical receipt does not cover every manifest sample",
            ));
        }
        for (evidence, entry) in self.samples.iter().zip(manifest.samples()) {
            if evidence.entry != entry.plan_entry().identity() || evidence.sample != entry.sample()
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural frequency numerical receipt sample order differs from its manifest",
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_identity()?;
        Ok(encode_contract(
            FREQUENCY_RECEIPT_PREFIX,
            "structural-frequency-numerical-receipt",
            self,
            FREQUENCY_RECEIPT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            FREQUENCY_RECEIPT_PREFIX,
            "structural-frequency-numerical-receipt",
            bytes,
            FREQUENCY_RECEIPT_LIMITS,
        )?)
    }

    fn validate_against_samples(
        &self,
        result: &StructuralFrequencySolverResult,
    ) -> AnalysisExecutionResult<()> {
        if self.samples.len() != result.samples().len() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency numerical receipt does not cover every result sample",
            ));
        }
        for (evidence, sample) in self.samples.iter().zip(result.samples()) {
            evidence.validate_for_sample(sample, &self.solution_space)?;
        }
        Ok(())
    }

    fn validate_identity(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.samples.is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency numerical receipt has an invalid schema or empty evidence"
                    .into(),
            ));
        }
        self.scope.validate()?;
        self.solution_space.validate(&self.scope)?;
        let mut entries = BTreeSet::new();
        let mut physical_samples = BTreeSet::new();
        for sample in &self.samples {
            sample.validate_intrinsic()?;
            sample.validate_solution_semantics(&self.solution_space)?;
            if !entries.insert(sample.entry) || !physical_samples.insert(sample.sample) {
                return Err(AnalysisExecutionError::Invalid(
                    "structural frequency numerical receipt contains duplicate sample evidence"
                        .into(),
                ));
            }
        }
        if self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency numerical receipt identity does not converge",
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
        let content = encode_contract(
            FREQUENCY_RECEIPT_PREFIX,
            "structural-frequency-numerical-receipt-content",
            &(
                self.schema_version,
                &self.scope,
                &self.solution_space,
                &self.samples,
            ),
            FREQUENCY_RECEIPT_LIMITS,
        )?;
        NumericalReceiptIdentity::new(self.source_result, Digest::sha256(content))
    }
}

#[cfg(test)]
mod tests;
