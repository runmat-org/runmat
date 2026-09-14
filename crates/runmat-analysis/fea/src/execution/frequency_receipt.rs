use std::collections::BTreeSet;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    contracts::{SolverMethod, SolverPreconditioner},
    frequency::FrequencyResponsePlanEntryIdentity,
    solve::{
        accelerator::AcceleratorSolveEvidence, backend::kind::LinearAlgebraBackendKind,
        dynamics::ModalProjectionEvidence,
    },
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, NumericalReceiptIdentity,
    SolverResultIdentity, StateIdentity, StructuralFrequencyPhysicalSample,
    StructuralFrequencyPhysicalSampleIdentity, StructuralFrequencyScope,
    StructuralFrequencySolverResult,
};

const FREQUENCY_RECEIPT_PREFIX: &[u8] = b"runmat-structural-frequency-receipt/v1\0";
const FREQUENCY_RECEIPT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

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

    fn expected_method(&self) -> SolverMethod {
        match self {
            Self::Direct => SolverMethod::StructuralDirectHarmonicGmres,
            Self::ModalReduced { .. } => SolverMethod::StructuralModalHarmonicDenseDirect,
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
        if self.preconditioner != SolverPreconditioner::None {
            return Err(AnalysisExecutionError::Invalid(
                "current structural harmonic solvers do not admit a preconditioner".into(),
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
mod tests {
    use super::*;
    use crate::{
        execution::{
            ComplexStructuralBalance, ConstraintPlanIdentity, DiscretizationIdentity,
            StructuralComplexVector, StructuralDampingOperatorIdentity,
            StructuralDofLayoutIdentity, StructuralExcitationSetIdentity,
            StructuralFormulationIdentity, StructuralFrequencyProblemIdentity,
            StructuralFrequencyScope, StructuralFrequencyScopeInput, StructuralKinematicsIdentity,
            StructuralMassOperatorIdentity, StructuralPhasorConvention,
            StructuralStiffnessOperatorIdentity,
        },
        frequency::FrequencyResponsePlan,
        pipeline::modal_basis::ModalBasisIdentity,
    };

    fn operator_digest() -> Digest {
        Digest::sha256(b"test-structural-operator")
    }

    fn basis_identity() -> ModalBasisIdentity {
        serde_json::from_value(serde_json::to_value(Digest::sha256(b"test-modal-basis")).unwrap())
            .unwrap()
    }

    fn scope() -> StructuralFrequencyScope {
        let plan = FrequencyResponsePlan::new(20.0, &[10.0], &[]).unwrap();
        StructuralFrequencyScope::new(StructuralFrequencyScopeInput {
            problem: StructuralFrequencyProblemIdentity::from_physical_digest(Digest::sha256(
                b"test-frequency-problem",
            ))
            .unwrap(),
            plan,
            discretization: DiscretizationIdentity::derive("test-discretization", &1_u8).unwrap(),
            constraint_plan: ConstraintPlanIdentity::derive("test-constraints", &1_u8).unwrap(),
            stiffness_operator: StructuralStiffnessOperatorIdentity::derive(
                "test-stiffness",
                &1_u8,
            )
            .unwrap(),
            mass_operator: StructuralMassOperatorIdentity::derive("test-mass", &1_u8).unwrap(),
            damping_operator: StructuralDampingOperatorIdentity::derive("test-damping", &1_u8)
                .unwrap(),
            operator_digest: operator_digest(),
            excitations: StructuralExcitationSetIdentity::derive("test-excitations", &1_u8)
                .unwrap(),
            dof_layout: StructuralDofLayoutIdentity::derive("test-layout", &1_u8).unwrap(),
            formulation: StructuralFormulationIdentity::derive("test-formulation", &1_u8).unwrap(),
            kinematics: StructuralKinematicsIdentity::derive("test-kinematics", &1_u8).unwrap(),
            physical_dof_count: 2,
            phasor_convention: StructuralPhasorConvention::PeakAmplitude,
        })
        .unwrap()
    }

    fn vector(real: [f64; 2], imaginary: [f64; 2]) -> StructuralComplexVector {
        StructuralComplexVector::new(real.to_vec(), imaginary.to_vec()).unwrap()
    }

    fn sample(
        scope: &StructuralFrequencyScope,
        ordinal: usize,
    ) -> StructuralFrequencyPhysicalSample {
        let entry = scope.plan().entries().unwrap()[ordinal];
        let scale = entry.frequency_hz() / 10.0;
        let applied = vector([scale, 0.0], [0.5 * scale, 0.0]);
        let zero = vector([0.0, 0.0], [0.0, 0.0]);
        let balance =
            ComplexStructuralBalance::new(applied.clone(), zero.clone(), zero, applied).unwrap();
        StructuralFrequencyPhysicalSample::new(
            scope,
            entry,
            vector([0.1 * scale, 0.0], [-0.2 * scale, 0.0]),
            vector([0.0, scale], [0.0, -scale]),
            balance,
        )
        .unwrap()
    }

    fn direct_evidence(
        sample: &StructuralFrequencyPhysicalSample,
    ) -> StructuralFrequencyEntrySolverEvidence {
        StructuralFrequencyEntrySolverEvidence::new(
            sample,
            StructuralFrequencySolverEvidenceInput {
                solver_method: SolverMethod::StructuralDirectHarmonicGmres,
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                preconditioner: SolverPreconditioner::None,
                iterations: 4,
                solver_residual_norm: 1.0e-12,
                solver_relative_residual: 1.0e-13,
                accelerator_evidence: Vec::new(),
            },
        )
        .unwrap()
    }

    fn modal_solution_space() -> StructuralFrequencySolutionSpace {
        StructuralFrequencySolutionSpace::ModalReduced {
            basis_state: StateIdentity::derive("test-basis-state", &1_u8).unwrap(),
            projection: ModalProjectionEvidence {
                basis: basis_identity(),
                operator_digest: operator_digest(),
                full_dof_count: 2,
                mode_count: 1,
                maximum_mass_off_diagonal_ratio: 0.0,
                maximum_stiffness_off_diagonal_ratio: 0.0,
                maximum_damping_asymmetry_ratio: 0.0,
            },
        }
    }

    fn modal_evidence(
        sample: &StructuralFrequencyPhysicalSample,
    ) -> StructuralFrequencyEntrySolverEvidence {
        StructuralFrequencyEntrySolverEvidence::new(
            sample,
            StructuralFrequencySolverEvidenceInput {
                solver_method: SolverMethod::StructuralModalHarmonicDenseDirect,
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                preconditioner: SolverPreconditioner::None,
                iterations: 1,
                solver_residual_norm: 1.0e-14,
                solver_relative_residual: 1.0e-15,
                accelerator_evidence: Vec::new(),
            },
        )
        .unwrap()
    }

    #[test]
    fn physical_result_and_receipt_round_trip_and_keep_method_separate() {
        let scope = scope();
        assert_eq!(
            StructuralFrequencyScope::from_canonical_bytes(&scope.canonical_bytes().unwrap())
                .unwrap(),
            scope
        );
        let samples = vec![sample(&scope, 0), sample(&scope, 1)];
        for physical_sample in &samples {
            assert_eq!(
                StructuralFrequencyPhysicalSample::from_canonical_bytes(
                    &physical_sample.canonical_bytes().unwrap()
                )
                .unwrap(),
                *physical_sample
            );
        }
        let result = StructuralFrequencySolverResult::assemble(scope, samples).unwrap();
        let manifest = super::super::StructuralFrequencySolverResultManifest::from_canonical_bytes(
            &result.manifest().canonical_bytes().unwrap(),
        )
        .unwrap();
        let reconstructed = StructuralFrequencySolverResult::from_manifest_and_samples(
            manifest,
            result.samples().to_vec(),
        )
        .unwrap();
        assert_eq!(reconstructed, result);

        let direct = StructuralFrequencyNumericalReceipt::new(
            &result,
            StructuralFrequencySolutionSpace::Direct,
            result.samples().iter().map(direct_evidence).collect(),
        )
        .unwrap();
        let decoded = StructuralFrequencyNumericalReceipt::from_canonical_bytes(
            &direct.canonical_bytes().unwrap(),
        )
        .unwrap();
        decoded.validate_for_result(&result).unwrap();

        let modal_evidence = result.samples().iter().map(modal_evidence).collect();
        let modal = StructuralFrequencyNumericalReceipt::new(
            &result,
            modal_solution_space(),
            modal_evidence,
        )
        .unwrap();
        assert_eq!(modal.source_result(), direct.source_result());
        assert_ne!(modal.identity(), direct.identity());
    }

    #[test]
    fn canonical_receipt_rejects_reidentified_invalid_modal_projection() {
        let scope = scope();
        let samples = vec![sample(&scope, 0), sample(&scope, 1)];
        let result = StructuralFrequencySolverResult::assemble(scope, samples).unwrap();
        let receipt = StructuralFrequencyNumericalReceipt::new(
            &result,
            modal_solution_space(),
            result.samples().iter().map(modal_evidence).collect(),
        )
        .unwrap();

        for mutation in 0..3 {
            let mut tampered = receipt.clone();
            let StructuralFrequencySolutionSpace::ModalReduced { projection, .. } =
                &mut tampered.solution_space
            else {
                unreachable!();
            };
            match mutation {
                0 => projection.mode_count = 0,
                1 => projection.mode_count = projection.full_dof_count + 1,
                _ => projection.operator_digest = Digest::sha256(b"another-operator"),
            }
            tampered.identity = tampered.derive_identity().unwrap();
            let encoded = encode_contract(
                FREQUENCY_RECEIPT_PREFIX,
                "structural-frequency-numerical-receipt",
                &tampered,
                FREQUENCY_RECEIPT_LIMITS,
            )
            .unwrap();
            assert!(StructuralFrequencyNumericalReceipt::from_canonical_bytes(&encoded).is_err());
        }
    }

    #[test]
    fn result_and_receipt_reject_missing_reordered_and_substituted_samples() {
        let scope = scope();
        let first = sample(&scope, 0);
        let second = sample(&scope, 1);
        assert!(
            StructuralFrequencySolverResult::assemble(scope.clone(), vec![first.clone()]).is_err()
        );
        assert!(StructuralFrequencySolverResult::assemble(
            scope.clone(),
            vec![second.clone(), first.clone()]
        )
        .is_err());
        let result = StructuralFrequencySolverResult::assemble(scope, vec![first, second]).unwrap();
        let mut evidence = result
            .samples()
            .iter()
            .map(direct_evidence)
            .collect::<Vec<_>>();
        evidence.swap(0, 1);
        assert!(StructuralFrequencyNumericalReceipt::new(
            &result,
            StructuralFrequencySolutionSpace::Direct,
            evidence,
        )
        .is_err());

        let mut tampered = serde_json::to_value(&result.samples()[0]).unwrap();
        tampered["reaction"]["real"][0] = serde_json::json!(99.0);
        assert!(serde_json::from_value::<StructuralFrequencyPhysicalSample>(tampered).is_err());
    }
}
