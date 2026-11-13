use super::*;
use crate::{
    execution::{
        ComplexStructuralBalance, ConstraintPlanIdentity, DiscretizationIdentity,
        StructuralComplexVector, StructuralDampingOperatorIdentity, StructuralDofLayoutIdentity,
        StructuralExcitationSetIdentity, StructuralFormulationIdentity,
        StructuralFrequencyProblemIdentity, StructuralFrequencyScope,
        StructuralFrequencyScopeInput, StructuralKinematicsIdentity,
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
    scope_with_dofs(2)
}

fn scope_with_dofs(physical_dof_count: usize) -> StructuralFrequencyScope {
    let plan = FrequencyResponsePlan::new(20.0, &[10.0], &[]).unwrap();
    StructuralFrequencyScope::new(StructuralFrequencyScopeInput {
        problem: StructuralFrequencyProblemIdentity::from_physical_digest(Digest::sha256(
            b"test-frequency-problem",
        ))
        .unwrap(),
        plan,
        discretization: DiscretizationIdentity::derive("test-discretization", &1_u8).unwrap(),
        constraint_plan: ConstraintPlanIdentity::derive("test-constraints", &1_u8).unwrap(),
        stiffness_operator: StructuralStiffnessOperatorIdentity::derive("test-stiffness", &1_u8)
            .unwrap(),
        mass_operator: StructuralMassOperatorIdentity::derive("test-mass", &1_u8).unwrap(),
        damping_operator: StructuralDampingOperatorIdentity::derive("test-damping", &1_u8).unwrap(),
        operator_digest: operator_digest(),
        excitations: StructuralExcitationSetIdentity::derive("test-excitations", &1_u8).unwrap(),
        dof_layout: StructuralDofLayoutIdentity::derive("test-layout", &1_u8).unwrap(),
        formulation: StructuralFormulationIdentity::derive("test-formulation", &1_u8).unwrap(),
        kinematics: StructuralKinematicsIdentity::derive("test-kinematics", &1_u8).unwrap(),
        physical_dof_count,
        phasor_convention: StructuralPhasorConvention::PeakAmplitude,
    })
    .unwrap()
}

#[test]
fn large_physical_sample_uses_its_numerical_contract_bound_not_small_identity_bound() {
    let scope = scope_with_dofs(30_000);
    let displacement =
        StructuralComplexVector::new(vec![0.123; 30_000], vec![0.456; 30_000]).unwrap();
    let applied = StructuralComplexVector::new(vec![1.234; 30_000], vec![0.567; 30_000]).unwrap();
    let zero = StructuralComplexVector::new(vec![0.0; 30_000], vec![0.0; 30_000]).unwrap();
    let balance =
        ComplexStructuralBalance::new(applied.clone(), zero.clone(), zero.clone(), applied)
            .unwrap();
    let sample = StructuralFrequencyPhysicalSample::new(
        &scope,
        scope.plan().entries().unwrap()[0],
        displacement,
        zero,
        balance,
    )
    .unwrap();
    let bytes = sample.canonical_bytes().unwrap();
    assert!(bytes.len() > 1024 * 1024);
    let decoded = StructuralFrequencyPhysicalSample::from_canonical_bytes(&bytes).unwrap();
    decoded.validate_against(&scope).unwrap();
    assert_eq!(sample, decoded);
    let mut tampered = serde_json::to_value(&sample).unwrap();
    tampered["displacement"]["real"][123] = serde_json::json!(0.124);
    assert!(serde_json::from_value::<StructuralFrequencyPhysicalSample>(tampered).is_err());
}

fn vector(real: [f64; 2], imaginary: [f64; 2]) -> StructuralComplexVector {
    StructuralComplexVector::new(real.to_vec(), imaginary.to_vec()).unwrap()
}

fn sample(scope: &StructuralFrequencyScope, ordinal: usize) -> StructuralFrequencyPhysicalSample {
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
            preconditioner: SolverPreconditioner::Jacobi,
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
        StructuralFrequencyScope::from_canonical_bytes(&scope.canonical_bytes().unwrap()).unwrap(),
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
    assert_eq!(
        decoded.samples[0].preconditioner(),
        SolverPreconditioner::Jacobi
    );
    let mut wrong_preconditioner = direct_evidence(&result.samples()[0]);
    wrong_preconditioner.preconditioner = SolverPreconditioner::None;
    assert!(wrong_preconditioner.validate_intrinsic().is_err());

    let modal_evidence = result.samples().iter().map(modal_evidence).collect();
    let modal =
        StructuralFrequencyNumericalReceipt::new(&result, modal_solution_space(), modal_evidence)
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
    assert!(StructuralFrequencySolverResult::assemble(scope.clone(), vec![first.clone()]).is_err());
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
