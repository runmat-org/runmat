use runmat_analysis_core::{IndependentDomain, ProcedureStep};

use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    ComputeBackend, ModalConvergencePolicy, ModalTruncationPolicy,
};

fn modal_workload_and_result() -> (StructuralModalWorkload, StructuralModalSolverResult) {
    let truncation = ModalTruncationPolicy::requested(3);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let workload = StructuralModalWorkload::new(super::super::StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: runmat_analysis_core::ProcedureStepId("modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        truncation,
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let modal = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        crate::ModalSolveOptions {
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            solver_mesh: None,
        },
    )
    .unwrap();
    let result = StructuralModalSolverResult::new(
        &workload,
        Digest::sha256(b"modal-topology"),
        modal.run,
        modal.basis,
    )
    .unwrap();
    (workload, result)
}

#[test]
fn modal_result_and_receipt_round_trip_with_exact_source_binding() {
    let (workload, result) = modal_workload_and_result();
    let restored =
        StructuralModalSolverResult::from_canonical_bytes(&result.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(restored, result);
    restored.validate_for(&workload).unwrap();

    let generic_workload =
        AnalysisWorkloadIdentity::derive("modal-result-test-workload", &workload.identity())
            .unwrap();
    let receipt = StructuralModalNumericalReceipt::new(generic_workload, &result).unwrap();
    let restored_receipt =
        StructuralModalNumericalReceipt::from_canonical_bytes(&receipt.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(restored_receipt, receipt);
    receipt
        .validate_against(receipt.source_result(), &result)
        .unwrap();
}

#[test]
fn portable_result_and_receipt_bind_residual_vector_inventory() {
    let truncation = ModalTruncationPolicy::requested(2);
    let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let direction = runmat_analysis_core::SpectrumDirection {
        frame: runmat_analysis_core::CoordinateFrame::Global,
        vector: [1.0, 0.5, 0.25],
    };
    let workload = StructuralModalWorkload::new(super::super::StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: runmat_analysis_core::ProcedureStepId("modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        truncation,
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: vec![direction.clone()],
        prestress_transition: None,
    })
    .unwrap();
    let modal = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        crate::ModalSolveOptions {
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: vec![direction],
            solver_mesh: None,
        },
    )
    .unwrap();
    let topology_digest = crate::physics::structural::topology::explicit_topology_digest(
        model.structural.as_ref().unwrap(),
    )
    .unwrap();
    let result =
        StructuralModalSolverResult::new(&workload, topology_digest, modal.run, modal.basis)
            .unwrap();
    let source = AnalysisWorkloadIdentity::derive(
        "modal-residual-result-test-workload",
        &workload.identity(),
    )
    .unwrap();
    let receipt = StructuralModalNumericalReceipt::new(source, &result).unwrap();
    assert_eq!(receipt.residual_vector_count(), 1);
    assert!(receipt.maximum_residual_vector_static_equilibrium_ratio() <= 1.0e-8);
    assert!(receipt.maximum_residual_vector_mass_orthogonality_ratio() <= 1.0e-8);

    let omitted = StructuralModalWorkload::new(super::super::StructuralModalWorkloadInput {
        model,
        step_id: runmat_analysis_core::ProcedureStepId("modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        truncation,
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    assert!(result.validate_for(&omitted).is_err());
}

#[test]
fn modal_result_rejects_a_tampered_basis_inventory() {
    let (_, result) = modal_workload_and_result();
    let mut wire = serde_json::to_value(&result).unwrap();
    wire["basis"]["modes"].as_array_mut().unwrap().pop();
    assert!(serde_json::from_value::<StructuralModalSolverResult>(wire).is_err());
}
