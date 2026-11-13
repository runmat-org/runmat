use runmat_analysis_core::{IndependentDomain, ProcedureStep, ProcedureStepId};
use runmat_meshing_core::CanonicalMeshingContract;

use super::*;
use crate::{
    execution::{StructuralModalNumericalReceipt, StructuralModalWorkloadInput},
    fixtures::{fixture_model, FixtureId},
    pipeline::modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
    results::observation::ObservationExecutionPolicy,
};

fn portable_fixture() -> (StructuralModalWorkload, SolverMeshArtifact) {
    let truncation = ModalTruncationPolicy::requested(2);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model,
        step_id: ProcedureStepId("modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation,
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    (workload, mesh)
}

#[test]
fn portable_modal_boundary_is_numerically_equivalent_to_the_in_process_api() {
    let (workload, mesh) = portable_fixture();
    let in_process = run_modal_with_options(
        workload.model(),
        workload.backend(),
        ModalSolveOptions {
            truncation: workload.truncation(),
            convergence: workload.convergence(),
            residual_vector_directions: workload.residual_vector_directions().to_vec(),
            solver_mesh: Some(mesh.clone()),
        },
    )
    .unwrap();
    let portable = execute_structural_modal_workload(&workload, &mesh).unwrap();

    assert_eq!(
        portable.topology_digest(),
        Digest::from_bytes(*mesh.canonical_digest.bytes())
    );
    portable.validate_for(&workload).unwrap();
    let source_workload =
        crate::execution::AnalysisWorkloadIdentity::for_test("portable-modal-equivalence").unwrap();
    let receipt = StructuralModalNumericalReceipt::new(source_workload, &portable).unwrap();
    let recovered =
        recover_completed_structural_modal_run(&workload, &mesh, &portable, &receipt).unwrap();

    assert_eq!(portable.basis(), &in_process.basis);
    assert_eq!(recovered, in_process);
}

#[test]
fn portable_modal_boundary_rejects_a_canonical_mesh_from_another_geometry_revision() {
    let (workload, mut mesh) = portable_fixture();
    mesh.geometry.geometry_revision += 1;
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    mesh.validate_canonical().unwrap();
    let error = execute_structural_modal_workload(&workload, &mesh).unwrap_err();
    assert!(matches!(error, FeaRunError::InvalidModel(_)));
}
