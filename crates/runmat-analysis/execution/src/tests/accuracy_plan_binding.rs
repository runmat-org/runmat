//! Plan binding for the nonlinear-solid and acoustic-response accuracy source contexts: the same
//! original receipt and inputs validate only under the authored accuracy plan.
use crate::{prepare_analysis_objects, PreparedAnalysisObjects};
use runmat_analysis_core::*;
use runmat_analysis_fea::execution::{
    AccuracyEvaluationRequest, AccuracySourceKind, AccuracySourceReceipt,
    AnalysisNumericalObjectKind as Kind,
};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_execution::Digest;
use runmat_execution_artifact::object::{ObjectInventoryLimits, ObjectNamespace};
use runmat_meshing_core::{CanonicalMeshingContract, ElementOrder, SolverMeshArtifact};

fn object(
    kind: Kind,
    identity: Digest,
    bytes: Vec<u8>,
    namespace: ObjectNamespace,
) -> PreparedAnalysisObjects {
    prepare_analysis_objects(
        kind,
        identity,
        &bytes,
        1024,
        namespace,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}

fn mesh_object(mesh: &SolverMeshArtifact) -> PreparedAnalysisObjects {
    object(
        Kind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        mesh.canonical_encode().unwrap(),
        ObjectNamespace::InputValue,
    )
}

// witness: FEA-NONLINEAR-GOVERNANCE-001/inequality_restart_converged_base @
//   nonlinear_structural/tet4/numerical_kernel; coverage: partial;
//   boundary: The nonlinear accuracy source context accepts a completed block only under the
//   authored accuracy plan and rejects a re-identified plan or loosened criterion; context check on
//   one cantilever block, not contact, restart or downstream transfer.
#[test]
fn nonlinear_accuracy_context_binds_the_authored_accuracy_plan() {
    use runmat_analysis_fea::execution::{
        NonlinearSolidBlockRequest, NonlinearSolidNumericalReceiptArtifact, NonlinearSolidWorkload,
    };
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (index, name) in [(0, "root"), (1, "tip")] {
        let face = &mut mesh.topology.boundary_faces[index];
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            name.into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    mesh.seal_canonical_digest().unwrap();
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence = AnalysisSequence::from_steps(
        "nonlinear_cantilever",
        vec![ProcedureStep::structural_nonlinear("load")],
    );
    let workload = NonlinearSolidWorkload {
        model,
        controls: runmat_analysis_fea::solve::nonlinear::NonlinearSolveOptions {
            increment_count: 1,
            ..Default::default()
        },
    };
    let request = NonlinearSolidBlockRequest {
        workload: workload.identity().unwrap(),
        topology: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        block_ordinal: 0,
        first_output_ordinal: 1,
        accepted_block_limit: 8,
        predecessor: None,
    };
    let block = workload
        .execute_block(mesh.clone(), &request, None)
        .unwrap();
    assert!(block.complete);
    let receipt = NonlinearSolidNumericalReceiptArtifact::new(&workload, &block).unwrap();
    let inputs = [
        object(
            Kind::NonlinearSolidWorkload,
            workload.identity().unwrap().digest(),
            workload.canonical_bytes().unwrap(),
            ObjectNamespace::InputValue,
        ),
        mesh_object(&mesh),
        object(
            Kind::NonlinearSolidBlockArtifact,
            block.identity,
            block.canonical_bytes().unwrap(),
            ObjectNamespace::ResultValue,
        ),
    ];
    let accuracy = AccuracyEvaluationRequest::new(
        workload.producer_identity().unwrap(),
        AccuracySourceKind::NonlinearSolid,
        receipt.source_result(),
        None,
        workload.model.sequence.steps[0].accuracy.clone(),
    )
    .unwrap();
    super::accuracy_source_context_tests::assert_accuracy_plan_identity_bound(
        &accuracy,
        &AccuracySourceReceipt::NonlinearSolid(receipt),
        None,
        &inputs.iter().collect::<Vec<_>>(),
    );
}

// witness: FEA-ACOUSTIC-MODAL-001/modal_publication @ acoustic_modal/tet10/numerical_kernel;
//   coverage: partial;
//   boundary: The acoustic-response accuracy source context accepts the modal cavity result only
//   under the authored accuracy plan and rejects a re-identified plan or loosened criterion;
//   context check only, no publication or mode accuracy.
#[test]
fn acoustic_response_accuracy_context_binds_the_authored_accuracy_plan() {
    use runmat_analysis_fea::execution::{
        AcousticResponseNumericalReceiptArtifact, AcousticResponseSolver, AcousticResponseWorkload,
    };
    use runmat_analysis_fea::physics::acoustic::AcousticModalPolicy;
    let mesh = runmat_meshing_core::fixtures::quadratic_slab_mesh(2);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.constraints.clear();
    model.boundaries = vec![AnalysisBoundary {
        boundary_id: "closed_end".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        operator: BoundaryOperator::AcousticRigidWall,
    }];
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::Quantity::new(3.0));
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.sequence.steps = vec![ProcedureStep::acoustic_modal("cavity", 4)];
    let workload = AcousticResponseWorkload {
        model,
        step_id: ProcedureStepId("cavity".into()),
        solver: AcousticResponseSolver::Modal {
            policy: AcousticModalPolicy::default(),
        },
    };
    let result = workload.execute(&mesh).unwrap();
    let receipt = AcousticResponseNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
    let inputs = [
        object(
            Kind::AcousticResponseWorkload,
            workload.identity().unwrap().digest(),
            workload.canonical_bytes().unwrap(),
            ObjectNamespace::InputValue,
        ),
        mesh_object(&mesh),
        object(
            Kind::SolverResult,
            result.source_result_identity().unwrap().digest(),
            result.canonical_bytes().unwrap(),
            ObjectNamespace::ResultValue,
        ),
    ];
    let accuracy = AccuracyEvaluationRequest::new(
        receipt.receipt().producer,
        AccuracySourceKind::AcousticResponse,
        receipt.source_result(),
        None,
        workload.model.sequence.steps[0].accuracy.clone(),
    )
    .unwrap();
    super::accuracy_source_context_tests::assert_accuracy_plan_identity_bound(
        &accuracy,
        &AccuracySourceReceipt::AcousticResponse(receipt),
        None,
        &inputs.iter().collect::<Vec<_>>(),
    );
}
