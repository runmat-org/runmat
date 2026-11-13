//! Original complete point state and truthful missing spatial evidence across cold tasks.
use super::*;
mod admission;
mod shared_history;
use runmat_analysis_execution::{
    import_accuracy, import_nonlinear_solid_block, import_semantic_state, prepare_accuracy,
    prepare_nonlinear_solid_state_publication, AccuracyPreparation, AnalysisHostWorkload,
    NonlinearSolidStatePublicationPreparation,
};
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, CanonicalSemanticStatePayload, SequenceIdentity},
    results::accuracy::{publication_gate, AccuracyStatus, PublicationDecision},
};
use runmat_execution_artifact::cache::FilesystemObjectStore;

#[test]
fn cold_original_material_history_and_accuracy_do_not_promote_missing_mesh_evidence() {
    for quadratic in [false, true] {
        let (mut workload, mesh) = fixture(quadratic);
        workload.model.sequence.steps[0].accuracy.criteria = vec![
            AccuracyCriterion {
                criterion_id: "newton".into(),
                target: AccuracyTarget::GlobalEquilibrium,
                contribution: ErrorContributionKind::Nonlinear,
                tolerance: AccuracyTolerance::Relative(1e-9),
                evidence: RequiredAccuracyEvidence::ResidualBound,
                mandatory: true,
            },
            AccuracyCriterion {
                criterion_id: "mesh".into(),
                target: AccuracyTarget::NumericalResult,
                contribution: ErrorContributionKind::SpatialDiscretization,
                tolerance: AccuracyTolerance::Relative(1e-7),
                evidence: RequiredAccuracyEvidence::ConvergenceStudy,
                mandatory: true,
            },
        ];
        let request = NonlinearSolidBlockRequest {
            workload: workload.identity().unwrap(),
            topology: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            block_ordinal: 0,
            first_output_ordinal: 1,
            accepted_block_limit: 12,
            predecessor: None,
        };
        let solve = prepare(&workload, &mesh, request, None);
        let policy = solve.output_policy;
        let limits = policy.inventory_limits();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("store");
        let done = super::process::run_terminated(directory.path(), &solve.program, policy);
        let store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let imported = import_nonlinear_solid_block(&store, done, &solve).unwrap();
        assert!(imported.block.complete);
        let original = imported.block.result.accepted_state.clone();
        let workload_objects = solve
            .inputs
            .iter()
            .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::NonlinearSolidWorkload)
            .unwrap()
            .clone();
        let mesh_objects = solve
            .inputs
            .iter()
            .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::Mesh)
            .unwrap()
            .clone();
        let accuracy = prepare_accuracy(AccuracyPreparation {
            solve_host: solve.host.clone(),
            source_kind: AccuracySourceKind::NonlinearSolid,
            source_numerical_receipt: imported.receipt_objects.clone(),
            source_observations: None,
            source_context_inputs: vec![
                workload_objects.clone(),
                mesh_objects,
                imported.objects.clone(),
            ],
            sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
            step: workload.model.sequence.steps[0].clone(),
            program_revision: solve.program.request.recipe.program_revision.clone(),
            chunk_bytes: policy.chunk_bytes(),
            inventory_limits: limits,
        })
        .unwrap();
        admission::reject_understated_decode(
            &accuracy.resource_plan,
            &accuracy.inputs,
            &accuracy.host,
            policy,
        );
        drop(store);
        let done = super::process::run_terminated(directory.path(), &accuracy.program, policy);
        let store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let cold_accuracy = import_accuracy(&store, done, &accuracy, limits).unwrap();
        assert_eq!(
            cold_accuracy.artifact.evaluation().report().status,
            AccuracyStatus::Indeterminate
        );
        let contributions = &cold_accuracy.artifact.evaluation().report().contributions;
        assert_eq!(
            contributions
                .iter()
                .find(|c| c.criterion_id == "newton")
                .unwrap()
                .status,
            AccuracyStatus::Satisfied
        );
        assert_eq!(
            contributions
                .iter()
                .find(|c| c.criterion_id == "mesh")
                .unwrap()
                .status,
            AccuracyStatus::Indeterminate
        );
        assert_eq!(
            publication_gate(&accuracy.request.plan, cold_accuracy.artifact.evaluation())
                .unwrap()
                .decision,
            PublicationDecision::Block
        );
        drop(store);
        for declaration in &workload.model.sequence.steps[0].state_outputs {
            let publication = prepare_nonlinear_solid_state_publication(
                NonlinearSolidStatePublicationPreparation {
                    source_host: solve.host.clone(),
                    workload: workload_objects.clone(),
                    block: imported.objects.clone(),
                    declaration: declaration.clone(),
                    program_revision: solve.program.request.recipe.program_revision.clone(),
                    output_policy: policy,
                },
            )
            .unwrap();
            admission::reject_understated_decode(
                &publication.resource_plan,
                &publication.inputs,
                &publication.host,
                policy,
            );
            let done =
                super::process::run_terminated(directory.path(), &publication.program, policy);
            let store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            let state = import_semantic_state(&store, done, &publication, limits).unwrap();
            assert_eq!(state.kind, declaration.kind);
            shared_history::verify(&state);
            if declaration.kind == StateKind::MaterialHistory {
                assert!(runmat_analysis_fea::execution::CanonicalSemanticState::new(
                    runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                        producer: publication.request.producer,
                        declaration: publication.request.declaration.clone(),
                        topology_digest: publication.request.topology_digest,
                        source_result: publication.request.source_result,
                        payload: CanonicalSemanticStatePayload::Fields {
                            fields: vec![AnalysisField::host_f64(
                                "alpha",
                                runmat_analysis_core::AnalysisFieldLayout::scalar(
                                    runmat_analysis_core::ObservationQuantity::Residual,
                                    runmat_analysis_core::FieldLocation::Global
                                ),
                                vec![1],
                                vec![0.02]
                            )]
                        },
                    }
                )
                .is_err());
            }
            let CanonicalSemanticStatePayload::NonlinearSolid { state: history } = state.payload
            else {
                panic!("original point and field payload required");
            };
            assert_eq!(history.block.result.accepted_state, original);
            assert!(history
                .block
                .result
                .accepted_state
                .point_history()
                .all(|(_, _, p)| (p.equivalent_plastic_strain - 0.02).abs() < 2e-9));
            for strain in history
                .block
                .result
                .total_strain_snapshots
                .last()
                .unwrap()
                .chunks_exact(6)
            {
                for (actual, expected) in
                    strain.iter().zip([0.0256, -0.0114, -0.0114, 0.0, 0.0, 0.0])
                {
                    assert!((actual - expected).abs() < 2e-9);
                }
            }
        }
    }
}
