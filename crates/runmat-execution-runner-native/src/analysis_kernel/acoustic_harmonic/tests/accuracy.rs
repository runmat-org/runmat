//! Accepted acoustic fields feed shared accuracy without a second solve.
use super::*;
use runmat_analysis_core::{
    AccuracyCriterion, AccuracyTarget, AccuracyTolerance, ErrorContributionKind,
    RequiredAccuracyEvidence,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_accuracy, import_acoustic_harmonic_result, prepare_accuracy,
    validate_accuracy_source_context, AccuracyPreparation, AnalysisHostResponse,
};
use runmat_analysis_fea::execution::{AccuracySourceKind, AccuracySourceReceipt, SequenceIdentity};
use runmat_analysis_fea::results::accuracy::AccuracyStatus;
use runmat_execution_artifact::cache::FilesystemObjectStore;

#[test]
fn native_acoustic_cold_receipt_shared_accuracy_and_exact_source_matrix() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut expected = None;
        for chunk in [1024, 8192] {
            let (mut workload, mesh, template) = task(order, chunk);
            workload.model.sequence.steps[0].accuracy.criteria = [
                RequiredAccuracyEvidence::ResidualBound,
                RequiredAccuracyEvidence::ConservationBalance,
            ]
            .into_iter()
            .enumerate()
            .map(|(i, evidence)| AccuracyCriterion {
                criterion_id: format!("closure_{i}"),
                target: AccuracyTarget::GlobalEquilibrium,
                contribution: ErrorContributionKind::Algebraic,
                tolerance: AccuracyTolerance::Relative(1e-7),
                evidence,
                mandatory: true,
            })
            .collect();
            let limits = ObjectInventoryLimits::default();
            let policy = AnalysisOutputPolicy::new(chunk, limits).unwrap();
            let task = prepare_acoustic_harmonic_task(AcousticHarmonicTaskPreparation {
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                authored_study_digest: Digest::sha256("acoustic-accuracy"),
                frozen_project_digest: None,
                program_revision: template.program.request.recipe.program_revision.clone(),
                artifact_access: template.host.artifact_access.clone(),
                workload: workload.clone(),
                mesh: mesh.clone(),
                output_policy: policy,
                required_capabilities: Default::default(),
            })
            .unwrap();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("objects");
            let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            task.program.write_to(&mut store).unwrap();
            let completed = execute_serial_analysis(
                &task.program.request,
                &mut store,
                &NativeAnalysisKernel,
                chunk,
                limits,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&task.host, &completed)
                .unwrap()
                .program_response();
            drop(store);
            let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            let imported =
                import_acoustic_harmonic_result(&store, response, &task, limits).unwrap();
            let mut context: Vec<_> = task
                .inputs
                .iter()
                .filter(|input| {
                    matches!(
                        input.manifest.kind,
                        AnalysisNumericalObjectKind::AcousticHarmonicWorkload
                            | AnalysisNumericalObjectKind::Mesh
                    )
                })
                .cloned()
                .collect();
            context.push(imported.objects.clone());
            let prepared = prepare_accuracy(AccuracyPreparation {
                solve_host: task.host.clone(),
                source_kind: AccuracySourceKind::AcousticHarmonic,
                source_numerical_receipt: imported.numerical_receipt_objects.clone(),
                source_observations: None,
                source_context_inputs: context.clone(),
                sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
                step: workload.model.sequence.steps[0].clone(),
                program_revision: task.program.request.recipe.program_revision.clone(),
                chunk_bytes: chunk,
                inventory_limits: limits,
            })
            .unwrap();
            let source = AccuracySourceReceipt::AcousticHarmonic(imported.numerical_receipt);
            let check = |inputs: &[_]| {
                validate_accuracy_source_context(
                    &prepared.request,
                    &source,
                    None,
                    &inputs.iter().collect::<Vec<_>>(),
                )
            };
            check(&context).unwrap();
            for removed in 0..context.len() {
                let mut missing = context.clone();
                missing.remove(removed);
                assert!(check(&missing).is_err());
                let mut duplicate = context.clone();
                duplicate.push(context[removed].clone());
                assert!(check(&duplicate).is_err());
            }
            let (_, _, foreign) = super::task(order, chunk);
            for kind in [
                AnalysisNumericalObjectKind::AcousticHarmonicWorkload,
                AnalysisNumericalObjectKind::Mesh,
            ] {
                let mut substituted = context.clone();
                let index = substituted
                    .iter()
                    .position(|v| v.manifest.kind == kind)
                    .unwrap();
                let other_mesh;
                let replacement = if kind == AnalysisNumericalObjectKind::Mesh {
                    other_mesh = super::task(
                        if order == ElementOrder::Tet4 {
                            ElementOrder::Tet10
                        } else {
                            ElementOrder::Tet4
                        },
                        chunk,
                    )
                    .2;
                    &other_mesh
                } else {
                    &foreign
                };
                substituted[index] = replacement
                    .inputs
                    .iter()
                    .find(|v| v.manifest.kind == kind)
                    .unwrap()
                    .clone();
                assert!(check(&substituted).is_err());
            }
            prepared.program.write_to(&mut store).unwrap();
            let completed = execute_serial_analysis(
                &prepared.program.request,
                &mut store,
                &NativeAnalysisKernel,
                chunk,
                limits,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&prepared.host, &completed)
                .unwrap()
                .program_response();
            drop(store);
            let store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            let accuracy = import_accuracy(&store, response, &prepared, limits).unwrap();
            assert_eq!(
                accuracy.artifact.evaluation().report().status,
                AccuracyStatus::Satisfied
            );
            let identity = accuracy.objects.manifest.logical_identity;
            if let Some(expected) = expected {
                assert_eq!(identity, expected);
            } else {
                expected = Some(identity);
            }
        }
    }
}
