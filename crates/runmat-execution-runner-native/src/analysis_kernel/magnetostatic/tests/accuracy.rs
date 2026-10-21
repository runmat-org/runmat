//! Authored post-mesh plan reaches the shared native accuracy worker without a re-solve.
use super::*;
use runmat_analysis_core::{
    AccuracyCriterion, AccuracyTarget, AccuracyTolerance, ErrorContributionKind,
    RequiredAccuracyEvidence,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_accuracy, import_magnetostatic_result, prepare_accuracy,
    AccuracyPreparation, AnalysisHostResponse,
};
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SequenceIdentity},
    properties::compile_property_map,
    results::accuracy::AccuracyStatus,
    study::{plan_magnetic_study_product, CompiledAnalysisStudy, MagneticStudyPlanningPolicy},
};
use runmat_execution_artifact::cache::FilesystemObjectStore;

#[test]
fn native_magnetic_authored_plan_accuracy_preserves_unknown_spatial_evidence() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut source, mesh, original) = fixture::task(order, 2, 1.0, 1.0, 8192);
        source.model.sequence.steps[0].accuracy.criteria = [
            (
                "residual",
                ErrorContributionKind::Algebraic,
                RequiredAccuracyEvidence::ResidualBound,
            ),
            (
                "closure",
                ErrorContributionKind::Algebraic,
                RequiredAccuracyEvidence::ConservationBalance,
            ),
            (
                "spatial",
                ErrorContributionKind::SpatialDiscretization,
                RequiredAccuracyEvidence::ConvergenceStudy,
            ),
        ]
        .into_iter()
        .map(|(id, contribution, evidence)| AccuracyCriterion {
            criterion_id: id.into(),
            target: if contribution == ErrorContributionKind::SpatialDiscretization {
                AccuracyTarget::NumericalResult
            } else {
                AccuracyTarget::GlobalEquilibrium
            },
            contribution,
            evidence,
            tolerance: AccuracyTolerance::Relative(1e-8),
            mandatory: true,
        })
        .collect();
        let compiled = CompiledAnalysisStudy {
            module_path: "qualification/magnetic.fea".into(),
            study_id: "magnetic".into(),
            study_inputs: Vec::new(),
            assessments: Vec::new(),
            outputs: Vec::new(),
            properties: compile_property_map(&source.model, Some(&mesh)).unwrap(),
            model: source.model,
            solver_mesh_digest: mesh.canonical_digest,
        };
        let product =
            plan_magnetic_study_product(compiled, &mesh, MagneticStudyPlanningPolicy {}).unwrap();
        let task = prepare_magnetostatic_task(MagnetostaticTaskPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: product.compiled.definition_digest().unwrap(),
            frozen_project_digest: None,
            program_revision: original.program.request.recipe.program_revision.clone(),
            artifact_access: original.host.artifact_access.clone(),
            workload: product.workload.clone(),
            mesh,
            output_policy: original.output_policy,
            required_capabilities: Default::default(),
        })
        .unwrap();
        let limits = task.output_policy.inventory_limits();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cold");
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        task.program.write_to(&mut store).unwrap();
        let done = execute_serial_analysis(
            &task.program.request,
            &mut store,
            &NativeAnalysisKernel,
            8192,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&task.host, &done)
            .unwrap()
            .program_response();
        let imported = import_magnetostatic_result(&store, response, &task, limits).unwrap();
        let mut context = task
            .inputs
            .iter()
            .filter(|input| {
                matches!(
                    input.manifest.kind,
                    AnalysisNumericalObjectKind::MagnetostaticWorkload
                        | AnalysisNumericalObjectKind::Mesh
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        context.push(imported.objects.clone());
        let accuracy = prepare_accuracy(AccuracyPreparation {
            solve_host: task.host.clone(),
            source_kind: AccuracySourceKind::Magnetostatic,
            source_numerical_receipt: imported.numerical_receipt_objects,
            source_observations: None,
            source_context_inputs: context,
            sequence: SequenceIdentity::from_solver_contract(&product.workload.model.sequence)
                .unwrap(),
            step: product.workload.model.sequence.steps[0].clone(),
            program_revision: task.program.request.recipe.program_revision.clone(),
            chunk_bytes: 8192,
            inventory_limits: limits,
        })
        .unwrap();
        accuracy.program.write_to(&mut store).unwrap();
        let done = execute_serial_analysis(
            &accuracy.program.request,
            &mut store,
            &NativeAnalysisKernel,
            8192,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&accuracy.host, &done)
            .unwrap()
            .program_response();
        drop(store);
        let store = FilesystemObjectStore::open(path, limits.max_object_bytes).unwrap();
        let report = import_accuracy(&store, response, &accuracy, limits).unwrap();
        assert_eq!(
            report.artifact.evaluation().report().status,
            AccuracyStatus::Indeterminate
        );
        assert_eq!(report.artifact.evidence().len(), 2);
        let mut missing = accuracy.inputs.clone();
        missing.retain(|input| input.manifest.kind != AnalysisNumericalObjectKind::Mesh);
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::EvaluateAccuracy,
                host: &accuracy.host,
                inputs: &missing,
                output_policy: task.output_policy,
            })
            .is_err());
    }
}
