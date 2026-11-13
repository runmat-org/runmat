//! Frozen completion through the real numerical store and final publication boundary.
use runmat_analysis_core::AssessmentStatus;
use runmat_analysis_execution::assessment_program::FrozenAssessmentCallable;
use runmat_execution_runner_native::{
    complete_frozen_fea_assessments, execute_prepared_structural_study_locally,
    freeze_fea_assessment_functions, prepare_checked_fea_study, prepare_structural_study_root,
    LocalFeaExecutionOptions, NativeExecutionConfig,
};
use runmat_runtime::analysis::{
    prepare_structural_study_publication, AnalysisStudyCompletionIdentity, FeaCheckStatus,
    FeaRootSelector, ResolvedFeaRoot,
};
use runmat_runtime::operations::OperationContext;

pub(super) async fn run() {
    let directory = tempfile::tempdir().unwrap();
    let project_directory = directory.path().canonicalize().unwrap();
    let sources = project_directory.join("src");
    std::fs::create_dir_all(sources.join("geometry")).unwrap();
    std::env::set_var(
        "RUNMAT_EXECUTION_STATE_DIR",
        project_directory.join("native-state"),
    );
    std::env::set_var(
        "RUNMAT_FEA_ARTIFACT_ROOT",
        project_directory.join("artifacts"),
    );
    std::env::set_var(
        "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        project_directory.join("study-artifacts"),
    );
    runmat_runtime::analysis::storage::configure_artifact_store_from_env().unwrap();
    std::fs::write(
        project_directory.join("runmat.toml"),
        "[package]\nname = \"frozen-assessment-publication\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    std::fs::write(
        sources.join("geometry/box.brep"),
        include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep"),
    )
    .unwrap();
    std::fs::write(
        sources.join("verify_peak.m"),
        include_str!("../../../runmat-cli/tests/fixtures/verify_peak.m"),
    )
    .unwrap();
    let path = sources.join("main.fea");
    std::fs::write(
        &path,
        include_str!("../../../runmat-cli/tests/fixtures/typed_box_transient_assessments.fea"),
    )
    .unwrap();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let project = runmat_package::build_frozen_project(
        &project_directory.join("runmat.toml"),
        std::collections::BTreeSet::new(),
    )
    .unwrap();
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project))
        .await
        .unwrap();
    let checked = session.load_and_check_fea_source_path(&path).await.unwrap();
    let report = checked.checked().check_result();
    assert_eq!(
        report.status,
        FeaCheckStatus::Valid,
        "{:?}",
        report.diagnostics
    );
    let options = LocalFeaExecutionOptions {
        root: Some(FeaRootSelector::Study {
            id: "structural_transient".into(),
        }),
        workers: Some(1),
        ..Default::default()
    };
    let ResolvedFeaRoot::Study(resolved) = checked.resolve_root(options.root.as_ref()).unwrap()
    else {
        panic!("expected authored transient study");
    };
    let submitted = freeze_fea_assessment_functions(&checked, &resolved.study.assessments)
        .await
        .unwrap();
    let restored: Vec<FrozenAssessmentCallable> =
        serde_json::from_slice(&serde_json::to_vec(&submitted).unwrap()).unwrap();
    assert_eq!(restored.len(), 1);
    restored[0].validate().unwrap();
    let prepared = prepare_checked_fea_study(
        &checked,
        resolved,
        &[],
        "frozen-assessment-publication",
        &options,
    )
    .await
    .unwrap();
    assert!(
        prepared.root.planning.structural().is_some(),
        "expected structural root"
    );
    let workers = prepared.workers;
    let structural_root = prepare_structural_study_root(prepared.root).unwrap();
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = project_directory.join("numerical-processes");
    let numerical = execute_prepared_structural_study_locally(config, structural_root, workers)
        .await
        .unwrap();
    // Completion consumes admitted numerical evidence and submitted executables. No source
    // file remains for an accidental recompilation or second geometry/solve pipeline.
    drop(checked);
    drop(session);
    std::fs::remove_dir_all(&sources).unwrap();
    let mut original = None;
    for _ in 0..2 {
        let publication = prepare_structural_study_publication(
            &numerical,
            OperationContext::new(None, None),
            AnalysisStudyCompletionIdentity::new("frozen-assessment-publication").unwrap(),
        )
        .unwrap();
        assert_eq!(publication.function_inputs().count(), 1);
        let result = complete_frozen_fea_assessments(publication, &restored)
            .await
            .unwrap();
        assert_eq!(result.assessments.len(), 2);
        assert!(result
            .assessments
            .iter()
            .all(|report| report.result.status == AssessmentStatus::Passed));
        let custom = result
            .assessments
            .iter()
            .find(|report| report.definition.assessment_id == "custom_deflection")
            .unwrap();
        assert!(custom.function_execution_identity.is_some());
        let measured = custom.evidence.as_ref().unwrap().value;
        assert!(measured > 0.0 && measured < 0.001, "{measured}");
        let manifest = result.result_manifest_artifact_path.as_ref().unwrap();
        let bytes = std::fs::read(manifest).unwrap();
        let replay = (manifest.clone(), bytes, result.assessments);
        if let Some(original) = &original {
            assert_eq!(
                &replay, original,
                "retry must preserve the same immutable publication"
            );
        } else {
            original = Some(replay);
        }
    }
    println!("assessment-publication-conformance: passed");
}
