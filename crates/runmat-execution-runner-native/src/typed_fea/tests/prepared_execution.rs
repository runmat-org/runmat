//! Prepared handoff metadata are fenced before any geometry/solver execution.
use super::prepared_validation::validate_prepared_execution;
use super::*;

const SOURCE: &str = r#"
geometry part {
 construct { feature body = box(width = 1 m, depth = 1 m, height = 1 m); result = feature.body }
}
mesh volume on geometry.part { tetrahedra { order = linear; target_size = 1 m; maximum_deviation = 0.02 m } }
model structure {}
study static {
 geometry = geometry.part; mesh = mesh.volume; model = model.structure
 analysis solve { structural static {} }
 output displacement = analysis.solve.field.node.displacement
}
"#;

#[tokio::test]
async fn prepared_execution_rejects_metadata_and_envelope_substitution_before_execution() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("study.fea");
    std::fs::write(&path, SOURCE).unwrap();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let checked = session.load_and_check_fea_source_path(&path).await.unwrap();
    let ResolvedFeaRoot::Study(resolved) = checked
        .resolve_root(Some(&FeaRootSelector::Study {
            id: "static".into(),
        }))
        .unwrap()
    else {
        panic!("expected exact authored study")
    };
    let options = LocalFeaExecutionOptions::default();
    let mut prepared = prepare_checked_fea_study(
        &checked,
        resolved,
        &[],
        "prepared-execution-substitution-test",
        &options,
    )
    .await
    .unwrap();
    validate_prepared_execution(&prepared, &options).unwrap();

    let current_threads = prepared.root.local_execution.thread_count().get();
    let changed_options = LocalFeaExecutionOptions {
        cpu_threads_per_worker: Some(
            runmat_analysis_fea::planning::resource::LocalThreadCount::new(
                if current_threads == 1 { 2 } else { 1 },
            )
            .unwrap(),
        ),
        ..options.clone()
    };
    assert!(validate_prepared_execution(&prepared, &changed_options)
        .unwrap_err()
        .contains("threads"));

    let workers = prepared.workers;
    prepared.workers = 0;
    assert!(validate_prepared_execution(&prepared, &options)
        .unwrap_err()
        .contains("worker"));
    prepared.workers = workers;
    let changed_options = LocalFeaExecutionOptions {
        workers: Some(0),
        ..options.clone()
    };
    assert!(validate_prepared_execution(&prepared, &changed_options).is_err());
    let changed_options = LocalFeaExecutionOptions {
        backend: runmat_analysis_fea::ComputeBackend::Gpu,
        ..options.clone()
    };
    assert!(validate_prepared_execution(&prepared, &changed_options)
        .unwrap_err()
        .contains("backend"));
    let changed_options = LocalFeaExecutionOptions {
        accelerators: vec![runmat_execution::resource::AcceleratorRequest {
            class: runmat_execution::resource::AcceleratorClass::new("unadmitted").unwrap(),
            count: 1,
            minimum_allocation_bytes: 1024,
            provider: None,
            required_features: Default::default(),
        }],
        ..options.clone()
    };
    assert!(validate_prepared_execution(&prepared, &changed_options)
        .unwrap_err()
        .contains("accelerator"));

    let revision = prepared.revision.clone();
    prepared.revision = runmat_execution::ProgramRevision::new(
        runmat_execution::Digest::sha256(b"foreign-prepared-graph"),
        *revision.source_digest(),
        revision.environment(),
    )
    .unwrap();
    assert!(validate_prepared_execution(&prepared, &options)
        .unwrap_err()
        .contains("revision"));
    prepared.revision = revision;
    prepared.realized.study_id.push_str("_foreign");
    let error = execute_prepared_fea_study_locally(prepared, &options)
        .await
        .unwrap_err();
    assert!(error.contains("original root study"), "{error}");
}
