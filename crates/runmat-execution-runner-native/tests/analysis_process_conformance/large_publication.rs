//! A study whose staged publication needs more than 128 MiB publishes and reopens through both a
//! transaction-only (cloud project) provider and a local project folder. With no host cap, the
//! publication stages within the run's admitted memory. A host cap below the requirement refuses
//! the save with a plain sentence that names the size and what to change.
use super::*;

/// Staging this study's publication needs more than this host cap.
const SMALL_HOST_CAP_BYTES: u64 = 128 * 1024 * 1024;

pub(in super::super) async fn run() {
    let directory = tempfile::tempdir().unwrap();
    let project = directory.path().canonicalize().unwrap();
    std::fs::create_dir_all(project.join("geometry")).unwrap();
    std::fs::write(
        project.join("geometry/box.brep"),
        include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep"),
    )
    .unwrap();
    let source = project.join("large.fea");
    std::fs::write(&source, large_study_source()).unwrap();
    std::env::set_var("RUNMAT_EXECUTION_STATE_DIR", project.join("native-state"));

    let server = Arc::new(MemoryFsProvider::new());
    server
        .create_dir_all(Path::new("/.artifacts"))
        .await
        .unwrap();
    let sandbox: Arc<dyn FsProvider> = Arc::new(
        SandboxFsProvider::new(project.join("sandbox-cache"))
            .unwrap()
            .with_captured_io(Arc::new(FakeTransactionServer(server.clone()))),
    );
    publish_large(
        "large transaction provider",
        &source,
        Some(sandbox),
        PathBuf::from("/.artifacts"),
        &ServerFiles(server),
        true,
    )
    .await;
    publish_large(
        "large local provider",
        &source,
        None,
        project.join("local-results"),
        &DiskFiles,
        false,
    )
    .await;
    runmat_runtime::analysis::storage::configure_artifact_store_from_env().unwrap();
}

/// The impulsively loaded box as a transient study with 200 saved time steps: each step adds a
/// displacement field, so its publication stages more than `SMALL_HOST_CAP_BYTES` while each
/// step's solve stays small.
fn large_study_source() -> String {
    let study =
        include_str!("../../../runmat-cli/tests/fixtures/typed_box_transient_assessments.fea");
    let custom = study
        .find("    assessment custom_deflection {")
        .expect("the fixture's custom assessment");
    let end = custom + study[custom..].find("    }\n").unwrap() + "    }\n".len();
    format!("{}{}", &study[..custom], &study[end..])
        .replace("import function verify_peak from \"./verify_peak.m\"\n", "")
        .replace("target_size = 0.75 m", "target_size = 0.1 m")
        .replace("time_step = 0.01 s", "time_step = 0.0005 s")
        .replace(
            "    output spatial_peak =",
            "    output displacement = analysis.response.field.node.displacement\n    output spatial_peak =",
        )
}

async fn publish_large(
    label: &str,
    source: &Path,
    store_provider: Option<Arc<dyn FsProvider>>,
    root: PathBuf,
    files: &dyn ResultFiles,
    check_cap_refusal: bool,
) {
    let config = |cap| FeaRuntimeConfig {
        artifact_root: Some(root.clone()),
        study_artifact_root: Some(root.join("studies")),
        publication_staging_cap_bytes: cap,
    };
    let previous = runmat_runtime::analysis::replace_fea_runtime(config(None)).unwrap();
    match store_provider {
        Some(provider) => {
            let _store = runmat_filesystem::replace_provider(provider);
            configure_artifact_store(AnalysisArtifactStoreConfig::Filesystem {
                root: root.clone(),
            })
            .unwrap();
        }
        None => {
            configure_artifact_store(AnalysisArtifactStoreConfig::Filesystem { root: root.clone() })
                .unwrap()
        }
    }
    if check_cap_refusal {
        runmat_runtime::analysis::replace_fea_runtime(config(Some(SMALL_HOST_CAP_BYTES))).unwrap();
        let started = std::time::Instant::now();
        let error = try_run_study(source)
            .await
            .err()
            .unwrap_or_else(|| panic!("{label}: the study fits a 128 MiB cap; refine its mesh"));
        eprintln!(
            "{label}: capped run ended in {:?}: {error}",
            started.elapsed()
        );
        // The outcome stays structured: a capacity refusal with its own code.
        let envelope = error
            .operation_error()
            .unwrap_or_else(|| panic!("{label}: the refusal is not an operation error: {error}"));
        assert_eq!(
            envelope.error_type,
            runmat_runtime::operations::OperationErrorType::Capacity,
            "{label}: {error}"
        );
        assert_eq!(
            envelope.error_code, "RM.FEA.RUN_STUDY.PUBLICATION_RESOURCE_LIMIT",
            "{label}: {error}"
        );
        assert_plain_capacity_refusal(label, &envelope.message);
        runmat_runtime::analysis::replace_fea_runtime(config(None)).unwrap();
    }
    let started = std::time::Instant::now();
    let (run, checked) = run_large_study(label, source).await;
    eprintln!("{label}: published in {:?}", started.elapsed());
    let stored = files.list(&root.join("meshes"));
    assert!(
        stored
            .iter()
            .any(|name| name.ends_with(".solver-mesh.cbor")),
        "{label}: no solver mesh in {stored:?}"
    );
    assert!(
        !files.list(&root.join("studies")).is_empty(),
        "{label}: study evidence was not committed"
    );
    reopen(label, source, &checked, &run).await;
    runmat_runtime::analysis::replace_fea_runtime(previous).unwrap();
}

/// The study's worker uses many threads: its factorization is the slowest step of this test.
fn large_options() -> LocalFeaExecutionOptions {
    LocalFeaExecutionOptions {
        root: Some(FeaRootSelector::Study {
            id: "structural_transient".into(),
        }),
        workers: Some(1),
        cpu_threads_per_worker: Some(
            runmat_analysis_fea::planning::resource::LocalThreadCount::new(16).unwrap(),
        ),
        ..Default::default()
    }
}

async fn run_large_study(
    label: &str,
    source: &Path,
) -> (
    runmat_runtime::analysis::AnalysisStudyRunData,
    runmat_core::CheckedFeaSessionSource,
) {
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let checked = session
        .load_and_check_fea_source_path(source)
        .await
        .unwrap();
    let completed = execute_checked_fea_locally(&checked, large_options())
        .await
        .unwrap_or_else(|error| panic!("{label}: run failed: {error}"));
    let LocalFeaExecution::Study(case) = completed else {
        panic!("{label}: expected one study");
    };
    (case.publication, checked)
}

async fn try_run_study(
    source: &Path,
) -> Result<(), runmat_execution_runner_native::FeaExecutionError> {
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let checked = session
        .load_and_check_fea_source_path(source)
        .await
        .unwrap();
    execute_checked_fea_locally(&checked, large_options())
        .await
        .map(|_| ())
}

/// The refusal is a sentence a user can act on, never a serialized outcome.
fn assert_plain_capacity_refusal(label: &str, error: &str) {
    assert!(
        !error.contains("\"outcome\"") && !error.contains('{'),
        "{label}: the refusal carries a serialized outcome: {error}"
    );
    for expected in [
        "Saving this study's results needs",
        "publication staging cap",
        "coarser mesh",
        "run the study again",
    ] {
        assert!(
            error.contains(expected),
            "{label}: the refusal does not say `{expected}`: {error}"
        );
    }
}
