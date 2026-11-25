//! A checked study publishes its meshes, evidence and results in one result
//! transaction through the result store's provider, never through the
//! process's installed provider. Reruns reuse the stored mesh, and a stale
//! mesh object under the same name is replaced. After every publication the
//! results reopen the way Desktop and the agent open them, outside any artifact
//! scope: the run document, the attempt list, the results query, field values,
//! the field scene, the `fea.results` object and a reassessment.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_execution_runner_native::{
    execute_checked_fea_locally, LocalFeaExecution, LocalFeaExecutionOptions,
};
use runmat_filesystem::{CapturedIo, FsProvider, MemoryFsProvider, SandboxFsProvider};
use runmat_runtime::analysis::{
    storage::{configure_artifact_store, AnalysisArtifactStoreConfig},
    FeaCheckStatus, FeaRootSelector, FeaRuntimeConfig,
};

#[path = "large_publication.rs"]
pub(super) mod large_publication;

/// A project server: capture and conditional publication go to an in-memory
/// authority. Its sandbox refuses direct workspace access.
struct FakeTransactionServer(Arc<MemoryFsProvider>);

#[async_trait::async_trait(?Send)]
impl CapturedIo for FakeTransactionServer {
    async fn lookup_workspace_publication(
        &self,
        request: &runmat_filesystem::workspace::WorkspacePublicationRequest,
    ) -> std::io::Result<Option<runmat_filesystem::workspace::WorkspacePublicationOutcome>> {
        self.0.lookup_workspace_publication(request).await
    }
    async fn capture_workspace(
        &self,
        request: runmat_filesystem::workspace::WorkspaceCaptureRequest,
    ) -> std::io::Result<runmat_filesystem::workspace::WorkspaceSnapshot> {
        self.0.capture_workspace(request).await
    }
    async fn publish_workspace(
        &self,
        request: runmat_filesystem::workspace::WorkspacePublicationRequest,
    ) -> std::io::Result<runmat_filesystem::workspace::WorkspacePublicationOutcome> {
        self.0.publish_workspace(request).await
    }
    async fn read_version(
        &self,
        _path: &Path,
        _version: &str,
        _size: u64,
        _snapshot: &str,
    ) -> std::io::Result<Vec<u8>> {
        Err(std::io::ErrorKind::Unsupported.into())
    }
    async fn publish_output(
        &self,
        _path: &Path,
        _data: &[u8],
        _expected: Option<&str>,
    ) -> std::io::Result<String> {
        Err(std::io::ErrorKind::Unsupported.into())
    }
    async fn remove_output(&self, _path: &Path, _expected: &str) -> std::io::Result<()> {
        Err(std::io::ErrorKind::Unsupported.into())
    }
}

/// Reads and writes of the result store under test.
trait ResultFiles {
    fn list(&self, directory: &Path) -> Vec<String>;
    fn read(&self, path: &Path) -> Vec<u8>;
    fn plant(&self, path: &Path, bytes: &[u8]);
}

struct ServerFiles(Arc<MemoryFsProvider>);
impl ResultFiles for ServerFiles {
    fn list(&self, directory: &Path) -> Vec<String> {
        let mut names: Vec<_> = futures::executor::block_on(self.0.read_dir(directory))
            .unwrap_or_default()
            .into_iter()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
    fn read(&self, path: &Path) -> Vec<u8> {
        self.0.read_project_path(path).unwrap()
    }
    fn plant(&self, path: &Path, bytes: &[u8]) {
        self.0.write_project_path(path, bytes).unwrap();
    }
}

struct DiskFiles;
impl ResultFiles for DiskFiles {
    fn list(&self, directory: &Path) -> Vec<String> {
        // A native coherent scope keeps its publication journal in `.artifacts`.
        let mut names: Vec<_> = std::fs::read_dir(directory)
            .map(|entries| {
                entries
                    .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                    .filter(|name| name != ".artifacts")
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }
    fn read(&self, path: &Path) -> Vec<u8> {
        std::fs::read(path).unwrap()
    }
    fn plant(&self, path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
    }
}

pub(super) async fn run() {
    let directory = tempfile::tempdir().unwrap();
    let project = directory.path().canonicalize().unwrap();
    std::fs::create_dir_all(project.join("geometry")).unwrap();
    std::fs::write(
        project.join("geometry/box.brep"),
        include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep"),
    )
    .unwrap();
    let source = project.join("study.fea");
    std::fs::write(
        &source,
        include_str!("../../../runmat-cli/tests/fixtures/typed_box_loaded_static.fea"),
    )
    .unwrap();
    std::env::set_var("RUNMAT_EXECUTION_STATE_DIR", project.join("native-state"));

    // A shared cloud project: the store's provider refuses direct access and
    // commits through the project server's transactions.
    let server = Arc::new(MemoryFsProvider::new());
    // The project folder exists on the server before the first run.
    server
        .create_dir_all(Path::new("/.artifacts"))
        .await
        .unwrap();
    let sandbox: Arc<dyn FsProvider> = Arc::new(
        SandboxFsProvider::new(project.join("sandbox-cache"))
            .unwrap()
            .with_captured_io(Arc::new(FakeTransactionServer(server.clone()))),
    );
    exercise(
        "transaction provider",
        &source,
        Some(sandbox),
        PathBuf::from("/.artifacts"),
        &ServerFiles(server),
    )
    .await;

    // A local project folder: the same publication writes directly.
    exercise(
        "local provider",
        &source,
        None,
        project.join("local-results"),
        &DiskFiles,
    )
    .await;
    runmat_runtime::analysis::storage::configure_artifact_store_from_env().unwrap();
}

async fn exercise(
    label: &str,
    source: &Path,
    store_provider: Option<Arc<dyn FsProvider>>,
    root: PathBuf,
    files: &dyn ResultFiles,
) {
    let previous = runmat_runtime::analysis::replace_fea_runtime(FeaRuntimeConfig {
        artifact_root: Some(root.clone()),
        study_artifact_root: Some(root.join("studies")),
        ..Default::default()
    })
    .unwrap();
    // The store captures the provider installed when it is configured. The
    // run itself keeps the process's native provider installed, as Desktop
    // keeps its pinned input snapshot installed.
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
    let meshes = root.join("meshes");

    let (first, checked) = run_study(label, source).await;
    let mut stored = files.list(&meshes);
    assert_eq!(
        stored.len(),
        2,
        "{label}: mesh and meshing evidence: {stored:?}"
    );
    let mesh_name = stored
        .iter()
        .find(|name| name.ends_with(".solver-mesh.cbor"))
        .cloned()
        .unwrap_or_else(|| panic!("{label}: no solver mesh in {stored:?}"));
    let mesh_path = meshes.join(&mesh_name);
    let mesh_bytes = files.read(&mesh_path);
    assert!(
        !files.list(&root.join("studies")).is_empty(),
        "{label}: study evidence was not committed"
    );
    // Result objects and the manifest sit beside meshes/ and studies/.
    let top = files.list(&root);
    assert!(
        top.iter().any(|name| name != "meshes" && name != "studies"),
        "{label}: result objects were not committed: {top:?}"
    );
    reopen(label, source, &checked, &first).await;

    // A rerun reuses the stored mesh object unchanged.
    let (second, checked) = run_study(label, source).await;
    // Meshing evidence records each run's own meshing (for example its
    // timing), so a rerun may add an evidence object; the mesh is reused.
    stored = files.list(&meshes);
    let meshes_stored = stored
        .iter()
        .filter(|name| name.ends_with(".solver-mesh.cbor"))
        .count();
    assert_eq!(meshes_stored, 1, "{label}: rerun added a mesh: {stored:?}");
    assert_eq!(
        files.read(&mesh_path),
        mesh_bytes,
        "{label}: rerun changed the mesh"
    );
    reopen(label, source, &checked, &second).await;

    // A stale object from an earlier failed run is replaced.
    files.plant(&mesh_path, b"stale mesh object from an earlier run");
    let (third, checked) = run_study(label, source).await;
    assert_eq!(
        files.read(&mesh_path),
        mesh_bytes,
        "{label}: the stale mesh was not replaced"
    );
    reopen(label, source, &checked, &third).await;

    // A sweep publishes one study run per case; each case reopens.
    let sweep = source.with_file_name("sweep.fea");
    std::fs::write(
        &sweep,
        sweep_source(&std::fs::read_to_string(source).unwrap()),
    )
    .unwrap();
    let cases = run_sweep(label, &sweep).await;
    assert_eq!(cases.len(), 2, "{label}: expected two sweep cases");
    for case in &cases {
        reopen_results(label, &sweep, case).await;
    }
    runmat_runtime::analysis::replace_fea_runtime(previous).unwrap();
}

async fn run_study(
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
    assert_eq!(
        checked.check_result().status,
        FeaCheckStatus::Valid,
        "{:?}",
        checked.check_result().diagnostics
    );
    let options = LocalFeaExecutionOptions {
        root: Some(study_root()),
        workers: Some(1),
        cpu_threads_per_worker: Some(
            runmat_analysis_fea::planning::resource::LocalThreadCount::new(1).unwrap(),
        ),
        ..Default::default()
    };
    let completed = execute_checked_fea_locally(&checked, options)
        .await
        .unwrap_or_else(|error| panic!("{label}: run failed: {error}"));
    let LocalFeaExecution::Study(case) = completed else {
        panic!("{label}: expected one study");
    };
    (case.publication, checked)
}

/// The loaded box with its load as a study input, swept over two loads.
fn sweep_source(study: &str) -> String {
    let swept = study
        .replace(
            "model loaded_box {\n",
            "model loaded_box {\n    input load : force { default = 1 kN }\n",
        )
        .replace("= [0, -1, 0.5] kN", "= [0, -input.load, 0]")
        .replace(
            "study box_static {\n",
            "study box_static {\n    input load : force { default = 1 kN }\n",
        )
        .replace(
            "model = model.loaded_box\n",
            "model = model.loaded_box(load = input.load)\n",
        );
    format!(
        "{swept}\nsweep load_cases over study.box_static {{\n    axis load {{\n        light = 0.5 kN\n        heavy = 1.5 kN\n    }}\n}}\n"
    )
}

async fn run_sweep(
    label: &str,
    source: &Path,
) -> Vec<runmat_runtime::analysis::AnalysisStudyRunData> {
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let checked = session
        .load_and_check_fea_source_path(source)
        .await
        .unwrap();
    assert_eq!(
        checked.check_result().status,
        FeaCheckStatus::Valid,
        "{label}: {:?}",
        checked.check_result().diagnostics
    );
    let options = LocalFeaExecutionOptions {
        root: Some(FeaRootSelector::Sweep {
            id: "load_cases".into(),
        }),
        workers: Some(1),
        cpu_threads_per_worker: Some(
            runmat_analysis_fea::planning::resource::LocalThreadCount::new(1).unwrap(),
        ),
        ..Default::default()
    };
    let completed = execute_checked_fea_locally(&checked, options)
        .await
        .unwrap_or_else(|error| panic!("{label}: sweep failed: {error}"));
    let LocalFeaExecution::Sweep { cases, .. } = completed else {
        panic!("{label}: expected a sweep");
    };
    cases
        .into_iter()
        .map(|case| {
            case.outcome
                .unwrap_or_else(|error| panic!("{label}: case {} failed: {error}", case.case_id))
                .publication
        })
        .collect()
}

fn study_root() -> FeaRootSelector {
    FeaRootSelector::Study {
        id: "box_static".into(),
    }
}

/// Open a published run the way Desktop and the agent do: synchronous calls
/// outside any artifact scope, then a reassessment that saves its evidence.
async fn reopen(
    label: &str,
    source: &Path,
    checked: &runmat_core::CheckedFeaSessionSource,
    run: &runmat_runtime::analysis::AnalysisStudyRunData,
) {
    reopen_results(label, source, run).await;
    reassess(label, checked, run).await;
}

async fn reopen_results(
    label: &str,
    source: &Path,
    run: &runmat_runtime::analysis::AnalysisStudyRunData,
) {
    use runmat_runtime::analysis::{
        analysis_document_study_result, analysis_field_page_by_run_id_op,
        analysis_results_by_run_id_op, field_scene, storage, AnalysisFieldRequestOptions,
        AnalysisFieldSelection, AnalysisResultsQuery,
    };
    use runmat_runtime::operations::OperationContext;
    let run_id = run.steps[0].run_id.clone();

    // The run document Desktop shows when a run finishes.
    let document =
        analysis_document_study_result(source.display().to_string(), run.clone(), Vec::new())
            .unwrap_or_err(label, "run document");
    assert!(
        !document.field_bindings.is_empty(),
        "{label}: the run document has no fields"
    );

    // The results panel: committed attempts and the attempt the id resolves to.
    let attempts = storage::run_attempts(&run_id).unwrap_or_err(label, "attempt list");
    let resolved = storage::resolve_run_attempt(&run_id).unwrap_or_err(label, "attempt");
    assert!(
        attempts.is_empty() || attempts.last() == Some(&resolved),
        "{label}: {run_id} resolves to {resolved}, not the latest of {attempts:?}"
    );
    assert!(
        storage::load_run_metadata(&run_id)
            .unwrap_or_err(label, "run metadata")
            .is_some(),
        "{label}: run metadata of {run_id} is missing"
    );

    // The results query, metadata only and with field values.
    let metadata = analysis_results_by_run_id_op(
        &run_id,
        AnalysisResultsQuery::metadata_only(),
        OperationContext::new(None, None),
    )
    .map_err(|error| error.message)
    .unwrap_or_err(label, "results metadata");
    let field_id = metadata
        .data
        .field_descriptors
        .iter()
        .find(|descriptor| descriptor.field_id.contains("displacement"))
        .or_else(|| metadata.data.field_descriptors.first())
        .unwrap_or_else(|| panic!("{label}: {run_id} has no fields"))
        .field_id
        .clone();
    let values = analysis_results_by_run_id_op(
        &run_id,
        AnalysisResultsQuery {
            include_field_values: true,
            ..AnalysisResultsQuery::metadata_only()
        },
        OperationContext::new(None, None),
    )
    .map_err(|error| error.message)
    .unwrap_or_err(label, "results with field values");
    assert!(
        values
            .data
            .fields
            .iter()
            .any(|field| field.field_id == field_id),
        "{label}: the results query returned no values for {field_id}"
    );

    // One field's values, as the field inspector pages them.
    let page = analysis_field_page_by_run_id_op(
        &run_id,
        AnalysisFieldSelection::new(field_id.clone(), None),
        AnalysisFieldRequestOptions::default(),
        OperationContext::new(None, None),
    )
    .map_err(|error| error.message)
    .unwrap_or_err(label, "field page");
    assert!(page.data.count > 0, "{label}: field {field_id} is empty");

    // The result view: render tiles, field values and the solver mesh.
    let (scene, _, _) = field_scene::build_scene_view(
        &run_id,
        Some(&field_id),
        "true",
        1.0,
        true,
        None,
        &Default::default(),
    )
    .map_err(|error| error.to_string())
    .unwrap_or_err(label, "field scene");
    assert!(
        !scene.chunks.is_empty(),
        "{label}: the field scene has no meshes"
    );

    // The `fea.results` object a script or the agent opens.
    runmat_runtime::call_builtin_async("fea.results", &[run_id.as_str().into()])
        .await
        .map_err(|error| error.to_string())
        .unwrap_or_err(label, "fea.results");
}

/// A reassessment of the saved study, which saves its own evidence.
async fn reassess(
    label: &str,
    checked: &runmat_core::CheckedFeaSessionSource,
    run: &runmat_runtime::analysis::AnalysisStudyRunData,
) {
    let manifest = run
        .result_manifest_artifact_path
        .clone()
        .unwrap_or_else(|| panic!("{label}: the run has no result manifest"));
    let definition: runmat_execution::Digest =
        serde_json::from_value(serde_json::Value::String(run.study_fingerprint.clone())).unwrap();
    let runmat_runtime::analysis::ResolvedFeaRoot::Study(study) =
        checked.resolve_root(Some(&study_root())).unwrap()
    else {
        panic!("{label}: expected a study root");
    };
    let reassessed = runmat_execution_runner_native::reassess_saved_fea_locally(
        checked,
        definition,
        Path::new(&manifest),
        &study.study,
        runmat_runtime::analysis::assessment::AssessmentEvaluationLimits::default(),
        64 * 1024 * 1024,
    )
    .await
    .unwrap_or_err(label, "reassessment");
    assert!(
        !reassessed.assessments.is_empty(),
        "{label}: the reassessment has no assessments"
    );
}

trait UnwrapOrErr<T> {
    fn unwrap_or_err(self, label: &str, what: &str) -> T;
}
impl<T, E: std::fmt::Display> UnwrapOrErr<T> for Result<T, E> {
    fn unwrap_or_err(self, label: &str, what: &str) -> T {
        self.unwrap_or_else(|error| panic!("{label}: opening the {what} failed: {error}"))
    }
}
