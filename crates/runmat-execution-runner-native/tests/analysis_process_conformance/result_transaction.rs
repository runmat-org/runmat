//! A checked study publishes its meshes, evidence and results in one result
//! transaction through the result store's provider, never through the
//! process's installed provider. Reruns reuse the stored mesh, and a stale
//! mesh object under the same name is replaced.
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

    let first = run_study(label, source).await;
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
    reopen(label, &first).await;

    // A rerun reuses the stored mesh object unchanged.
    let second = run_study(label, source).await;
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
    reopen(label, &second).await;

    // A stale object from an earlier failed run is replaced.
    files.plant(&mesh_path, b"stale mesh object from an earlier run");
    let third = run_study(label, source).await;
    assert_eq!(
        files.read(&mesh_path),
        mesh_bytes,
        "{label}: the stale mesh was not replaced"
    );
    reopen(label, &third).await;
    runmat_runtime::analysis::replace_fea_runtime(previous).unwrap();
}

async fn run_study(label: &str, source: &Path) -> runmat_runtime::analysis::AnalysisStudyRunData {
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
        root: Some(FeaRootSelector::Study {
            id: "box_static".into(),
        }),
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
    case.publication
}

async fn reopen(label: &str, run: &runmat_runtime::analysis::AnalysisStudyRunData) {
    let run_id = run.steps[0].run_id.clone();
    let loaded = runmat_runtime::analysis::storage::with_artifact_access(false, |store| {
        store.load_run(&run_id)
    })
    .await
    .unwrap()
    .unwrap();
    assert!(
        loaded.is_some(),
        "{label}: committed run {run_id} does not reopen"
    );
}
