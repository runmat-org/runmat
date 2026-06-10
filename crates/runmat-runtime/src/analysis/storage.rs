use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use chrono::Utc;
use runmat_filesystem::{DirEntry, FsFileType};
use serde::{Deserialize, Serialize};

use super::contracts::{AnalysisArtifactRecord, AnalysisRunResult};

const ARTIFACT_SCHEMA_VERSION: &str = "analysis_run_artifact/v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PersistedRunArtifact {
    schema_version: String,
    created_at: String,
    op_version: String,
    run: AnalysisRunResult,
}

pub trait AnalysisArtifactStore: Send + Sync {
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String>;
    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String>;
    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisArtifactStoreConfig {
    InMemory,
    Filesystem { root: PathBuf },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnalysisArtifactRetentionConfig {
    pub max_runs: Option<usize>,
    pub max_runs_per_kind: Option<usize>,
}

pub struct InMemoryAnalysisArtifactStore {
    runs: RwLock<HashMap<String, AnalysisRunResult>>,
}

impl InMemoryAnalysisArtifactStore {
    pub fn new() -> Self {
        Self {
            runs: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryAnalysisArtifactStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisArtifactStore for InMemoryAnalysisArtifactStore {
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
        let mut guard = self
            .runs
            .write()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        guard.insert(run.run_id.clone(), run.clone());
        Ok(AnalysisArtifactRecord {
            run_id: run.run_id.clone(),
            created_at: Utc::now().to_rfc3339(),
            op_version: run_operation_version(run),
            field_ids: vec![
                run.run.displacement_field.field_id.clone(),
                run.run.von_mises_field.field_id.clone(),
            ],
        })
    }

    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        Ok(guard.get(run_id).cloned())
    }

    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        Ok(guard.values().cloned().collect())
    }
}

pub struct FilesystemAnalysisArtifactStore {
    root: PathBuf,
}

impl FilesystemAnalysisArtifactStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn run_path(&self, run_id: &str) -> PathBuf {
        self.root.join("runs").join(format!("{run_id}.json"))
    }
}

impl AnalysisArtifactStore for FilesystemAnalysisArtifactStore {
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
        let path = self.run_path(&run.run_id);
        if let Some(parent) = path.parent() {
            fs_create_dir_all(parent)
                .map_err(|err| format!("failed to create artifact directory: {err}"))?;
        }
        let op_version = run_operation_version(run);
        let persisted = PersistedRunArtifact {
            schema_version: ARTIFACT_SCHEMA_VERSION.to_string(),
            created_at: Utc::now().to_rfc3339(),
            op_version: op_version.clone(),
            run: run.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&persisted)
            .map_err(|err| format!("failed to encode run artifact: {err}"))?;
        atomic_write(&path, &bytes)?;
        prune_filesystem_runs(&self.root)?;

        Ok(AnalysisArtifactRecord {
            run_id: run.run_id.clone(),
            created_at: Utc::now().to_rfc3339(),
            op_version,
            field_ids: vec![
                run.run.displacement_field.field_id.clone(),
                run.run.von_mises_field.field_id.clone(),
            ],
        })
    }

    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        let path = self.run_path(run_id);
        if !fs_exists(&path).map_err(|err| format!("failed to inspect run artifact: {err}"))? {
            return Ok(None);
        }
        let bytes = fs_read(&path).map_err(|err| format!("failed to read run artifact: {err}"))?;
        let run = match serde_json::from_slice::<PersistedRunArtifact>(&bytes) {
            Ok(persisted) => persisted.run,
            Err(_) => serde_json::from_slice::<AnalysisRunResult>(&bytes)
                .map_err(|err| format!("failed to parse run artifact: {err}"))?,
        };
        Ok(Some(run))
    }

    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        let runs_dir = self.root.join("runs");
        if !fs_exists(&runs_dir).map_err(|err| format!("failed to inspect artifacts: {err}"))? {
            return Ok(Vec::new());
        }
        let mut runs = Vec::new();
        for entry in
            fs_read_dir(&runs_dir).map_err(|err| format!("failed to scan artifacts: {err}"))?
        {
            let path = entry.path().to_path_buf();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let bytes =
                fs_read(&path).map_err(|err| format!("failed to read run artifact: {err}"))?;
            let parsed = match serde_json::from_slice::<PersistedRunArtifact>(&bytes) {
                Ok(persisted) => Some(persisted.run),
                Err(_) => serde_json::from_slice::<AnalysisRunResult>(&bytes).ok(),
            };
            if let Some(run) = parsed {
                runs.push(run);
            }
        }
        Ok(runs)
    }
}

fn run_operation_version(run: &AnalysisRunResult) -> String {
    if run
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_PLACEHOLDER")
    {
        "fea.run_acoustic/v1".to_string()
    } else if run.electromagnetic_results.is_some()
        || run
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_EM_PLACEHOLDER")
    {
        "fea.run_electromagnetic/v1".to_string()
    } else if run
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CHT_COUPLING")
    {
        "fea.run_cht/v1".to_string()
    } else if run
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_FSI_COUPLING")
    {
        "fea.run_fsi/v1".to_string()
    } else if run
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CFD_FLOW")
    {
        "fea.run_cfd/v1".to_string()
    } else if run.nonlinear_results.is_some() {
        "fea.run_nonlinear/v1".to_string()
    } else if run.transient_results.is_some() {
        "fea.run_transient/v1".to_string()
    } else if run.modal_results.is_some() {
        "fea.run_modal/v1".to_string()
    } else {
        "fea.run_linear_static/v1".to_string()
    }
}

fn prune_filesystem_runs(root: &PathBuf) -> Result<(), String> {
    let retention = current_retention_config();
    let max_runs = retention.max_runs.unwrap_or_else(|| {
        std::env::var("RUNMAT_FEA_ARTIFACT_MAX_RUNS")
            .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS"))
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
    });
    let max_runs_per_kind = retention.max_runs_per_kind.unwrap_or_else(|| {
        std::env::var("RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND")
            .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS_PER_KIND"))
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
    });
    if max_runs == 0 && max_runs_per_kind == 0 {
        return Ok(());
    }

    let runs_dir = root.join("runs");
    if !fs_exists(&runs_dir).map_err(|err| format!("failed to inspect artifacts: {err}"))? {
        return Ok(());
    }
    let mut artifacts = Vec::new();
    for entry in fs_read_dir(&runs_dir).map_err(|err| format!("failed to scan artifacts: {err}"))? {
        let path = entry.path().to_path_buf();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let bytes = fs_read(&path).map_err(|err| format!("failed to read artifact file: {err}"))?;
        let (op_version, run_id) = match serde_json::from_slice::<PersistedRunArtifact>(&bytes) {
            Ok(persisted) => (persisted.op_version, persisted.run.run_id),
            Err(_) => match serde_json::from_slice::<AnalysisRunResult>(&bytes) {
                Ok(run) => (run_operation_version(&run), run.run_id),
                Err(_) => continue,
            },
        };
        let modified = fs_modified(&path).ok().flatten();
        artifacts.push((path, op_version, run_id, modified));
    }
    artifacts.sort_by(|a, b| b.3.cmp(&a.3));

    let mut to_remove = Vec::new();
    if max_runs_per_kind > 0 {
        let mut per_kind_counts: HashMap<String, usize> = HashMap::new();
        for (path, op_version, _run_id, _modified) in &artifacts {
            let count = per_kind_counts.entry(op_version.clone()).or_default();
            *count += 1;
            if *count > max_runs_per_kind {
                to_remove.push(path.clone());
            }
        }
    }
    if max_runs > 0 {
        for (index, (path, _op_version, _run_id, _modified)) in artifacts.iter().enumerate() {
            if index >= max_runs {
                to_remove.push(path.clone());
            }
        }
    }
    to_remove.sort();
    to_remove.dedup();
    for path in to_remove {
        let _ = fs_remove_file(path);
    }
    Ok(())
}

fn global_store() -> &'static RwLock<Arc<dyn AnalysisArtifactStore>> {
    static STORE: OnceLock<RwLock<Arc<dyn AnalysisArtifactStore>>> = OnceLock::new();
    STORE.get_or_init(|| {
        let default = store_from_config(config_from_env());
        RwLock::new(default)
    })
}

fn retention_config() -> &'static RwLock<AnalysisArtifactRetentionConfig> {
    static CONFIG: OnceLock<RwLock<AnalysisArtifactRetentionConfig>> = OnceLock::new();
    CONFIG.get_or_init(|| RwLock::new(AnalysisArtifactRetentionConfig::default()))
}

fn current_retention_config() -> AnalysisArtifactRetentionConfig {
    retention_config()
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_run_id() -> String {
    let seq = NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed);
    format!("run_{}_{}", Utc::now().timestamp_millis(), seq)
}

pub fn persist_run_result(run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.persist_run(run)
}

pub fn load_run_result(run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_run(run_id)
}

pub fn list_run_results() -> Result<Vec<AnalysisRunResult>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.list_runs()
}

pub fn configure_artifact_store(config: AnalysisArtifactStoreConfig) -> Result<(), String> {
    let mut guard = global_store()
        .write()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    *guard = store_from_config(config);
    Ok(())
}

pub fn configure_artifact_retention(config: AnalysisArtifactRetentionConfig) -> Result<(), String> {
    let mut guard = retention_config()
        .write()
        .map_err(|_| "analysis artifact retention config lock poisoned".to_string())?;
    *guard = config;
    Ok(())
}

pub fn configure_artifact_store_from_env() -> Result<(), String> {
    configure_artifact_store(config_from_env())
}

fn store_from_config(config: AnalysisArtifactStoreConfig) -> Arc<dyn AnalysisArtifactStore> {
    match config {
        AnalysisArtifactStoreConfig::InMemory => Arc::new(InMemoryAnalysisArtifactStore::new()),
        AnalysisArtifactStoreConfig::Filesystem { root } => {
            Arc::new(FilesystemAnalysisArtifactStore::new(root))
        }
    }
}

fn config_from_env() -> AnalysisArtifactStoreConfig {
    let mode = std::env::var("RUNMAT_FEA_ARTIFACT_STORE")
        .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_STORE"))
        .unwrap_or_else(|_| "filesystem".to_string())
        .to_lowercase();
    if mode == "filesystem" {
        let root = std::env::var("RUNMAT_FEA_ARTIFACT_ROOT")
            .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_ROOT"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_filesystem_artifact_root());
        AnalysisArtifactStoreConfig::Filesystem { root }
    } else {
        AnalysisArtifactStoreConfig::InMemory
    }
}

pub fn default_filesystem_artifact_root() -> PathBuf {
    PathBuf::from("artifacts")
}

fn atomic_write(path: &PathBuf, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    fs_write(&tmp, bytes).map_err(|err| format!("failed to write temp artifact file: {err}"))?;
    fs_rename(&tmp, path).map_err(|err| {
        let _ = fs_remove_file(&tmp);
        format!("failed to atomically replace run artifact: {err}")
    })
}

fn fs_create_dir_all(path: impl Into<PathBuf>) -> std::io::Result<()> {
    runmat_filesystem::create_dir_all(path.into())
}

fn fs_read(path: impl Into<PathBuf>) -> std::io::Result<Vec<u8>> {
    runmat_filesystem::read(path.into())
}

fn fs_write(path: impl Into<PathBuf>, bytes: &[u8]) -> std::io::Result<()> {
    runmat_filesystem::write(path.into(), bytes)
}

fn fs_remove_file(path: impl Into<PathBuf>) -> std::io::Result<()> {
    match runmat_filesystem::remove_file(path.into()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn fs_rename(from: impl Into<PathBuf>, to: impl Into<PathBuf>) -> std::io::Result<()> {
    runmat_filesystem::rename(from.into(), to.into())
}

fn fs_read_dir(path: impl Into<PathBuf>) -> std::io::Result<Vec<DirEntry>> {
    runmat_filesystem::read_dir(path.into())
}

fn fs_exists(path: impl Into<PathBuf>) -> std::io::Result<bool> {
    match runmat_filesystem::metadata(path.into()) {
        Ok(metadata) => Ok(matches!(
            metadata.file_type(),
            FsFileType::Directory | FsFileType::File | FsFileType::Symlink | FsFileType::Other
        )),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

fn fs_modified(path: impl Into<PathBuf>) -> std::io::Result<Option<std::time::SystemTime>> {
    runmat_filesystem::metadata(path.into()).map(|metadata| metadata.modified())
}

#[cfg(test)]
pub fn set_artifact_store_for_tests(store: Arc<dyn AnalysisArtifactStore>) {
    let mut guard = global_store()
        .write()
        .expect("analysis artifact store lock poisoned");
    *guard = store;
}

#[cfg(test)]
pub fn reset_artifact_store_for_tests() {
    let mut guard = global_store()
        .write()
        .expect("analysis artifact store lock poisoned");
    *guard = Arc::new(InMemoryAnalysisArtifactStore::new());
    *retention_config()
        .write()
        .expect("analysis artifact retention config lock poisoned") =
        AnalysisArtifactRetentionConfig::default();
}
