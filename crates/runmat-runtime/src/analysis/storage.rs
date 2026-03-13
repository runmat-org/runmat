use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use chrono::Utc;
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
            fs::create_dir_all(parent)
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
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|err| format!("failed to read run artifact: {err}"))?;
        let run = match serde_json::from_slice::<PersistedRunArtifact>(&bytes) {
            Ok(persisted) => persisted.run,
            Err(_) => serde_json::from_slice::<AnalysisRunResult>(&bytes)
                .map_err(|err| format!("failed to parse run artifact: {err}"))?,
        };
        Ok(Some(run))
    }

    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        let runs_dir = self.root.join("runs");
        if !runs_dir.exists() {
            return Ok(Vec::new());
        }
        let mut runs = Vec::new();
        for entry in
            fs::read_dir(&runs_dir).map_err(|err| format!("failed to scan artifacts: {err}"))?
        {
            let entry = entry.map_err(|err| format!("failed to read artifact entry: {err}"))?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let bytes =
                fs::read(&path).map_err(|err| format!("failed to read run artifact: {err}"))?;
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
    if run.electromagnetic_results.is_some()
        || run
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_EM_PLACEHOLDER")
    {
        "analysis.run_electromagnetic/v1".to_string()
    } else if run.nonlinear_results.is_some() {
        "analysis.run_nonlinear/v1".to_string()
    } else if run.transient_results.is_some() {
        "analysis.run_transient/v1".to_string()
    } else if run.modal_results.is_some() {
        "analysis.run_modal/v1".to_string()
    } else {
        "analysis.run_linear_static/v1".to_string()
    }
}

fn prune_filesystem_runs(root: &PathBuf) -> Result<(), String> {
    let max_runs = std::env::var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let max_runs_per_kind = std::env::var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS_PER_KIND")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if max_runs == 0 && max_runs_per_kind == 0 {
        return Ok(());
    }

    let runs_dir = root.join("runs");
    if !runs_dir.exists() {
        return Ok(());
    }
    let mut artifacts = Vec::new();
    for entry in
        fs::read_dir(&runs_dir).map_err(|err| format!("failed to scan artifacts: {err}"))?
    {
        let entry = entry.map_err(|err| format!("failed to read artifact entry: {err}"))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).map_err(|err| format!("failed to read artifact file: {err}"))?;
        let (op_version, run_id) = match serde_json::from_slice::<PersistedRunArtifact>(&bytes) {
            Ok(persisted) => (persisted.op_version, persisted.run.run_id),
            Err(_) => match serde_json::from_slice::<AnalysisRunResult>(&bytes) {
                Ok(run) => (run_operation_version(&run), run.run_id),
                Err(_) => continue,
            },
        };
        let modified = entry.metadata().and_then(|meta| meta.modified()).ok();
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
        let _ = fs::remove_file(path);
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
    let mode = std::env::var("RUNMAT_ANALYSIS_ARTIFACT_STORE")
        .unwrap_or_else(|_| "in_memory".to_string())
        .to_lowercase();
    if mode == "filesystem" {
        let root = std::env::var("RUNMAT_ANALYSIS_ARTIFACT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/runmat-analysis-store")
            });
        AnalysisArtifactStoreConfig::Filesystem { root }
    } else {
        AnalysisArtifactStoreConfig::InMemory
    }
}

fn atomic_write(path: &PathBuf, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    fs::write(&tmp, bytes).map_err(|err| format!("failed to write temp artifact file: {err}"))?;
    fs::rename(&tmp, path).map_err(|err| {
        let _ = fs::remove_file(&tmp);
        format!("failed to atomically replace run artifact: {err}")
    })
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
}
