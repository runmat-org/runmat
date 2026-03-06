use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use chrono::Utc;

use super::contracts::{AnalysisArtifactRecord, AnalysisRunResult};

pub trait AnalysisArtifactStore: Send + Sync {
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String>;
    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String>;
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
            op_version: "analysis.run_linear_static/v1".to_string(),
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
        let bytes = serde_json::to_vec_pretty(run)
            .map_err(|err| format!("failed to encode run artifact: {err}"))?;
        atomic_write(&path, &bytes)?;

        Ok(AnalysisArtifactRecord {
            run_id: run.run_id.clone(),
            created_at: Utc::now().to_rfc3339(),
            op_version: "analysis.run_linear_static/v1".to_string(),
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
        let run = serde_json::from_slice::<AnalysisRunResult>(&bytes)
            .map_err(|err| format!("failed to parse run artifact: {err}"))?;
        Ok(Some(run))
    }
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
