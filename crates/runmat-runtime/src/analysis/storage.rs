use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

use chrono::Utc;
use runmat_filesystem::{AtomicCreateResult, DirEntry, FsFileType, FsMetadata};
use serde::{Deserialize, Serialize};

use runmat_analysis_fea::AnalysisRunIdentity;

use super::contracts::{AnalysisArtifactRecord, AnalysisRunResult};

const ARTIFACT_SCHEMA_VERSION: &str = "analysis_run_artifact/2026-09";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    runs: RwLock<HashMap<String, PersistedRunArtifact>>,
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
        validate_run_identity(run)?;
        let mut guard = self
            .runs
            .write()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        if let Some(existing) = guard.get(&run.run_id) {
            validate_persisted_run(existing, &run.run_id)?;
            if existing.run != *run {
                return Err("analysis run identity already contains different content".into());
            }
            return artifact_record(run, existing.created_at.clone());
        }
        let created_at = Utc::now().to_rfc3339();
        guard.insert(
            run.run_id.clone(),
            PersistedRunArtifact {
                schema_version: ARTIFACT_SCHEMA_VERSION.into(),
                created_at: created_at.clone(),
                op_version: run_operation_version(run)?,
                run: run.clone(),
            },
        );
        artifact_record(run, created_at)
    }

    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        validate_run_id_token(run_id)?;
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        guard
            .get(run_id)
            .map(|persisted| {
                validate_persisted_run(persisted, run_id)?;
                Ok(persisted.run.clone())
            })
            .transpose()
    }

    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        guard
            .iter()
            .map(|(run_id, persisted)| {
                validate_persisted_run(persisted, run_id)?;
                Ok(persisted.run.clone())
            })
            .collect()
    }
}

pub struct FilesystemAnalysisArtifactStore {
    root: PathBuf,
}

impl FilesystemAnalysisArtifactStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn run_path(&self, run_id: &str) -> Result<PathBuf, String> {
        validate_run_id_token(run_id)?;
        Ok(self.root.join("runs").join(format!("{run_id}.json")))
    }

    fn ensure_runs_directory(&self) -> Result<(), String> {
        ensure_owned_directory(&self.root, "analysis artifact root")?;
        let runs = self.root.join("runs");
        ensure_owned_directory(&runs, "analysis artifact runs directory")?;
        Ok(())
    }
}

impl AnalysisArtifactStore for FilesystemAnalysisArtifactStore {
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
        validate_run_identity(run)?;
        let path = self.run_path(&run.run_id)?;
        self.ensure_runs_directory()?;
        let op_version = run_operation_version(run)?;
        let persisted = PersistedRunArtifact {
            schema_version: ARTIFACT_SCHEMA_VERSION.to_string(),
            created_at: Utc::now().to_rfc3339(),
            op_version: op_version.clone(),
            run: run.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&persisted)
            .map_err(|err| format!("failed to encode run artifact: {err}"))?;
        match runmat_filesystem::atomic_create_new(&path, &bytes)
            .map_err(|error| format!("failed to publish immutable run artifact: {error}"))?
        {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing_bytes) => {
                let existing = decode_persisted_run(&existing_bytes, &run.run_id)?;
                if existing.run != *run {
                    return Err("analysis run identity already contains different content".into());
                }
                return artifact_record(run, existing.created_at);
            }
        }
        prune_filesystem_runs(&self.root)?;

        artifact_record(run, persisted.created_at)
    }

    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        let path = self.run_path(run_id)?;
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&self.root.join("runs"), "analysis artifact runs directory")?
        {
            return Ok(None);
        }
        match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {}
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}")),
        }
        let bytes = fs_read(&path).map_err(|err| format!("failed to read run artifact: {err}"))?;
        Ok(Some(decode_persisted_run(&bytes, run_id)?.run))
    }

    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        let runs_dir = self.root.join("runs");
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&runs_dir, "analysis artifact runs directory")?
        {
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
            if entry.file_type() != FsFileType::File {
                return Err(format!(
                    "analysis run artifact must be a regular file: {}",
                    path.display()
                ));
            }
            let run_id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| {
                    format!(
                        "analysis run artifact filename is invalid: {}",
                        path.display()
                    )
                })?;
            validate_run_id_token(run_id)?;
            let bytes =
                fs_read(&path).map_err(|err| format!("failed to read run artifact: {err}"))?;
            if let Ok(persisted) = decode_persisted_run(&bytes, run_id) {
                runs.push(persisted.run);
            }
        }
        Ok(runs)
    }
}

fn decode_persisted_run(
    bytes: &[u8],
    expected_run_id: &str,
) -> Result<PersistedRunArtifact, String> {
    validate_run_id_token(expected_run_id)?;
    let persisted = serde_json::from_slice::<PersistedRunArtifact>(bytes)
        .map_err(|error| format!("failed to parse run artifact: {error}"))?;
    validate_persisted_run(&persisted, expected_run_id)?;
    Ok(persisted)
}

fn validate_persisted_run(
    persisted: &PersistedRunArtifact,
    expected_run_id: &str,
) -> Result<(), String> {
    if persisted.schema_version != ARTIFACT_SCHEMA_VERSION {
        return Err(format!(
            "unsupported analysis run artifact schema `{}`",
            persisted.schema_version
        ));
    }
    if persisted.run.run_id != expected_run_id {
        return Err(format!(
            "analysis run artifact key `{expected_run_id}` does not match embedded run ID `{}`",
            persisted.run.run_id
        ));
    }
    let expected_op_version = run_operation_version(&persisted.run)?;
    if persisted.op_version != expected_op_version {
        return Err(format!(
            "analysis run artifact operation version `{}` does not match derived version `{expected_op_version}`",
            persisted.op_version
        ));
    }
    validate_run_identity(&persisted.run)?;
    Ok(())
}

pub(super) fn ensure_owned_directory(path: &Path, label: &str) -> Result<(), String> {
    match fs_symlink_metadata(path) {
        Ok(metadata) => return validate_owned_directory_metadata(metadata, path, label),
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(format!("failed to inspect {label}: {error}")),
    }
    fs_create_dir_all(path).map_err(|error| format!("failed to create {label}: {error}"))?;
    let metadata = fs_symlink_metadata(path)
        .map_err(|error| format!("failed to inspect created {label}: {error}"))?;
    validate_owned_directory_metadata(metadata, path, label)
}

pub(super) fn owned_directory_exists(path: &Path, label: &str) -> Result<bool, String> {
    match fs_symlink_metadata(path) {
        Ok(metadata) => {
            validate_owned_directory_metadata(metadata, path, label)?;
            Ok(true)
        }
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!("failed to inspect {label}: {error}")),
    }
}

fn validate_owned_directory_metadata(
    metadata: FsMetadata,
    path: &Path,
    label: &str,
) -> Result<(), String> {
    if metadata.file_type() == FsFileType::Directory {
        Ok(())
    } else {
        Err(format!(
            "{label} must be a real directory and may not be a symlink: {}",
            path.display()
        ))
    }
}

fn run_operation_version(run: &AnalysisRunResult) -> Result<String, String> {
    super::run_operation_for_identity(&run.run.identity)
        .map(|operation| operation.version().to_string())
        .ok_or_else(|| "analysis run identity has no registered runtime operation".to_string())
}

fn artifact_record(
    run: &AnalysisRunResult,
    created_at: String,
) -> Result<AnalysisArtifactRecord, String> {
    Ok(AnalysisArtifactRecord {
        run_id: run.run_id.clone(),
        created_at,
        identity: run.run.identity.clone(),
        op_version: run_operation_version(run)?,
        field_ids: super::analysis_run_field_ids(run),
    })
}

fn validate_run_identity(run: &AnalysisRunResult) -> Result<(), String> {
    validate_run_id_token(&run.run_id)?;
    let identity = &run.run.identity;
    super::run_operation_for_identity(identity)
        .ok_or_else(|| "analysis run identity has no registered runtime operation".to_string())?;
    let (expected_publishable, expected_publication_reasons) =
        super::result_policy::evaluate_production_publication(&run.run, run.numerically_acceptable);
    if run.production_publishable != expected_publishable {
        return Err(
            "production publication must require both numerical acceptance and a production-qualified formulation"
                .to_string(),
        );
    }
    if run.production_publication_reasons != expected_publication_reasons {
        return Err(
            "production publication reasons do not match the failed publication gates".to_string(),
        );
    }
    let expected_status =
        super::result_policy::classify_run_status(run.result_quality, run.numerically_acceptable);
    if run.run_status != expected_status {
        return Err(format!(
            "analysis run status {:?} does not match numerical outcome {:?}",
            run.run_status, expected_status
        ));
    }
    if !run.provenance.fallback_events.is_empty() && !identity.permits_fallback() {
        return Err(
            "analysis run identity forbids the execution fallbacks recorded by provenance"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_run_id_token(run_id: &str) -> Result<(), String> {
    const MAX_RUN_ID_BYTES: usize = 128;
    if run_id.is_empty()
        || run_id.len() > MAX_RUN_ID_BYTES
        || !run_id.is_ascii()
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(format!(
            "analysis run ID must be a 1-{MAX_RUN_ID_BYTES} byte ASCII token containing only letters, digits, `_`, or `-`"
        ));
    }
    Ok(())
}

fn prune_filesystem_runs(root: &Path) -> Result<(), String> {
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
    ensure_owned_directory(root, "analysis artifact root")?;
    ensure_owned_directory(&runs_dir, "analysis artifact runs directory")?;
    let mut artifacts = Vec::new();
    for entry in fs_read_dir(&runs_dir).map_err(|err| format!("failed to scan artifacts: {err}"))? {
        let path = entry.path().to_path_buf();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        if entry.file_type() != FsFileType::File {
            return Err(format!(
                "analysis run artifact must be a regular file: {}",
                path.display()
            ));
        }
        let run_id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| {
                format!(
                    "analysis run artifact filename is invalid: {}",
                    path.display()
                )
            })?;
        let bytes = fs_read(&path).map_err(|err| format!("failed to read artifact file: {err}"))?;
        let Ok(persisted) = decode_persisted_run(&bytes, run_id) else {
            continue;
        };
        let identity = persisted.run.run.identity;
        let modified = fs_modified(&path).ok().flatten();
        artifacts.push((path, identity, modified));
    }
    artifacts.sort_by(|a, b| b.2.cmp(&a.2));

    let mut to_remove = Vec::new();
    if max_runs_per_kind > 0 {
        let mut per_kind_counts: HashMap<AnalysisRunIdentity, usize> = HashMap::new();
        for (path, identity, _modified) in &artifacts {
            let count = per_kind_counts.entry(identity.clone()).or_default();
            *count += 1;
            if *count > max_runs_per_kind {
                to_remove.push(path.clone());
            }
        }
    }
    if max_runs > 0 {
        for (index, (path, _identity, _modified)) in artifacts.iter().enumerate() {
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

fn fs_create_dir_all(path: impl Into<PathBuf>) -> std::io::Result<()> {
    runmat_filesystem::create_dir_all(path.into())
}

fn fs_read(path: impl Into<PathBuf>) -> std::io::Result<Vec<u8>> {
    runmat_filesystem::read(path.into())
}

fn fs_remove_file(path: impl Into<PathBuf>) -> std::io::Result<()> {
    match runmat_filesystem::remove_file(path.into()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn fs_read_dir(path: impl Into<PathBuf>) -> std::io::Result<Vec<DirEntry>> {
    runmat_filesystem::read_dir(path.into())
}

fn fs_symlink_metadata(path: impl Into<PathBuf>) -> std::io::Result<FsMetadata> {
    runmat_filesystem::symlink_metadata(path.into())
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

#[cfg(test)]
pub(crate) fn artifact_store_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
