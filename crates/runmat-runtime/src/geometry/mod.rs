use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

use self::capture::DEFAULT_SVG_CAPTURE_ADAPTER;
use chrono::Utc;
use runmat_geometry_core::{EntityKind, EntityRef, GeometryAsset, Region};
use runmat_geometry_io::{
    import::GeometryImportError, import_geometry, GeometryFormat, GeometryImportOptions,
};
use runmat_geometry_ops::{compute_stats, find_region, GeometryStats, QueryError};
use runmat_meshing_core::{
    prepare_geometry_for_analysis, MeshingOptions, MeshingPrepResult, MeshingProfile,
};
use serde::{Deserialize, Serialize};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};
use crate::{build_runtime_error, BuiltinResult};

const GEOMETRY_INSPECT_OPERATION: &str = "geometry.inspect";
const GEOMETRY_INSPECT_OP_VERSION: &str = "geometry.inspect/v1";
const GEOMETRY_LOAD_OPERATION: &str = "geometry.load";
const GEOMETRY_LOAD_OP_VERSION: &str = "geometry.load/v1";
const GEOMETRY_COMPUTE_STATS_OPERATION: &str = "geometry.compute_stats";
const GEOMETRY_COMPUTE_STATS_OP_VERSION: &str = "geometry.compute_stats/v1";
const GEOMETRY_LIST_REGIONS_OPERATION: &str = "geometry.list_regions";
const GEOMETRY_LIST_REGIONS_OP_VERSION: &str = "geometry.list_regions/v1";
const GEOMETRY_QUERY_ENTITIES_OPERATION: &str = "geometry.query_entities";
const GEOMETRY_QUERY_ENTITIES_OP_VERSION: &str = "geometry.query_entities/v1";
const GEOMETRY_CAPTURE_VIEW_OPERATION: &str = "geometry.capture_view";
const GEOMETRY_CAPTURE_VIEW_OP_VERSION: &str = "geometry.capture_view/v1";
const GEOMETRY_PREP_FOR_ANALYSIS_OPERATION: &str = "geometry.prep_for_analysis";
const GEOMETRY_PREP_FOR_ANALYSIS_OP_VERSION: &str = "geometry.prep_for_analysis/v1";
const GEOMETRY_PREP_ARTIFACT_HEALTH_OPERATION: &str = "geometry.prep_artifact_health";
const GEOMETRY_PREP_ARTIFACT_HEALTH_OP_VERSION: &str = "geometry.prep_artifact_health/v1";
const DEFAULT_QUERY_LIMIT: usize = 2048;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryInspectResult {
    pub format: String,
    pub byte_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryRegionsResult {
    pub regions: Vec<Region>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryEntityQuery {
    pub region_id: Option<String>,
    pub mesh_id: Option<String>,
    pub entity_kind: EntityKind,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryEntityQueryResult {
    pub entities: Vec<EntityRef>,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryCaptureViewSpec {
    pub format: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryCaptureViewResult {
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryPrepProfile {
    SurfaceOnly,
    AnalysisReady,
    AdaptiveRefine,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryPrepForAnalysisSpec {
    pub profile: GeometryPrepProfile,
    pub target_element_budget: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryPrepForAnalysisResult {
    pub prep_artifact_id: String,
    pub prep: MeshingPrepResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredGeometryPrepArtifact {
    pub prep_artifact_id: String,
    pub schema_version: String,
    pub created_at: String,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub prep: MeshingPrepResult,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepArtifactMetrics {
    pub created_count: u64,
    pub loaded_count: u64,
    pub pruned_count: u64,
    pub stale_reject_count: u64,
    pub mismatch_reject_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryPrepArtifactHealthQuery {
    pub include_per_geometry: bool,
}

impl Default for GeometryPrepArtifactHealthQuery {
    fn default() -> Self {
        Self {
            include_per_geometry: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryPrepArtifactHealthEntry {
    pub geometry_id: String,
    pub latest_revision: u32,
    pub artifact_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryPrepArtifactHealthResult {
    pub schema_version: String,
    pub current_artifact_count: usize,
    pub age_p50_seconds: Option<f64>,
    pub age_p95_seconds: Option<f64>,
    pub metrics: PrepArtifactMetrics,
    pub per_geometry: Vec<GeometryPrepArtifactHealthEntry>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeometryPrepArtifactConfig {
    pub artifact_root: Option<PathBuf>,
    pub max_artifacts: Option<usize>,
    pub max_artifacts_per_geometry: Option<usize>,
    pub max_age_seconds: Option<u64>,
    pub require_latest_revision: Option<bool>,
}

type PrepStore = Arc<RwLock<HashMap<String, StoredGeometryPrepArtifact>>>;

fn prep_store() -> &'static PrepStore {
    static STORE: OnceLock<PrepStore> = OnceLock::new();
    STORE.get_or_init(|| Arc::new(RwLock::new(HashMap::new())))
}

fn prep_artifact_counter() -> &'static AtomicU64 {
    static COUNTER: OnceLock<AtomicU64> = OnceLock::new();
    COUNTER.get_or_init(|| AtomicU64::new(1))
}

fn prep_metrics() -> &'static Arc<RwLock<PrepArtifactMetrics>> {
    static METRICS: OnceLock<Arc<RwLock<PrepArtifactMetrics>>> = OnceLock::new();
    METRICS.get_or_init(|| Arc::new(RwLock::new(PrepArtifactMetrics::default())))
}

fn prep_config() -> &'static RwLock<GeometryPrepArtifactConfig> {
    static CONFIG: OnceLock<RwLock<GeometryPrepArtifactConfig>> = OnceLock::new();
    CONFIG.get_or_init(|| RwLock::new(GeometryPrepArtifactConfig::default()))
}

fn current_prep_config() -> GeometryPrepArtifactConfig {
    prep_config()
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

pub fn configure_prep_artifacts(config: GeometryPrepArtifactConfig) -> Result<(), String> {
    let mut guard = prep_config()
        .write()
        .map_err(|_| "geometry prep artifact config lock poisoned".to_string())?;
    *guard = config;
    Ok(())
}

fn increment_metric(f: impl FnOnce(&mut PrepArtifactMetrics)) {
    if let Ok(mut metrics) = prep_metrics().write() {
        f(&mut metrics);
    }
}

fn prep_artifact_root() -> Option<PathBuf> {
    current_prep_config().artifact_root.or_else(|| {
        std::env::var("RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT")
            .ok()
            .map(PathBuf::from)
    })
}

pub(crate) fn require_latest_prep_revision() -> bool {
    current_prep_config()
        .require_latest_revision
        .unwrap_or_else(|| {
            std::env::var("RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION")
                .ok()
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(true)
        })
}

fn prep_artifact_path(root: &PathBuf, prep_artifact_id: &str) -> PathBuf {
    root.join("prep").join(format!("{prep_artifact_id}.json"))
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

fn fs_read_dir(path: impl Into<PathBuf>) -> std::io::Result<Vec<runmat_filesystem::DirEntry>> {
    runmat_filesystem::read_dir(path.into())
}

fn fs_exists(path: impl Into<PathBuf>) -> std::io::Result<bool> {
    match runmat_filesystem::metadata(path.into()) {
        Ok(_) => Ok(true),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

#[derive(Debug, Clone, Copy)]
struct PrepArtifactRetentionPolicy {
    max_artifacts: usize,
    max_artifacts_per_geometry: usize,
    max_age_seconds: u64,
}

impl PrepArtifactRetentionPolicy {
    fn current() -> Self {
        let config = current_prep_config();
        Self {
            max_artifacts: config.max_artifacts.unwrap_or_else(|| {
                std::env::var("RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0)
            }),
            max_artifacts_per_geometry: config.max_artifacts_per_geometry.unwrap_or_else(|| {
                std::env::var("RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0)
            }),
            max_age_seconds: config.max_age_seconds.unwrap_or_else(|| {
                std::env::var("RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(0)
            }),
        }
    }
}

fn persist_prep_artifact(
    geometry: &GeometryAsset,
    prep: MeshingPrepResult,
) -> Result<StoredGeometryPrepArtifact, String> {
    let prep_artifact_id = format!(
        "prep:{}:{}:{}",
        geometry.geometry_id,
        geometry.revision,
        prep_artifact_counter().fetch_add(1, Ordering::Relaxed)
    );
    let artifact = StoredGeometryPrepArtifact {
        prep_artifact_id: prep_artifact_id.clone(),
        schema_version: "geometry_prep_artifact/v1".to_string(),
        created_at: Utc::now().to_rfc3339(),
        source_geometry_id: geometry.geometry_id.clone(),
        source_geometry_revision: geometry.revision,
        prep,
    };

    prep_store()
        .write()
        .map_err(|_| "geometry prep artifact store lock poisoned".to_string())?
        .insert(prep_artifact_id.clone(), artifact.clone());
    increment_metric(|metrics| metrics.created_count = metrics.created_count.saturating_add(1));
    tracing::info!(
        target: "runmat_geometry",
        "prep_artifact_created id={} geometry_id={} revision={}",
        prep_artifact_id,
        geometry.geometry_id,
        geometry.revision
    );

    if let Some(root) = prep_artifact_root() {
        let path = prep_artifact_path(&root, &prep_artifact_id);
        if let Some(parent) = path.parent() {
            fs_create_dir_all(parent)
                .map_err(|err| format!("failed to create prep artifact directory: {err}"))?;
        }
        let bytes = serde_json::to_vec_pretty(&artifact)
            .map_err(|err| format!("failed to encode prep artifact: {err}"))?;
        fs_write(&path, &bytes).map_err(|err| format!("failed to write prep artifact: {err}"))?;
    }

    prune_prep_artifacts(PrepArtifactRetentionPolicy::current())?;

    Ok(artifact)
}

pub(crate) fn load_prep_artifact(
    prep_artifact_id: &str,
) -> Result<Option<StoredGeometryPrepArtifact>, String> {
    if let Some(artifact) = prep_store()
        .read()
        .map_err(|_| "geometry prep artifact store lock poisoned".to_string())?
        .get(prep_artifact_id)
        .cloned()
    {
        return Ok(Some(artifact));
    }

    let Some(root) = prep_artifact_root() else {
        return Ok(None);
    };
    let path = prep_artifact_path(&root, prep_artifact_id);
    if !fs_exists(&path).map_err(|err| format!("failed to inspect prep artifact: {err}"))? {
        return Ok(None);
    }
    let bytes = fs_read(&path).map_err(|err| format!("failed to read prep artifact: {err}"))?;
    let artifact = serde_json::from_slice::<StoredGeometryPrepArtifact>(&bytes)
        .map_err(|err| format!("failed to decode prep artifact: {err}"))?;
    prep_store()
        .write()
        .map_err(|_| "geometry prep artifact store lock poisoned".to_string())?
        .insert(prep_artifact_id.to_string(), artifact.clone());
    increment_metric(|metrics| metrics.loaded_count = metrics.loaded_count.saturating_add(1));
    tracing::info!(
        target: "runmat_geometry",
        "prep_artifact_loaded id={} geometry_id={} revision={}",
        prep_artifact_id,
        artifact.source_geometry_id,
        artifact.source_geometry_revision
    );
    prune_prep_artifacts(PrepArtifactRetentionPolicy::current())?;
    Ok(Some(artifact))
}

pub(crate) fn record_prep_stale_reject() {
    increment_metric(|metrics| {
        metrics.stale_reject_count = metrics.stale_reject_count.saturating_add(1)
    });
    tracing::warn!(target: "runmat_geometry", "prep_artifact_rejected reason=stale");
}

pub(crate) fn record_prep_mismatch_reject() {
    increment_metric(|metrics| {
        metrics.mismatch_reject_count = metrics.mismatch_reject_count.saturating_add(1)
    });
    tracing::warn!(
        target: "runmat_geometry",
        "prep_artifact_rejected reason=mismatch"
    );
}

pub fn geometry_prep_artifact_health_op(
    query: GeometryPrepArtifactHealthQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryPrepArtifactHealthResult>, OperationErrorEnvelope> {
    let artifacts = list_prep_artifacts().map_err(|err| {
        operation_error(
            GEOMETRY_PREP_ARTIFACT_HEALTH_OPERATION,
            GEOMETRY_PREP_ARTIFACT_HEALTH_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.GEOMETRY.PREP_ARTIFACT_HEALTH.STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to list prep artifacts: {err}"),
            BTreeMap::new(),
        )
    })?;

    let now = Utc::now();
    let mut age_seconds = Vec::new();
    let mut per_geometry_map: HashMap<String, (u32, usize)> = HashMap::new();
    for artifact in &artifacts {
        if let Ok(created) = chrono::DateTime::parse_from_rfc3339(&artifact.created_at) {
            let age = now.signed_duration_since(created.with_timezone(&Utc));
            age_seconds.push(age.num_seconds().max(0) as f64);
        }
        let entry = per_geometry_map
            .entry(artifact.source_geometry_id.clone())
            .or_insert((artifact.source_geometry_revision, 0));
        if artifact.source_geometry_revision > entry.0 {
            entry.0 = artifact.source_geometry_revision;
        }
        entry.1 = entry.1.saturating_add(1);
    }
    age_seconds.sort_by(|a, b| a.total_cmp(b));

    let per_geometry = if query.include_per_geometry {
        let mut values = per_geometry_map
            .into_iter()
            .map(|(geometry_id, (latest_revision, artifact_count))| {
                GeometryPrepArtifactHealthEntry {
                    geometry_id,
                    latest_revision,
                    artifact_count,
                }
            })
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.geometry_id.cmp(&b.geometry_id));
        values
    } else {
        Vec::new()
    };

    let metrics = prep_metrics()
        .read()
        .map_err(|_| {
            operation_error(
                GEOMETRY_PREP_ARTIFACT_HEALTH_OPERATION,
                GEOMETRY_PREP_ARTIFACT_HEALTH_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.GEOMETRY.PREP_ARTIFACT_HEALTH.STORE_FAILED",
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                "geometry prep metrics store lock poisoned",
                BTreeMap::new(),
            )
        })?
        .clone();

    Ok(OperationEnvelope::new(
        GEOMETRY_PREP_ARTIFACT_HEALTH_OPERATION,
        GEOMETRY_PREP_ARTIFACT_HEALTH_OP_VERSION,
        &context,
        GeometryPrepArtifactHealthResult {
            schema_version: "geometry-prep-artifact-health/v1".to_string(),
            current_artifact_count: artifacts.len(),
            age_p50_seconds: percentile(&age_seconds, 0.5),
            age_p95_seconds: percentile(&age_seconds, 0.95),
            metrics,
            per_geometry,
        },
    ))
}

fn percentile(sorted: &[f64], ratio: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() - 1) as f64 * ratio.clamp(0.0, 1.0)).round() as usize;
    sorted.get(index).copied()
}

pub(crate) fn latest_prep_revision_for_geometry(geometry_id: &str) -> Result<Option<u32>, String> {
    let mut revisions = list_prep_artifacts()?
        .into_iter()
        .filter(|artifact| artifact.source_geometry_id == geometry_id)
        .map(|artifact| artifact.source_geometry_revision)
        .collect::<Vec<_>>();
    revisions.sort_unstable();
    Ok(revisions.pop())
}

fn list_prep_artifacts() -> Result<Vec<StoredGeometryPrepArtifact>, String> {
    let mut artifacts = prep_store()
        .read()
        .map_err(|_| "geometry prep artifact store lock poisoned".to_string())?
        .values()
        .cloned()
        .collect::<Vec<_>>();
    if artifacts.is_empty() {
        if let Some(root) = prep_artifact_root() {
            let prep_dir = root.join("prep");
            if fs_exists(&prep_dir)
                .map_err(|err| format!("failed to inspect prep artifacts: {err}"))?
            {
                for entry in fs_read_dir(&prep_dir)
                    .map_err(|err| format!("failed to scan prep artifacts: {err}"))?
                {
                    let path = entry.path().to_path_buf();
                    if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                        continue;
                    }
                    let bytes = fs_read(&path)
                        .map_err(|err| format!("failed to read prep artifact: {err}"))?;
                    if let Ok(artifact) =
                        serde_json::from_slice::<StoredGeometryPrepArtifact>(&bytes)
                    {
                        artifacts.push(artifact);
                    }
                }
            }
        }
    }
    Ok(artifacts)
}

fn prune_prep_artifacts(policy: PrepArtifactRetentionPolicy) -> Result<(), String> {
    if policy.max_artifacts == 0
        && policy.max_artifacts_per_geometry == 0
        && policy.max_age_seconds == 0
    {
        return Ok(());
    }

    let now = Utc::now();
    let mut artifacts = list_prep_artifacts()?;
    artifacts.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    let mut remove_ids = Vec::new();
    if policy.max_age_seconds > 0 {
        for artifact in &artifacts {
            if let Ok(created) = chrono::DateTime::parse_from_rfc3339(&artifact.created_at) {
                let age = now.signed_duration_since(created.with_timezone(&Utc));
                if age.num_seconds().max(0) as u64 > policy.max_age_seconds {
                    remove_ids.push(artifact.prep_artifact_id.clone());
                }
            }
        }
    }

    if policy.max_artifacts_per_geometry > 0 {
        let mut per_geometry_counts: HashMap<String, usize> = HashMap::new();
        for artifact in &artifacts {
            let count = per_geometry_counts
                .entry(artifact.source_geometry_id.clone())
                .or_default();
            *count += 1;
            if *count > policy.max_artifacts_per_geometry {
                remove_ids.push(artifact.prep_artifact_id.clone());
            }
        }
    }

    if policy.max_artifacts > 0 {
        for (index, artifact) in artifacts.iter().enumerate() {
            if index >= policy.max_artifacts {
                remove_ids.push(artifact.prep_artifact_id.clone());
            }
        }
    }

    remove_ids.sort();
    remove_ids.dedup();
    if remove_ids.is_empty() {
        return Ok(());
    }

    {
        let mut store = prep_store()
            .write()
            .map_err(|_| "geometry prep artifact store lock poisoned".to_string())?;
        for id in &remove_ids {
            store.remove(id);
        }
    }

    if let Some(root) = prep_artifact_root() {
        for id in &remove_ids {
            let path = prep_artifact_path(&root, id);
            let _ = fs_remove_file(path);
        }
    }

    increment_metric(|metrics| {
        metrics.pruned_count = metrics.pruned_count.saturating_add(remove_ids.len() as u64)
    });
    tracing::info!(
        target: "runmat_geometry",
        "prep_artifact_pruned count={}",
        remove_ids.len()
    );

    Ok(())
}

#[doc(hidden)]
pub fn reset_prep_artifact_store_for_tests() {
    if let Ok(mut store) = prep_store().write() {
        store.clear();
    }
    prep_artifact_counter().store(1, Ordering::Relaxed);
    if let Ok(mut metrics) = prep_metrics().write() {
        *metrics = PrepArtifactMetrics::default();
    }
    if let Ok(mut config) = prep_config().write() {
        *config = GeometryPrepArtifactConfig::default();
    }
}

#[cfg(test)]
pub(crate) fn prep_artifact_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Default for GeometryPrepForAnalysisSpec {
    fn default() -> Self {
        Self {
            profile: GeometryPrepProfile::AnalysisReady,
            target_element_budget: 250_000,
        }
    }
}

pub trait GeometryViewCaptureAdapter {
    fn adapter_name(&self) -> &'static str;
    fn capture(
        &self,
        asset: &GeometryAsset,
        view_spec: &GeometryCaptureViewSpec,
    ) -> Result<GeometryCaptureViewResult, String>;
}

thread_local! {
    static GEOMETRY_CAPTURE_ADAPTER: RefCell<Option<&'static dyn GeometryViewCaptureAdapter>> =
        RefCell::new(None);
}

mod capture;

pub struct ThreadGeometryCaptureAdapterGuard {
    previous: Option<&'static dyn GeometryViewCaptureAdapter>,
}

impl ThreadGeometryCaptureAdapterGuard {
    pub fn set(adapter: Option<&'static dyn GeometryViewCaptureAdapter>) -> Self {
        let previous = GEOMETRY_CAPTURE_ADAPTER.with(|slot| slot.replace(adapter));
        Self { previous }
    }
}

impl Drop for ThreadGeometryCaptureAdapterGuard {
    fn drop(&mut self) {
        GEOMETRY_CAPTURE_ADAPTER.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

pub fn geometry_inspect_op(
    path: &str,
    bytes: &[u8],
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryInspectResult>, OperationErrorEnvelope> {
    let format = runmat_geometry_io::detect_geometry_format(path, bytes);
    let data = GeometryInspectResult {
        format: format_name(format).to_string(),
        byte_count: bytes.len(),
    };
    Ok(OperationEnvelope::new(
        GEOMETRY_INSPECT_OPERATION,
        GEOMETRY_INSPECT_OP_VERSION,
        &context,
        data,
    ))
}

pub fn geometry_inspect(path: &str, bytes: &[u8]) -> BuiltinResult<GeometryInspectResult> {
    let envelope =
        geometry_inspect_op(path, bytes, OperationContext::new(None, None)).map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_INSPECT_OPERATION)
                .with_identifier("RunMat:GeometryInspectFailed")
                .build()
        })?;
    Ok(envelope.data)
}

pub fn geometry_load_op(
    path: &str,
    bytes: &[u8],
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryAsset>, OperationErrorEnvelope> {
    geometry_load_with_options_op(path, bytes, GeometryImportOptions::default(), context)
}

pub fn geometry_load_with_options_op(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryAsset>, OperationErrorEnvelope> {
    let imported = import_geometry(path, bytes, options)
        .map_err(|error| map_geometry_load_error(path, error, &context))?;
    Ok(OperationEnvelope::new(
        GEOMETRY_LOAD_OPERATION,
        GEOMETRY_LOAD_OP_VERSION,
        &context,
        imported.asset,
    ))
}

pub fn geometry_load(path: &str, bytes: &[u8]) -> BuiltinResult<GeometryAsset> {
    let envelope =
        geometry_load_op(path, bytes, OperationContext::new(None, None)).map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_LOAD_OPERATION)
                .with_identifier("RunMat:GeometryLoadFailed")
                .build()
        })?;
    Ok(envelope.data)
}

pub fn geometry_compute_stats_op(
    asset: &GeometryAsset,
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryStats>, OperationErrorEnvelope> {
    Ok(OperationEnvelope::new(
        GEOMETRY_COMPUTE_STATS_OPERATION,
        GEOMETRY_COMPUTE_STATS_OP_VERSION,
        &context,
        compute_stats(asset),
    ))
}

pub fn geometry_compute_stats(asset: &GeometryAsset) -> BuiltinResult<GeometryStats> {
    let envelope =
        geometry_compute_stats_op(asset, OperationContext::new(None, None)).map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_COMPUTE_STATS_OPERATION)
                .with_identifier("RunMat:GeometryStatsFailed")
                .build()
        })?;
    Ok(envelope.data)
}

pub fn geometry_list_regions_op(
    asset: &GeometryAsset,
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryRegionsResult>, OperationErrorEnvelope> {
    Ok(OperationEnvelope::new(
        GEOMETRY_LIST_REGIONS_OPERATION,
        GEOMETRY_LIST_REGIONS_OP_VERSION,
        &context,
        GeometryRegionsResult {
            regions: asset.regions.clone(),
        },
    ))
}

pub fn geometry_list_regions(asset: &GeometryAsset) -> BuiltinResult<GeometryRegionsResult> {
    let envelope =
        geometry_list_regions_op(asset, OperationContext::new(None, None)).map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_LIST_REGIONS_OPERATION)
                .with_identifier("RunMat:GeometryListRegionsFailed")
                .build()
        })?;
    Ok(envelope.data)
}

pub fn geometry_query_entities_op(
    asset: &GeometryAsset,
    query: GeometryEntityQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryEntityQueryResult>, OperationErrorEnvelope> {
    let requested_limit = query.limit.unwrap_or(DEFAULT_QUERY_LIMIT);
    if requested_limit == 0 {
        return Err(operation_error(
            GEOMETRY_QUERY_ENTITIES_OPERATION,
            GEOMETRY_QUERY_ENTITIES_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.GEOMETRY.QUERY_ENTITIES.INVALID_LIMIT",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "entity query limit must be greater than zero",
            BTreeMap::new(),
        ));
    }

    if let Some(region_id) = query.region_id.as_ref() {
        find_region(asset, region_id)
            .map_err(|error| map_geometry_query_error(region_id, error, &context))?;
    }

    let mut entities = Vec::new();
    let mut produced_total = 0usize;

    for mesh in &asset.meshes {
        if query
            .mesh_id
            .as_ref()
            .is_some_and(|mesh_id| mesh_id != &mesh.mesh_id)
        {
            continue;
        }

        let count = match query.entity_kind {
            EntityKind::Node => mesh.vertex_count as usize,
            EntityKind::Element | EntityKind::Face => mesh.element_count as usize,
            EntityKind::Edge => 0,
        };

        produced_total += count;

        if entities.len() >= requested_limit {
            continue;
        }

        let remaining = requested_limit - entities.len();
        let emit = count.min(remaining);
        for entity_id in 0..emit {
            entities.push(EntityRef {
                geometry_id: asset.geometry_id.clone(),
                geometry_revision: asset.revision,
                mesh_id: mesh.mesh_id.clone(),
                entity_kind: query.entity_kind,
                entity_id: entity_id as u64,
            });
        }
    }

    Ok(OperationEnvelope::new(
        GEOMETRY_QUERY_ENTITIES_OPERATION,
        GEOMETRY_QUERY_ENTITIES_OP_VERSION,
        &context,
        GeometryEntityQueryResult {
            entities,
            truncated: produced_total > requested_limit,
        },
    ))
}

pub fn geometry_query_entities(
    asset: &GeometryAsset,
    query: GeometryEntityQuery,
) -> BuiltinResult<GeometryEntityQueryResult> {
    let envelope = geometry_query_entities_op(asset, query, OperationContext::new(None, None))
        .map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_QUERY_ENTITIES_OPERATION)
                .with_identifier("RunMat:GeometryQueryEntitiesFailed")
                .build()
        })?;
    Ok(envelope.data)
}

pub fn geometry_capture_view_op(
    asset: &GeometryAsset,
    view_spec: GeometryCaptureViewSpec,
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryCaptureViewResult>, OperationErrorEnvelope> {
    if view_spec.width == 0 || view_spec.height == 0 {
        return Err(operation_error(
            GEOMETRY_CAPTURE_VIEW_OPERATION,
            GEOMETRY_CAPTURE_VIEW_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.GEOMETRY.CAPTURE_VIEW.INVALID_SPEC",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "capture view dimensions must be greater than zero",
            BTreeMap::from([
                ("width".to_string(), view_spec.width.to_string()),
                ("height".to_string(), view_spec.height.to_string()),
            ]),
        ));
    }

    let adapter = GEOMETRY_CAPTURE_ADAPTER.with(|slot| *slot.borrow());
    if let Some(adapter) = adapter {
        let capture = adapter.capture(asset, &view_spec).map_err(|message| {
            operation_error(
                GEOMETRY_CAPTURE_VIEW_OPERATION,
                GEOMETRY_CAPTURE_VIEW_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.GEOMETRY.CAPTURE_VIEW.BACKEND_FAILED",
                    error_type: OperationErrorType::Backend,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                message,
                BTreeMap::from([
                    ("geometry_id".to_string(), asset.geometry_id.clone()),
                    ("adapter".to_string(), adapter.adapter_name().to_string()),
                ]),
            )
        })?;
        return Ok(OperationEnvelope::new(
            GEOMETRY_CAPTURE_VIEW_OPERATION,
            GEOMETRY_CAPTURE_VIEW_OP_VERSION,
            &context,
            capture,
        ));
    }

    if view_spec.format.eq_ignore_ascii_case("svg") {
        let capture = DEFAULT_SVG_CAPTURE_ADAPTER
            .capture(asset, &view_spec)
            .map_err(|message| {
                operation_error(
                    GEOMETRY_CAPTURE_VIEW_OPERATION,
                    GEOMETRY_CAPTURE_VIEW_OP_VERSION,
                    &context,
                    OperationErrorSpec {
                        error_code: "RM.GEOMETRY.CAPTURE_VIEW.BACKEND_FAILED",
                        error_type: OperationErrorType::Backend,
                        retryable: true,
                        severity: OperationErrorSeverity::Error,
                    },
                    message,
                    BTreeMap::from([
                        ("geometry_id".to_string(), asset.geometry_id.clone()),
                        (
                            "adapter".to_string(),
                            DEFAULT_SVG_CAPTURE_ADAPTER.adapter_name().to_string(),
                        ),
                    ]),
                )
            })?;
        return Ok(OperationEnvelope::new(
            GEOMETRY_CAPTURE_VIEW_OPERATION,
            GEOMETRY_CAPTURE_VIEW_OP_VERSION,
            &context,
            capture,
        ));
    }

    Err(operation_error(
        GEOMETRY_CAPTURE_VIEW_OPERATION,
        GEOMETRY_CAPTURE_VIEW_OP_VERSION,
        &context,
        OperationErrorSpec {
            error_code: "RM.GEOMETRY.CAPTURE_VIEW.UNSUPPORTED",
            error_type: OperationErrorType::Backend,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        "geometry view capture is not wired in runtime yet",
        BTreeMap::from([("geometry_id".to_string(), asset.geometry_id.clone())]),
    ))
}

pub fn geometry_capture_view(
    asset: &GeometryAsset,
    view_spec: GeometryCaptureViewSpec,
) -> BuiltinResult<GeometryCaptureViewResult> {
    let envelope = geometry_capture_view_op(asset, view_spec, OperationContext::new(None, None))
        .map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_CAPTURE_VIEW_OPERATION)
                .with_identifier("RunMat:GeometryCaptureViewFailed")
                .build()
        })?;
    Ok(envelope.data)
}

pub fn geometry_prep_for_analysis_op(
    asset: &GeometryAsset,
    spec: GeometryPrepForAnalysisSpec,
    context: OperationContext,
) -> Result<OperationEnvelope<GeometryPrepForAnalysisResult>, OperationErrorEnvelope> {
    if spec.target_element_budget == 0 {
        return Err(operation_error(
            GEOMETRY_PREP_FOR_ANALYSIS_OPERATION,
            GEOMETRY_PREP_FOR_ANALYSIS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.GEOMETRY.PREP_FOR_ANALYSIS.INVALID_SPEC",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            "prep-for-analysis target_element_budget must be greater than zero",
            BTreeMap::from([(
                "target_element_budget".to_string(),
                spec.target_element_budget.to_string(),
            )]),
        ));
    }

    let profile = match spec.profile {
        GeometryPrepProfile::SurfaceOnly => MeshingProfile::SurfaceOnly,
        GeometryPrepProfile::AnalysisReady => MeshingProfile::AnalysisReady,
        GeometryPrepProfile::AdaptiveRefine => MeshingProfile::AdaptiveRefine,
    };
    let prepared = prepare_geometry_for_analysis(
        asset,
        MeshingOptions {
            profile,
            target_element_budget: spec.target_element_budget,
        },
    )
    .map_err(|error| {
        operation_error(
            GEOMETRY_PREP_FOR_ANALYSIS_OPERATION,
            GEOMETRY_PREP_FOR_ANALYSIS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.GEOMETRY.PREP_FOR_ANALYSIS.FAILED",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to prepare geometry for analysis: {error}"),
            BTreeMap::from([("geometry_id".to_string(), asset.geometry_id.clone())]),
        )
    })?;

    let artifact = persist_prep_artifact(asset, prepared).map_err(|error| {
        operation_error(
            GEOMETRY_PREP_FOR_ANALYSIS_OPERATION,
            GEOMETRY_PREP_FOR_ANALYSIS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.GEOMETRY.PREP_FOR_ANALYSIS.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to persist prep artifact: {error}"),
            BTreeMap::from([("geometry_id".to_string(), asset.geometry_id.clone())]),
        )
    })?;

    Ok(OperationEnvelope::new(
        GEOMETRY_PREP_FOR_ANALYSIS_OPERATION,
        GEOMETRY_PREP_FOR_ANALYSIS_OP_VERSION,
        &context,
        GeometryPrepForAnalysisResult {
            prep_artifact_id: artifact.prep_artifact_id,
            prep: artifact.prep,
        },
    ))
}

pub fn geometry_prep_for_analysis(
    asset: &GeometryAsset,
    spec: GeometryPrepForAnalysisSpec,
) -> BuiltinResult<GeometryPrepForAnalysisResult> {
    let envelope = geometry_prep_for_analysis_op(asset, spec, OperationContext::new(None, None))
        .map_err(|error| {
            build_runtime_error(error.message)
                .with_builtin(GEOMETRY_PREP_FOR_ANALYSIS_OPERATION)
                .with_identifier("RunMat:GeometryPrepForAnalysisFailed")
                .build()
        })?;
    Ok(envelope.data)
}

fn format_name(format: GeometryFormat) -> &'static str {
    match format {
        runmat_geometry_io::GeometryFormat::Stl => "stl",
        runmat_geometry_io::GeometryFormat::Step => "step",
        runmat_geometry_io::GeometryFormat::Obj => "obj",
        runmat_geometry_io::GeometryFormat::Ply => "ply",
        runmat_geometry_io::GeometryFormat::Gltf => "gltf",
        runmat_geometry_io::GeometryFormat::Unknown => "unknown",
    }
}

fn map_geometry_load_error(
    path: &str,
    error: GeometryImportError,
    context: &OperationContext,
) -> OperationErrorEnvelope {
    let (error_code, error_type, retryable) = match &error {
        GeometryImportError::UnsupportedFormat => (
            "RM.GEOMETRY.LOAD.FORMAT_UNSUPPORTED",
            OperationErrorType::Input,
            false,
        ),
        GeometryImportError::ParseFailed(_) => (
            "RM.GEOMETRY.LOAD.PARSE_FAILED",
            OperationErrorType::Validation,
            false,
        ),
        GeometryImportError::CapacityExceeded { .. } => (
            "RM.GEOMETRY.LOAD.CAPACITY_LIMIT_EXCEEDED",
            OperationErrorType::Capacity,
            false,
        ),
    };
    operation_error(
        GEOMETRY_LOAD_OPERATION,
        GEOMETRY_LOAD_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code,
            error_type,
            retryable,
            severity: OperationErrorSeverity::Error,
        },
        error.to_string(),
        BTreeMap::from([("path".to_string(), path.to_string())]),
    )
}

fn map_geometry_query_error(
    region_id: &str,
    error: QueryError,
    context: &OperationContext,
) -> OperationErrorEnvelope {
    match error {
        QueryError::RegionNotFound => operation_error(
            GEOMETRY_QUERY_ENTITIES_OPERATION,
            GEOMETRY_QUERY_ENTITIES_OP_VERSION,
            context,
            OperationErrorSpec {
                error_code: "RM.GEOMETRY.QUERY_ENTITIES.REGION_NOT_FOUND",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!("region '{region_id}' does not exist"),
            BTreeMap::from([("region_id".to_string(), region_id.to_string())]),
        ),
    }
}

#[cfg(test)]
mod tests;
