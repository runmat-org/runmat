use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

use self::capture::DEFAULT_SVG_CAPTURE_ADAPTER;
use chrono::Utc;
use runmat_geometry_core::{
    EntityIdRange, EntityKind, EntityRef, GeometryAsset, GeometrySource, MeshKind, Region,
    SourceGeometry, SourceGeometryKind, TessellationProfile, UnitSystem,
};
use runmat_geometry_io::{
    import::GeometryImportError, import_geometry_with_context, GeometryFormat,
    GeometryImportContext, GeometryImportOptions,
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
const DEFAULT_MAPPING_RANGE_PREVIEW_LIMIT: usize = 8;

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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryBoundsSummary {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryMeshSummary {
    pub mesh_id: String,
    pub kind: MeshKind,
    pub vertex_count: u64,
    pub element_count: u64,
    pub surface_vertex_count: Option<u64>,
    pub surface_triangle_count: Option<u64>,
    pub bounds: Option<GeometryBoundsSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryRegionMappingSummaryEntry {
    pub region_id: String,
    pub mesh_id: String,
    pub entity_kind: EntityKind,
    pub range_count: usize,
    pub entity_count: u64,
    pub range_preview: Vec<EntityIdRange>,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryRegionMappingSummary {
    pub mapping_count: usize,
    pub mapped_region_count: usize,
    pub total_entity_count: u64,
    pub range_preview_limit: usize,
    pub entries: Vec<GeometryRegionMappingSummaryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryCadRegionStatus {
    NotCad,
    MetadataOnly,
    GenericFaceTopology,
    SemanticRegions,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryCadSummary {
    pub backend: Option<String>,
    pub source_format: Option<String>,
    pub face_region_count: usize,
    pub mapped_face_region_count: usize,
    pub semantic_region_count: usize,
    pub mapped_semantic_region_count: usize,
    pub region_status: GeometryCadRegionStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryAssetSummary {
    pub geometry_id: String,
    pub revision: u32,
    pub source: GeometrySource,
    pub source_geometry: SourceGeometry,
    pub tessellation_profile: TessellationProfile,
    pub units: UnitSystem,
    pub meshes: Vec<GeometryMeshSummary>,
    pub mapping_summary: GeometryRegionMappingSummary,
    pub cad: GeometryCadSummary,
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

#[cfg(feature = "plot-core")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryPreviewPresentation {
    Analysis,
    Cad,
}

#[cfg(feature = "plot-core")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryPreviewFigureOptions {
    pub edge_overlay_triangle_limit: usize,
    pub presentation: GeometryPreviewPresentation,
    pub xray: bool,
}

#[cfg(feature = "plot-core")]
impl Default for GeometryPreviewFigureOptions {
    fn default() -> Self {
        Self {
            edge_overlay_triangle_limit: 250_000,
            presentation: GeometryPreviewPresentation::Analysis,
            xray: false,
        }
    }
}

#[cfg(feature = "plot-core")]
impl GeometryPreviewFigureOptions {
    pub fn cad_preview() -> Self {
        Self {
            edge_overlay_triangle_limit: 250_000,
            presentation: GeometryPreviewPresentation::Cad,
            xray: false,
        }
    }
}

#[cfg(feature = "plot-core")]
const CAD_DEFAULT_FACE_COLOR: glam::Vec4 = glam::Vec4::new(0.66, 0.72, 0.80, 1.0);
#[cfg(feature = "plot-core")]
const CAD_FEATURE_EDGE_COLOR: glam::Vec4 = glam::Vec4::new(0.08, 0.10, 0.13, 1.0);

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

fn prep_artifact_path(root: &Path, prep_artifact_id: &str) -> PathBuf {
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
    let import_context = current_geometry_import_context();
    let imported = import_geometry_with_context(path, bytes, options, &import_context)
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

fn current_geometry_import_context() -> GeometryImportContext {
    crate::interrupt::current_interrupt()
        .map(GeometryImportContext::with_cancellation)
        .unwrap_or_default()
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

pub fn geometry_asset_summary(asset: &GeometryAsset) -> GeometryAssetSummary {
    geometry_asset_summary_with_options(asset, DEFAULT_MAPPING_RANGE_PREVIEW_LIMIT)
}

pub fn geometry_asset_summary_with_options(
    asset: &GeometryAsset,
    range_preview_limit: usize,
) -> GeometryAssetSummary {
    GeometryAssetSummary {
        geometry_id: asset.geometry_id.clone(),
        revision: asset.revision,
        source: asset.source.clone(),
        source_geometry: asset.source_geometry.clone(),
        tessellation_profile: asset.tessellation_profile.clone(),
        units: asset.units,
        meshes: mesh_summaries(asset),
        mapping_summary: region_mapping_summary(asset, range_preview_limit),
        cad: cad_summary(asset),
    }
}

fn mesh_summaries(asset: &GeometryAsset) -> Vec<GeometryMeshSummary> {
    asset
        .meshes
        .iter()
        .map(|mesh| {
            let surface_mesh = asset
                .surface_meshes
                .iter()
                .find(|surface| surface.mesh_id == mesh.mesh_id);
            GeometryMeshSummary {
                mesh_id: mesh.mesh_id.clone(),
                kind: mesh.kind,
                vertex_count: mesh.vertex_count,
                element_count: mesh.element_count,
                surface_vertex_count: surface_mesh.map(|surface| surface.vertices.len() as u64),
                surface_triangle_count: surface_mesh.map(|surface| surface.triangles.len() as u64),
                bounds: surface_mesh.and_then(|surface| bounds_for_vertices(&surface.vertices)),
            }
        })
        .collect()
}

fn bounds_for_vertices(vertices: &[[f64; 3]]) -> Option<GeometryBoundsSummary> {
    let first = vertices.first().copied()?;
    let mut min = first;
    let mut max = first;
    for vertex in vertices.iter().skip(1) {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    Some(GeometryBoundsSummary { min, max })
}

fn region_mapping_summary(
    asset: &GeometryAsset,
    range_preview_limit: usize,
) -> GeometryRegionMappingSummary {
    let mut mapped_regions = BTreeSet::new();
    let mut total_entity_count = 0_u64;
    let entries = asset
        .region_entity_mappings
        .iter()
        .map(|mapping| {
            mapped_regions.insert(mapping.region_id.clone());
            let entity_count = mapping.entity_count();
            total_entity_count = total_entity_count.saturating_add(entity_count);
            let range_count = mapping.ranges.len();
            GeometryRegionMappingSummaryEntry {
                region_id: mapping.region_id.clone(),
                mesh_id: mapping.mesh_id.clone(),
                entity_kind: mapping.entity_kind,
                range_count,
                entity_count,
                range_preview: mapping
                    .ranges
                    .iter()
                    .take(range_preview_limit)
                    .copied()
                    .collect(),
                truncated: range_count > range_preview_limit,
            }
        })
        .collect();

    GeometryRegionMappingSummary {
        mapping_count: asset.region_entity_mappings.len(),
        mapped_region_count: mapped_regions.len(),
        total_entity_count,
        range_preview_limit,
        entries,
    }
}

fn cad_summary(asset: &GeometryAsset) -> GeometryCadSummary {
    let importer_parts = asset.source.importer_version.split('/').collect::<Vec<_>>();
    let backend = match importer_parts.as_slice() {
        ["cad", backend, ..] => Some((*backend).to_string()),
        ["step", ..] if asset.source_geometry.kind == SourceGeometryKind::Cad => {
            Some("metadata".to_string())
        }
        _ => None,
    };
    let source_format = match importer_parts.as_slice() {
        ["cad", _, format, ..] => Some((*format).to_string()),
        [format, ..] if asset.source_geometry.kind == SourceGeometryKind::Cad => {
            Some((*format).to_string())
        }
        _ => None,
    };
    let face_region_ids = asset
        .regions
        .iter()
        .filter(|region| region.tag.as_deref() == Some("occt_face"))
        .map(|region| region.region_id.as_str())
        .collect::<BTreeSet<_>>();
    let mapped_face_region_ids = asset
        .region_entity_mappings
        .iter()
        .filter_map(|mapping| {
            (mapping.entity_kind == EntityKind::Face
                && face_region_ids.contains(mapping.region_id.as_str()))
            .then_some(mapping.region_id.as_str())
        })
        .collect::<BTreeSet<_>>();
    let semantic_region_ids = asset
        .regions
        .iter()
        .filter(|region| {
            region.cad_ownership.is_some()
                && region
                    .tag
                    .as_deref()
                    .is_some_and(|tag| tag.starts_with("cad_"))
        })
        .map(|region| region.region_id.as_str())
        .collect::<BTreeSet<_>>();
    let mapped_semantic_region_ids = asset
        .region_entity_mappings
        .iter()
        .filter_map(|mapping| {
            (mapping.entity_kind == EntityKind::Face
                && semantic_region_ids.contains(mapping.region_id.as_str()))
            .then_some(mapping.region_id.as_str())
        })
        .collect::<BTreeSet<_>>();
    let region_status = if asset.source_geometry.kind != SourceGeometryKind::Cad {
        GeometryCadRegionStatus::NotCad
    } else if mapped_face_region_ids.is_empty() {
        GeometryCadRegionStatus::MetadataOnly
    } else if !mapped_semantic_region_ids.is_empty() {
        GeometryCadRegionStatus::SemanticRegions
    } else {
        GeometryCadRegionStatus::GenericFaceTopology
    };

    GeometryCadSummary {
        backend,
        source_format,
        face_region_count: face_region_ids.len(),
        mapped_face_region_count: mapped_face_region_ids.len(),
        semantic_region_count: semantic_region_ids.len(),
        mapped_semantic_region_count: mapped_semantic_region_ids.len(),
        region_status,
    }
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
        return Ok(OperationEnvelope::new(
            GEOMETRY_QUERY_ENTITIES_OPERATION,
            GEOMETRY_QUERY_ENTITIES_OP_VERSION,
            &context,
            query_region_entities(asset, &query, region_id, requested_limit),
        ));
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

fn query_region_entities(
    asset: &GeometryAsset,
    query: &GeometryEntityQuery,
    region_id: &str,
    requested_limit: usize,
) -> GeometryEntityQueryResult {
    if query.entity_kind == EntityKind::Node {
        return query_region_nodes(asset, query, region_id, requested_limit);
    }

    let mut entities = Vec::new();
    let mut produced_total = 0usize;
    for mapping in asset.region_entity_mappings.iter().filter(|mapping| {
        mapping.region_id == region_id
            && query
                .mesh_id
                .as_ref()
                .is_none_or(|mesh_id| mesh_id == &mapping.mesh_id)
            && mapping_matches_query_kind(mapping.entity_kind, query.entity_kind)
    }) {
        let mapped_total = mapping.entity_count() as usize;
        produced_total = produced_total.saturating_add(mapped_total);
        if entities.len() >= requested_limit {
            continue;
        }
        for range in &mapping.ranges {
            let Some(end) = range.end_exclusive() else {
                continue;
            };
            for entity_id in range.start..end {
                if entities.len() >= requested_limit {
                    break;
                }
                entities.push(EntityRef {
                    geometry_id: asset.geometry_id.clone(),
                    geometry_revision: asset.revision,
                    mesh_id: mapping.mesh_id.clone(),
                    entity_kind: query.entity_kind,
                    entity_id,
                });
            }
        }
    }

    GeometryEntityQueryResult {
        entities,
        truncated: produced_total > requested_limit,
    }
}

fn query_region_nodes(
    asset: &GeometryAsset,
    query: &GeometryEntityQuery,
    region_id: &str,
    requested_limit: usize,
) -> GeometryEntityQueryResult {
    let mut node_refs = BTreeSet::<(String, u64)>::new();
    let mut truncated = false;

    for mapping in asset.region_entity_mappings.iter().filter(|mapping| {
        mapping.region_id == region_id
            && query
                .mesh_id
                .as_ref()
                .is_none_or(|mesh_id| mesh_id == &mapping.mesh_id)
            && mapping_matches_query_kind(mapping.entity_kind, EntityKind::Face)
    }) {
        let Some(surface_mesh) = asset
            .surface_meshes
            .iter()
            .find(|mesh| mesh.mesh_id == mapping.mesh_id)
        else {
            continue;
        };
        for range in &mapping.ranges {
            let Some(end) = range.end_exclusive() else {
                continue;
            };
            for face_id in range.start..end {
                let Some(triangle) = surface_mesh.triangles.get(face_id as usize) else {
                    continue;
                };
                for vertex_id in triangle {
                    node_refs.insert((mapping.mesh_id.clone(), *vertex_id as u64));
                    if node_refs.len() > requested_limit {
                        truncated = true;
                        break;
                    }
                }
                if truncated {
                    break;
                }
            }
            if truncated {
                break;
            }
        }
        if truncated {
            break;
        }
    }

    let entities = node_refs
        .into_iter()
        .take(requested_limit)
        .map(|(mesh_id, entity_id)| EntityRef {
            geometry_id: asset.geometry_id.clone(),
            geometry_revision: asset.revision,
            mesh_id,
            entity_kind: EntityKind::Node,
            entity_id,
        })
        .collect();

    GeometryEntityQueryResult {
        entities,
        truncated,
    }
}

fn mapping_matches_query_kind(mapping_kind: EntityKind, query_kind: EntityKind) -> bool {
    mapping_kind == query_kind
        || matches!(
            (mapping_kind, query_kind),
            (EntityKind::Face, EntityKind::Element) | (EntityKind::Element, EntityKind::Face)
        )
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

#[cfg(feature = "plot-core")]
pub fn geometry_preview_figure(
    asset: &GeometryAsset,
    title: impl Into<String>,
    options: GeometryPreviewFigureOptions,
) -> Result<runmat_plot::plots::Figure, String> {
    if asset.surface_meshes.is_empty() {
        return Err("geometry asset does not contain renderable surface mesh data".to_string());
    }

    let cad_presentation = options.presentation == GeometryPreviewPresentation::Cad;
    let mut figure = if cad_presentation {
        runmat_plot::plots::Figure::new()
            .with_grid(false)
            .with_legend(false)
            .with_axis_equal(true)
    } else {
        let mut figure = runmat_plot::plots::Figure::new()
            .with_title(title)
            .with_labels("X", "Y")
            .with_grid(true)
            .with_axis_equal(true);
        figure.z_label = Some("Z".to_string());
        figure
    };
    if cad_presentation {
        figure.set_axes_view(0, -38.0, 24.0);
    }

    for (index, surface_mesh) in asset.surface_meshes.iter().enumerate() {
        let vertices = surface_mesh
            .vertices
            .iter()
            .map(|vertex| {
                Ok(glam::Vec3::new(
                    f64_to_f32_coordinate(vertex[0])?,
                    f64_to_f32_coordinate(vertex[1])?,
                    f64_to_f32_coordinate(vertex[2])?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut mesh = runmat_plot::plots::MeshPlot::new(vertices, surface_mesh.triangles.clone())?;
        mesh.set_mesh_id(Some(surface_mesh.mesh_id.clone()));
        mesh.set_regions(mesh_regions_for_surface(asset, &surface_mesh.mesh_id));
        if !cad_presentation {
            mesh.set_label(Some(format!(
                "{}: {} triangles",
                surface_mesh.mesh_id,
                surface_mesh.triangles.len()
            )));
        }

        if cad_presentation {
            let presentation = cad_mesh_presentation(
                asset,
                &surface_mesh.mesh_id,
                surface_mesh.triangles.len(),
                surface_mesh.vertices.len(),
            );
            mesh.set_face_color(CAD_DEFAULT_FACE_COLOR);
            mesh.set_edge_color(CAD_FEATURE_EDGE_COLOR);
            mesh.set_face_alpha(if options.xray { 0.34 } else { 1.0 });
            mesh.set_edge_alpha(if options.xray { 0.9 } else { 0.72 });
            if let Some(colors) = presentation.vertex_colors {
                mesh.set_vertex_colors(Some(colors))?;
            }
            if let Some(groups) = presentation.feature_edge_groups {
                mesh.set_feature_edge_groups(Some(groups))?;
                mesh.set_edge_mode(runmat_plot::plots::MeshEdgeMode::Feature);
                mesh.set_edge_width(0.85);
            } else if surface_mesh.triangles.len() > options.edge_overlay_triangle_limit {
                mesh.set_edge_mode(runmat_plot::plots::MeshEdgeMode::None);
                mesh.set_edge_width(0.0);
            } else {
                mesh.set_edge_mode(runmat_plot::plots::MeshEdgeMode::All);
                mesh.set_edge_width(0.28);
            }
        } else {
            let color = preview_mesh_color(index);
            mesh.set_face_color(color);
            mesh.set_edge_color(glam::Vec4::new(0.86, 0.91, 1.0, 0.82));
            mesh.set_face_alpha(0.92);
            if surface_mesh.triangles.len() > options.edge_overlay_triangle_limit {
                mesh.set_edge_width(0.0);
            } else {
                mesh.set_edge_width(0.35);
            }
        }
        figure.add_mesh_plot(mesh);
    }

    Ok(figure)
}

#[cfg(feature = "plot-core")]
#[derive(Debug, Default)]
struct CadMeshPresentation {
    feature_edge_groups: Option<Vec<u64>>,
    vertex_colors: Option<Vec<glam::Vec4>>,
}

#[cfg(feature = "plot-core")]
fn cad_mesh_presentation(
    asset: &GeometryAsset,
    mesh_id: &str,
    triangle_count: usize,
    vertex_count: usize,
) -> CadMeshPresentation {
    if triangle_count == 0 {
        return CadMeshPresentation::default();
    }

    let prefer_face_mappings = asset.source_geometry.kind == SourceGeometryKind::Cad;
    let mut feature_edge_groups = vec![0_u64; triangle_count];
    let mut vertex_colors = vec![CAD_DEFAULT_FACE_COLOR; vertex_count];
    let mut group_ids_by_region = BTreeMap::<String, u64>::new();
    let mut assigned_groups = false;
    let mut assigned_colors = false;
    let surface_triangles = asset
        .surface_meshes
        .iter()
        .find(|surface_mesh| surface_mesh.mesh_id == mesh_id)
        .map(|surface_mesh| surface_mesh.triangles.as_slice());

    for mapping in asset.region_entity_mappings.iter().filter(|mapping| {
        mapping.mesh_id == mesh_id
            && matches!(mapping.entity_kind, EntityKind::Face | EntityKind::Element)
    }) {
        let Some(region) = asset
            .regions
            .iter()
            .find(|region| region.region_id == mapping.region_id)
        else {
            continue;
        };
        let face_id = region
            .cad_ownership
            .as_ref()
            .and_then(|ownership| ownership.face_id);
        if prefer_face_mappings && face_id.is_none() {
            continue;
        }
        let group_id = face_id
            .map(|face_id| face_id.saturating_add(1))
            .unwrap_or_else(|| {
                if let Some(group_id) = group_ids_by_region.get(&mapping.region_id) {
                    *group_id
                } else {
                    let group_id = group_ids_by_region.len() as u64 + 1;
                    group_ids_by_region.insert(mapping.region_id.clone(), group_id);
                    group_id
                }
            });
        let color = cad_region_color(region);
        for range in &mapping.ranges {
            for triangle_index in bounded_range(range, triangle_count) {
                feature_edge_groups[triangle_index] = group_id;
                assigned_groups = true;
                if let Some(color) = color {
                    assigned_colors |= color_vertices_for_triangle(
                        surface_triangles,
                        triangle_index,
                        color,
                        &mut vertex_colors,
                    );
                }
            }
        }
    }

    CadMeshPresentation {
        feature_edge_groups: assigned_groups.then_some(feature_edge_groups),
        vertex_colors: assigned_colors.then_some(vertex_colors),
    }
}

#[cfg(feature = "plot-core")]
fn bounded_range(range: &EntityIdRange, upper_bound: usize) -> std::ops::Range<usize> {
    let start = usize::try_from(range.start).unwrap_or(usize::MAX);
    let count = usize::try_from(range.count).unwrap_or(usize::MAX);
    let start = start.min(upper_bound);
    let end = start.saturating_add(count).min(upper_bound);
    start..end
}

#[cfg(feature = "plot-core")]
fn color_vertices_for_triangle(
    triangles: Option<&[[u32; 3]]>,
    triangle_index: usize,
    color: glam::Vec4,
    vertex_colors: &mut [glam::Vec4],
) -> bool {
    let Some(triangle) = triangles.and_then(|triangles| triangles.get(triangle_index)) else {
        return false;
    };
    let mut colored = false;
    for vertex_id in triangle {
        if let Some(slot) = vertex_colors.get_mut(*vertex_id as usize) {
            *slot = color;
            colored = true;
        }
    }
    colored
}

#[cfg(feature = "plot-core")]
fn cad_region_color(region: &Region) -> Option<glam::Vec4> {
    region
        .cad_ownership
        .as_ref()
        .and_then(|ownership| ownership.color.as_ref())
        .and_then(|color| parse_cad_hex_rgba(&color.hex_rgba))
        .map(cad_display_color)
}

#[cfg(feature = "plot-core")]
fn parse_cad_hex_rgba(value: &str) -> Option<glam::Vec4> {
    let value = value.trim().trim_start_matches('#');
    if value.len() != 6 && value.len() != 8 {
        return None;
    }
    let r = u8::from_str_radix(&value[0..2], 16).ok()? as f32 / 255.0;
    let g = u8::from_str_radix(&value[2..4], 16).ok()? as f32 / 255.0;
    let b = u8::from_str_radix(&value[4..6], 16).ok()? as f32 / 255.0;
    let a = if value.len() == 8 {
        u8::from_str_radix(&value[6..8], 16).ok()? as f32 / 255.0
    } else {
        1.0
    };
    Some(glam::Vec4::new(r, g, b, a))
}

#[cfg(feature = "plot-core")]
fn cad_display_color(color: glam::Vec4) -> glam::Vec4 {
    let rgb = glam::Vec3::new(color.x, color.y, color.z);
    let gray = glam::Vec3::splat((rgb.x + rgb.y + rgb.z) / 3.0);
    let softened = rgb
        .lerp(gray, 0.18)
        .lerp(CAD_DEFAULT_FACE_COLOR.truncate(), 0.16);
    glam::Vec4::new(softened.x, softened.y, softened.z, color.w.max(0.2))
}

#[cfg(feature = "plot-core")]
fn mesh_regions_for_surface(
    asset: &GeometryAsset,
    mesh_id: &str,
) -> Vec<runmat_plot::plots::MeshRegion> {
    asset
        .region_entity_mappings
        .iter()
        .filter(|mapping| {
            mapping.mesh_id == mesh_id
                && matches!(mapping.entity_kind, EntityKind::Face | EntityKind::Element)
        })
        .filter_map(|mapping| {
            let triangle_ranges = mapping
                .ranges
                .iter()
                .filter_map(|range| {
                    let start = u32::try_from(range.start).ok()?;
                    let count = u32::try_from(range.count).ok()?;
                    if count == 0 {
                        None
                    } else {
                        Some(runmat_plot::plots::MeshTriangleRange::new(start, count))
                    }
                })
                .collect::<Vec<_>>();
            if triangle_ranges.is_empty() {
                return None;
            }
            let region = asset
                .regions
                .iter()
                .find(|region| region.region_id == mapping.region_id);
            Some(runmat_plot::plots::MeshRegion::new(
                mapping.region_id.clone(),
                region.map(|region| region.name.clone()),
                region.and_then(|region| region.tag.clone()),
                triangle_ranges,
            ))
        })
        .collect()
}

#[cfg(feature = "plot-core")]
fn f64_to_f32_coordinate(value: f64) -> Result<f32, String> {
    if !value.is_finite() {
        return Err("geometry preview mesh contains a non-finite coordinate".to_string());
    }
    if value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err("geometry preview mesh coordinate exceeds f32 render range".to_string());
    }
    Ok(value as f32)
}

#[cfg(feature = "plot-core")]
fn preview_mesh_color(index: usize) -> glam::Vec4 {
    const PALETTE: [[f32; 4]; 6] = [
        [0.18, 0.48, 0.86, 1.0],
        [0.13, 0.62, 0.44, 1.0],
        [0.84, 0.43, 0.18, 1.0],
        [0.57, 0.38, 0.77, 1.0],
        [0.73, 0.62, 0.18, 1.0],
        [0.20, 0.62, 0.75, 1.0],
    ];
    glam::Vec4::from_array(PALETTE[index % PALETTE.len()])
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
        runmat_geometry_io::GeometryFormat::Iges => "iges",
        runmat_geometry_io::GeometryFormat::Brep => "brep",
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
        GeometryImportError::BackendUnavailable(_) => (
            "RM.GEOMETRY.LOAD.BACKEND_UNAVAILABLE",
            OperationErrorType::Backend,
            false,
        ),
        GeometryImportError::Cancelled => (
            "RM.GEOMETRY.LOAD.CANCELLED",
            OperationErrorType::Cancelled,
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
