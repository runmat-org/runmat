use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

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
const DEFAULT_QUERY_LIMIT: usize = 2048;

#[derive(Debug, Clone)]
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

type PrepStore = Arc<RwLock<HashMap<String, StoredGeometryPrepArtifact>>>;

fn prep_store() -> &'static PrepStore {
    static STORE: OnceLock<PrepStore> = OnceLock::new();
    STORE.get_or_init(|| Arc::new(RwLock::new(HashMap::new())))
}

fn prep_artifact_counter() -> &'static AtomicU64 {
    static COUNTER: OnceLock<AtomicU64> = OnceLock::new();
    COUNTER.get_or_init(|| AtomicU64::new(1))
}

fn prep_artifact_root() -> Option<PathBuf> {
    std::env::var("RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT")
        .ok()
        .map(PathBuf::from)
}

fn prep_artifact_path(root: &PathBuf, prep_artifact_id: &str) -> PathBuf {
    root.join("prep").join(format!("{prep_artifact_id}.json"))
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

    if let Some(root) = prep_artifact_root() {
        let path = prep_artifact_path(&root, &prep_artifact_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create prep artifact directory: {err}"))?;
        }
        let bytes = serde_json::to_vec_pretty(&artifact)
            .map_err(|err| format!("failed to encode prep artifact: {err}"))?;
        fs::write(&path, bytes).map_err(|err| format!("failed to write prep artifact: {err}"))?;
    }

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
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path).map_err(|err| format!("failed to read prep artifact: {err}"))?;
    let artifact = serde_json::from_slice::<StoredGeometryPrepArtifact>(&bytes)
        .map_err(|err| format!("failed to decode prep artifact: {err}"))?;
    prep_store()
        .write()
        .map_err(|_| "geometry prep artifact store lock poisoned".to_string())?
        .insert(prep_artifact_id.to_string(), artifact.clone());
    Ok(Some(artifact))
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
    let imported = import_geometry(path, bytes, GeometryImportOptions::default())
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
                error_code: "GEOMETRY_QUERY_INVALID_LIMIT",
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
                error_code: "GEOMETRY_CAPTURE_INVALID_SPEC",
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
                    error_code: "GEOMETRY_CAPTURE_BACKEND_FAILED",
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
                        error_code: "GEOMETRY_CAPTURE_BACKEND_FAILED",
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
            error_code: "GEOMETRY_CAPTURE_UNSUPPORTED",
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
                error_code: "GEOMETRY_PREP_INVALID_SPEC",
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
                error_code: "GEOMETRY_PREP_FAILED",
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
                error_code: "GEOMETRY_PREP_ARTIFACT_STORE_FAILED",
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
            "GEOMETRY_FORMAT_UNSUPPORTED",
            OperationErrorType::Input,
            false,
        ),
        GeometryImportError::ParseFailed(_) => (
            "GEOMETRY_PARSE_FAILED",
            OperationErrorType::Validation,
            false,
        ),
        GeometryImportError::CapacityExceeded { .. } => (
            "CAPACITY_LIMIT_EXCEEDED",
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
                error_code: "GEOMETRY_REGION_NOT_FOUND",
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
