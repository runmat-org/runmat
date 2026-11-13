pub(crate) mod cad;
mod gltf;
mod obj;
mod ply;
mod stl;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use runmat_geometry_core::{
    GeometryAsset, GeometrySource, LengthUnit, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
};
use thiserror::Error;

use crate::{
    report::{ImportDiagnostic, ImportDiagnosticSeverity, ImportReport, ImportResult},
    sniff::{detect_geometry_format, GeometryFormat},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LabeledSubshapeRemapConflictKind {
    Deleted,
    Split,
    Merged,
    MissingTarget,
    AmbiguousTarget,
    UnsupportedKind,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LabeledSubshapeRemapConflict {
    pub kind: LabeledSubshapeRemapConflictKind,
    pub label_entries: Vec<String>,
    pub source_topology_ids: Vec<String>,
    pub candidate_topology_ids: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryImportOptions {
    pub max_triangles: Option<u64>,
    pub budget_policy: GeometryImportBudgetPolicy,
    pub units: LengthUnit,
    #[serde(default)]
    pub source_units: Option<LengthUnit>,
    pub tessellation_profile: TessellationProfile,
    pub relative_deflection: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GeometryImportBudgetPolicy {
    Strict,
    Truncate,
}

impl Default for GeometryImportOptions {
    fn default() -> Self {
        Self {
            max_triangles: Some(16_000_000),
            budget_policy: GeometryImportBudgetPolicy::Strict,
            units: LengthUnit::Meter,
            source_units: None,
            tessellation_profile: TessellationProfile::default(),
            relative_deflection: false,
        }
    }
}

#[derive(Clone, Default)]
pub struct GeometryImportContext {
    cancellation: Option<CancellationProbe>,
}

pub(crate) type CancellationProbe = Arc<dyn Fn() -> bool + Send + Sync>;

impl std::fmt::Debug for GeometryImportContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeometryImportContext")
            .field("cancellable", &self.cancellation.is_some())
            .finish()
    }
}

impl GeometryImportContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cancellation(cancellation: Arc<AtomicBool>) -> Self {
        Self::with_cancellation_probe(move || cancellation.load(Ordering::Relaxed))
    }

    /// Cooperative synchronous import boundary. The probe does not register
    /// wakeups or interrupt a noncooperative native geometry operation.
    pub fn with_cancellation_probe(probe: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self {
            cancellation: Some(Arc::new(probe)),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .map(|is_cancelled| is_cancelled())
            .unwrap_or(false)
    }

    pub fn check_cancelled(&self) -> Result<(), GeometryImportError> {
        if self.is_cancelled() {
            Err(GeometryImportError::Cancelled)
        } else {
            Ok(())
        }
    }

    #[cfg(all(
        any(not(target_arch = "wasm32"), target_os = "emscripten"),
        feature = "occt-native"
    ))]
    pub(crate) fn cancellation_probe(&self) -> Option<CancellationProbe> {
        self.cancellation.clone()
    }
}

#[derive(Debug, Error, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "detail", rename_all = "snake_case")]
pub enum GeometryImportError {
    #[error("GEOMETRY_FORMAT_UNSUPPORTED: unsupported geometry format")]
    UnsupportedFormat,
    #[error("GEOMETRY_PARSE_FAILED: {0}")]
    ParseFailed(String),
    #[error("GEOMETRY_BACKEND_UNAVAILABLE: {0}")]
    BackendUnavailable(String),
    #[error("GEOMETRY_INVALID: {0}")]
    InvalidGeometry(String),
    #[error("CAPACITY_LIMIT_EXCEEDED: triangle count {triangles} exceeds limit {limit}")]
    CapacityExceeded { triangles: u64, limit: u64 },
    #[error("EXACT_REPRESENTATION_LIMIT_EXCEEDED: representation exceeds {limit} bytes")]
    ExactRepresentationCapacityExceeded { limit: u64 },
    #[error("EXACT_ENTITY_LIMIT_EXCEEDED: a topology inventory exceeds {limit} entities")]
    ExactEntityCapacityExceeded { limit: u64 },
    #[error("EXACT_VALIDATION_BUDGET_EXCEEDED: {0}")]
    ExactValidationBudgetExceeded(String),
    #[error("GEOMETRY_HEALING_LIMIT_EXCEEDED: {failure:?}")]
    HealingLimitExceeded {
        failure: runmat_geometry_core::GeometryHealingFailure,
    },
    #[error("GEOMETRY_REVISION_CONFLICT: {conflict:?}")]
    RevisionConflict {
        conflict: LabeledSubshapeRemapConflict,
    },
    #[error("GEOMETRY_IMPORT_OPTIONS_INVALID: {0}")]
    InvalidOptions(String),
    /// Authored units cannot be applied because the file does not declare a single length unit
    /// (none, or several different ones).
    #[error("GEOMETRY_UNITS_UNRESOLVED: {0}")]
    UnresolvedUnits(String),
    #[error("GEOMETRY_IMPORT_CANCELLED: geometry import cancelled")]
    Cancelled,
}

pub fn import_geometry(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
) -> Result<ImportResult, GeometryImportError> {
    import_geometry_with_context(path, bytes, options, &GeometryImportContext::new())
}

pub fn import_geometry_with_context(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<ImportResult, GeometryImportError> {
    import_geometry_with_selection_projection(path, bytes, options, context, None)
}

/// Tessellate an already admitted exact geometry without assigning it a second identity.
/// The supplied topology and source-face join must come from the same materialization.
pub fn import_materialized_geometry_preview(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: &GeometryImportContext,
    topology: &runmat_geometry_core::ExactBRepTopology,
    source_faces: &std::collections::BTreeMap<u64, runmat_geometry_core::PersistentEntityId>,
) -> Result<ImportResult, GeometryImportError> {
    let selection = crate::ExactSelectionProjection::new(topology, source_faces);
    import_geometry_with_selection_projection(path, bytes, options, context, Some(&selection))
}

fn import_geometry_with_selection_projection(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: &GeometryImportContext,
    exact_selection: Option<&crate::ExactSelectionProjection>,
) -> Result<ImportResult, GeometryImportError> {
    context.check_cancelled()?;
    let format = detect_geometry_format(path, bytes);
    let unit_evidence = crate::source_unit_evidence(path, bytes)
        .resolve_fallback(options.source_units, "explicit preview-import source units");
    let target_units = options.units;
    let result = match format {
        GeometryFormat::Stl => stl::import_stl(path, bytes, options, context),
        GeometryFormat::Step | GeometryFormat::Iges | GeometryFormat::Brep => {
            #[cfg(all(target_arch = "wasm32", feature = "occt-wasm-host"))]
            if let Some(imported) =
                crate::occt::import_geometry_host(path, bytes, &options, context)?
            {
                if imported.asset.source.unit_evidence != unit_evidence {
                    return Err(GeometryImportError::InvalidGeometry(
                        "OCCT host source-unit evidence differs from the input file".into(),
                    ));
                }
                return Ok(imported);
            }
            cad::import_cad(path, bytes, format, options, context, exact_selection)
        }
        GeometryFormat::Obj => obj::import_obj(path, bytes, options, context),
        GeometryFormat::Ply => ply::import_ply(path, bytes, options, context),
        GeometryFormat::Gltf => gltf::import_gltf(path, bytes, options, context),
        _ => Err(GeometryImportError::UnsupportedFormat),
    };
    context.check_cancelled()?;
    result.and_then(|mut imported| {
        // Faceted readers return raw source coordinates. Convert only when the
        // source scale is known; unitless previews remain explicitly unresolved.
        if matches!(
            format,
            GeometryFormat::Stl | GeometryFormat::Obj | GeometryFormat::Ply | GeometryFormat::Gltf
        ) {
            if let Some(source_unit) = unit_evidence.uniform_unit() {
                let scale = source_unit.meters_per_unit() / target_units.meters_per_unit();
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(GeometryImportError::InvalidOptions(
                        "source-to-preview coordinate scale is invalid".into(),
                    ));
                }
                for mesh in &mut imported.asset.surface_meshes {
                    for (index, vertex) in mesh.vertices.iter_mut().enumerate() {
                        check_cancelled_periodic(context, index)?;
                        for component in vertex {
                            *component *= scale;
                        }
                    }
                }
            }
        }
        imported.asset.coordinate_units = unit_evidence
            .physical_scale_resolved()
            .then(|| runmat_geometry_core::physical::CoordinateUnits::new(target_units));
        imported.asset.source.unit_evidence = unit_evidence;
        imported
            .asset
            .validate()
            .map_err(|message| GeometryImportError::InvalidGeometry(message.into()))?;
        Ok(imported)
    })
}

pub(crate) fn check_cancelled_periodic(
    context: &GeometryImportContext,
    index: usize,
) -> Result<(), GeometryImportError> {
    if index & 0x3ff == 0 {
        context.check_cancelled()?;
    }
    Ok(())
}

pub(crate) struct BuildAssetInput<'a> {
    pub path: &'a str,
    pub importer_version: &'a str,
    pub units: LengthUnit,
    pub tessellation_profile: TessellationProfile,
    pub vertex_count: u64,
    pub element_count: u64,
    pub surface_meshes: Vec<SurfaceMesh>,
    pub diagnostics: Vec<ImportDiagnostic>,
}

pub(crate) fn build_asset(input: BuildAssetInput<'_>) -> GeometryAsset {
    let BuildAssetInput {
        path,
        importer_version,
        units,
        tessellation_profile,
        vertex_count,
        element_count,
        surface_meshes,
        diagnostics,
    } = input;
    let (regions, region_entity_mappings) = default_surface_regions(&surface_meshes);

    GeometryAsset {
        geometry_id: format!("geo:{}", path),
        source: GeometrySource {
            unit_evidence: Default::default(),
            path: path.to_string(),
            sha256: String::new(),
            importer_version: importer_version.to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Mesh,
            assembly: None,
            material_evidence: Vec::new(),
            exchange: None,
            cad_evaluators: Vec::new(),
        },
        tessellation_profile,
        coordinate_units: Some(runmat_geometry_core::physical::CoordinateUnits::new(units)),
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: "mesh_1".to_string(),
            kind: MeshKind::Surface,
            vertex_count,
            element_count,
        }],
        surface_meshes,
        regions,
        region_entity_mappings,
        diagnostics: diagnostics
            .iter()
            .map(|item| runmat_geometry_core::Diagnostic {
                code: item.code.clone(),
                severity: match item.severity {
                    ImportDiagnosticSeverity::Info => {
                        runmat_geometry_core::DiagnosticSeverity::Info
                    }
                    ImportDiagnosticSeverity::Warning => {
                        runmat_geometry_core::DiagnosticSeverity::Warning
                    }
                    ImportDiagnosticSeverity::Error => {
                        runmat_geometry_core::DiagnosticSeverity::Error
                    }
                },
                message: item.message.clone(),
            })
            .collect(),
    }
}

pub(crate) fn default_surface_regions(
    surface_meshes: &[SurfaceMesh],
) -> (Vec<Region>, Vec<RegionEntityMapping>) {
    if surface_meshes.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let region_id = "region_default".to_string();
    let regions = vec![Region {
        region_id: region_id.clone(),
        name: "Default Region".into(),
        tag: Some("mesh_default".to_string()),
        cad_ownership: None,
    }];
    let mappings = surface_meshes
        .iter()
        .map(|mesh| {
            RegionEntityMapping::all_faces(
                region_id.clone(),
                mesh.mesh_id.clone(),
                mesh.triangles.len() as u64,
            )
        })
        .collect();
    (regions, mappings)
}

pub(crate) fn push_entity_range(ranges: &mut Vec<runmat_geometry_core::EntityIdRange>, id: u64) {
    if let Some(last) = ranges.last_mut() {
        if last.end_exclusive() == Some(id) {
            last.count += 1;
            return;
        }
    }
    ranges.push(runmat_geometry_core::EntityIdRange::new(id, 1));
}

pub(crate) fn build_result(
    asset: GeometryAsset,
    diagnostics: Vec<ImportDiagnostic>,
) -> ImportResult {
    let normalized = diagnostics
        .iter()
        .any(|item| item.code == "GEOMETRY_NORMALIZE_DEGENERATE_REMOVED");
    ImportResult {
        asset,
        report: ImportReport {
            diagnostics,
            normalized,
        },
    }
}

pub(crate) fn push_mesh_count_diagnostics(
    diagnostics: &mut Vec<ImportDiagnostic>,
    format_name: &str,
    vertex_count: u64,
    triangle_count: u64,
) {
    diagnostics.push(ImportDiagnostic {
        code: "GEOMETRY_IMPORT_VERTEX_COUNT".to_string(),
        severity: ImportDiagnosticSeverity::Info,
        message: format!("{format_name} import resolved {vertex_count} mesh vertices"),
    });
    diagnostics.push(ImportDiagnostic {
        code: "GEOMETRY_IMPORT_TRIANGLE_COUNT".to_string(),
        severity: ImportDiagnosticSeverity::Info,
        message: format!("{format_name} import resolved {triangle_count} mesh triangles"),
    });
}

pub(crate) fn strip_utf8_bom_bytes(bytes: &[u8]) -> (&[u8], bool) {
    const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
    if bytes.starts_with(UTF8_BOM) {
        (&bytes[UTF8_BOM.len()..], true)
    } else {
        (bytes, false)
    }
}

pub(crate) fn strip_utf8_bom_text(text: &str) -> (&str, bool) {
    if let Some(stripped) = text.strip_prefix('\u{feff}') {
        (stripped, true)
    } else {
        (text, false)
    }
}

pub(crate) fn push_utf8_bom_stripped_diagnostic(
    diagnostics: &mut Vec<ImportDiagnostic>,
    format_name: &str,
) {
    diagnostics.push(ImportDiagnostic {
        code: "GEOMETRY_IMPORT_UTF8_BOM_STRIPPED".to_string(),
        severity: ImportDiagnosticSeverity::Info,
        message: format!("{format_name} import stripped UTF-8 BOM prefix before parsing payload"),
    });
}

pub(crate) fn capacity_guard(
    triangle_count: u64,
    options: &GeometryImportOptions,
) -> Result<(), GeometryImportError> {
    if let Some(limit) = options.max_triangles {
        if triangle_count > limit {
            return Err(GeometryImportError::CapacityExceeded {
                triangles: triangle_count,
                limit,
            });
        }
    }
    Ok(())
}

pub(crate) fn parse_f64(value: &str) -> Result<f64, GeometryImportError> {
    value
        .parse::<f64>()
        .map_err(|_| GeometryImportError::ParseFailed(format!("invalid number: {value}")))
}

pub(crate) fn is_degenerate_triangle(vertices: &[[f64; 3]; 3]) -> bool {
    vertices[0] == vertices[1] || vertices[1] == vertices[2] || vertices[0] == vertices[2]
}

#[cfg(test)]
mod control_tests {
    use super::*;
    #[test]
    fn import_control_follows_its_live_caller_probe() {
        let caller = Arc::new(AtomicBool::new(false));
        let context = GeometryImportContext::with_cancellation(caller.clone());
        assert!(!context.is_cancelled());
        caller.store(true, Ordering::Release);
        assert!(matches!(
            context.check_cancelled(),
            Err(GeometryImportError::Cancelled)
        ));
    }
}
