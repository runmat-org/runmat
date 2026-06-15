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
    GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region, RegionEntityMapping,
    SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
};
use thiserror::Error;

use crate::{
    report::{ImportDiagnostic, ImportDiagnosticSeverity, ImportReport, ImportResult},
    sniff::{detect_geometry_format, GeometryFormat},
};

#[derive(Debug, Clone)]
pub struct GeometryImportOptions {
    pub max_triangles: Option<u64>,
    pub budget_policy: GeometryImportBudgetPolicy,
    pub units: UnitSystem,
    pub tessellation_profile: TessellationProfile,
    pub relative_deflection: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryImportBudgetPolicy {
    Strict,
    Truncate,
}

impl Default for GeometryImportOptions {
    fn default() -> Self {
        Self {
            max_triangles: Some(16_000_000),
            budget_policy: GeometryImportBudgetPolicy::Strict,
            units: UnitSystem::Meter,
            tessellation_profile: TessellationProfile::default(),
            relative_deflection: false,
        }
    }
}

#[derive(Clone, Default)]
pub struct GeometryImportContext {
    cancellation: Option<Arc<AtomicBool>>,
}

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
        Self {
            cancellation: Some(cancellation),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .map(|flag| flag.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    pub fn check_cancelled(&self) -> Result<(), GeometryImportError> {
        if self.is_cancelled() {
            Err(GeometryImportError::Cancelled)
        } else {
            Ok(())
        }
    }

    #[cfg(any(
        all(not(target_arch = "wasm32"), feature = "occt-native"),
        all(target_arch = "wasm32", feature = "occt-wasm-host")
    ))]
    pub(crate) fn cancellation_flag(&self) -> Option<Arc<AtomicBool>> {
        self.cancellation.clone()
    }
}

#[derive(Debug, Error)]
pub enum GeometryImportError {
    #[error("GEOMETRY_FORMAT_UNSUPPORTED: unsupported geometry format")]
    UnsupportedFormat,
    #[error("GEOMETRY_PARSE_FAILED: {0}")]
    ParseFailed(String),
    #[error("GEOMETRY_BACKEND_UNAVAILABLE: {0}")]
    BackendUnavailable(String),
    #[error("CAPACITY_LIMIT_EXCEEDED: triangle count {triangles} exceeds limit {limit}")]
    CapacityExceeded { triangles: u64, limit: u64 },
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
    context.check_cancelled()?;
    let format = detect_geometry_format(path, bytes);
    let result = match format {
        GeometryFormat::Stl => stl::import_stl(path, bytes, options, context),
        GeometryFormat::Step | GeometryFormat::Iges | GeometryFormat::Brep => {
            cad::import_cad(path, bytes, format, options, context)
        }
        GeometryFormat::Obj => obj::import_obj(path, bytes, options, context),
        GeometryFormat::Ply => ply::import_ply(path, bytes, options, context),
        GeometryFormat::Gltf => gltf::import_gltf(path, bytes, options, context),
        _ => Err(GeometryImportError::UnsupportedFormat),
    };
    context.check_cancelled()?;
    result
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

pub(crate) fn build_asset(
    path: &str,
    importer_version: &str,
    units: UnitSystem,
    tessellation_profile: TessellationProfile,
    vertex_count: u64,
    element_count: u64,
    surface_meshes: Vec<SurfaceMesh>,
    diagnostics: Vec<ImportDiagnostic>,
) -> GeometryAsset {
    let (regions, region_entity_mappings) = default_surface_regions(&surface_meshes);

    GeometryAsset {
        geometry_id: format!("geo:{}", path),
        source: GeometrySource {
            path: path.to_string(),
            sha256: String::new(),
            importer_version: importer_version.to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Mesh,
            assembly: None,
            material_evidence: Vec::new(),
        },
        tessellation_profile,
        units,
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
        name: "Default Region".to_string(),
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
