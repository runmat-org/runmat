mod gltf;
mod obj;
mod ply;
mod step;
mod stl;

use runmat_geometry_core::{
    GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, SourceGeometry, SourceGeometryKind,
    TessellationProfile, UnitSystem,
};
use thiserror::Error;

use crate::{
    report::{ImportDiagnostic, ImportDiagnosticSeverity, ImportReport, ImportResult},
    sniff::{detect_geometry_format, GeometryFormat},
};

#[derive(Debug, Clone)]
pub struct GeometryImportOptions {
    pub max_triangles: Option<u64>,
    pub units: UnitSystem,
}

impl Default for GeometryImportOptions {
    fn default() -> Self {
        Self {
            max_triangles: Some(16_000_000),
            units: UnitSystem::Meter,
        }
    }
}

#[derive(Debug, Error)]
pub enum GeometryImportError {
    #[error("GEOMETRY_FORMAT_UNSUPPORTED: unsupported geometry format")]
    UnsupportedFormat,
    #[error("GEOMETRY_PARSE_FAILED: {0}")]
    ParseFailed(String),
    #[error("CAPACITY_LIMIT_EXCEEDED: triangle count {triangles} exceeds limit {limit}")]
    CapacityExceeded { triangles: u64, limit: u64 },
}

pub fn import_geometry(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
) -> Result<ImportResult, GeometryImportError> {
    match detect_geometry_format(path, bytes) {
        GeometryFormat::Stl => stl::import_stl(path, bytes, options),
        GeometryFormat::Step => step::import_step(path, bytes, options),
        GeometryFormat::Obj => obj::import_obj(path, bytes, options),
        GeometryFormat::Ply => ply::import_ply(path, bytes, options),
        GeometryFormat::Gltf => gltf::import_gltf(path, bytes, options),
        _ => Err(GeometryImportError::UnsupportedFormat),
    }
}

pub(crate) fn build_asset(
    path: &str,
    importer_version: &str,
    units: UnitSystem,
    vertex_count: u64,
    element_count: u64,
    diagnostics: Vec<ImportDiagnostic>,
) -> GeometryAsset {
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
        tessellation_profile: TessellationProfile::default(),
        units,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: "mesh_1".to_string(),
            kind: MeshKind::Surface,
            vertex_count,
            element_count,
        }],
        regions: Vec::new(),
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
