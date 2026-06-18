use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};
use runmat_geometry_core::SurfaceMesh;

use super::{
    build_asset, build_result, capacity_guard, check_cancelled_periodic, is_degenerate_triangle,
    parse_f64, push_mesh_count_diagnostics, BuildAssetInput, GeometryImportContext,
    GeometryImportError, GeometryImportOptions,
};

pub(super) fn import_stl(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    context.check_cancelled()?;
    if looks_like_binary_stl(bytes) {
        return import_binary_stl(path, bytes, options, context);
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| GeometryImportError::ParseFailed("invalid ASCII STL payload".to_string()))?;
    import_ascii_stl(path, text, options, context)
}

fn import_ascii_stl(
    path: &str,
    text: &str,
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    let mut diagnostics = Vec::<ImportDiagnostic>::new();
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut triangle_count = 0u64;
    let mut current = Vec::<[f64; 3]>::new();

    for (line_index, line) in text.lines().enumerate() {
        check_cancelled_periodic(context, line_index)?;
        let trimmed = line.trim();
        if !trimmed.starts_with("vertex ") {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let _ = parts.next();
        let x = parse_f64(parts.next().ok_or_else(|| {
            GeometryImportError::ParseFailed("missing vertex x component".to_string())
        })?)?;
        let y = parse_f64(parts.next().ok_or_else(|| {
            GeometryImportError::ParseFailed("missing vertex y component".to_string())
        })?)?;
        let z = parse_f64(parts.next().ok_or_else(|| {
            GeometryImportError::ParseFailed("missing vertex z component".to_string())
        })?)?;
        current.push([x, y, z]);

        if current.len() == 3 {
            let tri = [current[0], current[1], current[2]];
            capacity_guard(triangle_count + 1, &options)?;
            if is_degenerate_triangle(&tri) {
                diagnostics.push(ImportDiagnostic {
                    code: "GEOMETRY_NORMALIZE_DEGENERATE_REMOVED".to_string(),
                    severity: ImportDiagnosticSeverity::Warning,
                    message: "Removed degenerate STL triangle during import".to_string(),
                });
            } else {
                let base = u32::try_from(vertices.len()).map_err(|_| {
                    GeometryImportError::ParseFailed(
                        "STL vertex count exceeds render mesh index range".to_string(),
                    )
                })?;
                vertices.extend_from_slice(&tri);
                triangles.push([base, base + 1, base + 2]);
                triangle_count += 1;
            }
            current.clear();
        }
    }

    push_mesh_count_diagnostics(
        &mut diagnostics,
        "stl",
        vertices.len() as u64,
        triangle_count,
    );
    let asset = build_asset(BuildAssetInput {
        path,
        importer_version: "stl/v1",
        units: options.units,
        tessellation_profile: options.tessellation_profile.clone(),
        vertex_count: vertices.len() as u64,
        element_count: triangle_count,
        surface_meshes: vec![SurfaceMesh::new("mesh_1", vertices, triangles)],
        diagnostics: diagnostics.clone(),
    });
    Ok(build_result(asset, diagnostics))
}

fn import_binary_stl(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    if bytes.len() < 84 {
        return Err(GeometryImportError::ParseFailed(
            "binary STL payload is too small".to_string(),
        ));
    }
    let triangle_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let expected_size = 84usize + triangle_count.saturating_mul(50);
    if bytes.len() != expected_size {
        return Err(GeometryImportError::ParseFailed(format!(
            "binary STL size mismatch: expected {} bytes from header triangle count {}, got {}",
            expected_size,
            triangle_count,
            bytes.len()
        )));
    }

    let mut diagnostics = Vec::<ImportDiagnostic>::new();
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut accepted_triangles = 0u64;

    for index in 0..triangle_count {
        check_cancelled_periodic(context, index)?;
        let tri_offset = 84 + index * 50;
        let v0 = read_binary_vertex(bytes, tri_offset + 12)?;
        let v1 = read_binary_vertex(bytes, tri_offset + 24)?;
        let v2 = read_binary_vertex(bytes, tri_offset + 36)?;
        let tri = [v0, v1, v2];

        capacity_guard((index + 1) as u64, &options)?;
        if is_degenerate_triangle(&tri) {
            diagnostics.push(ImportDiagnostic {
                code: "GEOMETRY_NORMALIZE_DEGENERATE_REMOVED".to_string(),
                severity: ImportDiagnosticSeverity::Warning,
                message: "Removed degenerate binary STL triangle during import".to_string(),
            });
            continue;
        }

        let base = u32::try_from(vertices.len()).map_err(|_| {
            GeometryImportError::ParseFailed(
                "binary STL vertex count exceeds render mesh index range".to_string(),
            )
        })?;
        vertices.extend_from_slice(&tri);
        triangles.push([base, base + 1, base + 2]);
        accepted_triangles += 1;
    }

    push_mesh_count_diagnostics(
        &mut diagnostics,
        "stl",
        vertices.len() as u64,
        accepted_triangles,
    );
    let asset = build_asset(BuildAssetInput {
        path,
        importer_version: "stl/v1",
        units: options.units,
        tessellation_profile: options.tessellation_profile.clone(),
        vertex_count: vertices.len() as u64,
        element_count: accepted_triangles,
        surface_meshes: vec![SurfaceMesh::new("mesh_1", vertices, triangles)],
        diagnostics: diagnostics.clone(),
    });
    Ok(build_result(asset, diagnostics))
}

fn looks_like_binary_stl(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let triangle_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    84usize + triangle_count.saturating_mul(50) == bytes.len()
}

fn read_binary_vertex(bytes: &[u8], offset: usize) -> Result<[f64; 3], GeometryImportError> {
    let end = offset + 12;
    if end > bytes.len() {
        return Err(GeometryImportError::ParseFailed(
            "binary STL vertex data out of bounds".to_string(),
        ));
    }
    let x = f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]) as f64;
    let y = f32::from_le_bytes([
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ]) as f64;
    let z = f32::from_le_bytes([
        bytes[offset + 8],
        bytes[offset + 9],
        bytes[offset + 10],
        bytes[offset + 11],
    ]) as f64;
    Ok([x, y, z])
}
