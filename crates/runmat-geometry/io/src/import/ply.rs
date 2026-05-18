use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};

use super::{
    build_asset, build_result, capacity_guard, is_degenerate_triangle, parse_f64,
    GeometryImportError, GeometryImportOptions,
};

pub(super) fn import_ply(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| GeometryImportError::ParseFailed("invalid UTF-8 PLY payload".to_string()))?;
    let lines = text.lines().collect::<Vec<_>>();
    if lines.is_empty() || lines[0].trim() != "ply" {
        return Err(GeometryImportError::ParseFailed(
            "missing PLY header".to_string(),
        ));
    }

    let mut vertex_count = None;
    let mut face_count = None;
    let mut header_end = None;
    let mut ascii_format = false;
    for (index, line) in lines.iter().enumerate().skip(1) {
        let trimmed = line.trim();
        if trimmed == "end_header" {
            header_end = Some(index + 1);
            break;
        }
        if trimmed.starts_with("format ") {
            ascii_format = trimmed == "format ascii 1.0";
            continue;
        }
        if let Some(count) = trimmed.strip_prefix("element vertex ") {
            vertex_count = Some(parse_u64(count, "vertex count")? as usize);
            continue;
        }
        if let Some(count) = trimmed.strip_prefix("element face ") {
            face_count = Some(parse_u64(count, "face count")? as usize);
            continue;
        }
    }
    if !ascii_format {
        return Err(GeometryImportError::ParseFailed(
            "only ASCII PLY format 1.0 is supported".to_string(),
        ));
    }
    let header_end = header_end.ok_or_else(|| {
        GeometryImportError::ParseFailed("PLY missing end_header marker".to_string())
    })?;
    let vertex_count = vertex_count.ok_or_else(|| {
        GeometryImportError::ParseFailed("PLY missing vertex element count".to_string())
    })?;
    let face_count = face_count.ok_or_else(|| {
        GeometryImportError::ParseFailed("PLY missing face element count".to_string())
    })?;

    let mut diagnostics = Vec::<ImportDiagnostic>::new();
    let mut vertices = Vec::<[f64; 3]>::with_capacity(vertex_count);
    let mut triangle_count = 0u64;

    for vertex_line in lines.iter().skip(header_end).take(vertex_count) {
        let mut parts = vertex_line.split_whitespace();
        let x = parse_f64(parts.next().ok_or_else(|| {
            GeometryImportError::ParseFailed("PLY vertex line missing x component".to_string())
        })?)?;
        let y = parse_f64(parts.next().ok_or_else(|| {
            GeometryImportError::ParseFailed("PLY vertex line missing y component".to_string())
        })?)?;
        let z = parse_f64(parts.next().ok_or_else(|| {
            GeometryImportError::ParseFailed("PLY vertex line missing z component".to_string())
        })?)?;
        vertices.push([x, y, z]);
    }
    if vertices.len() != vertex_count {
        return Err(GeometryImportError::ParseFailed(format!(
            "PLY declared {} vertices but parsed {}",
            vertex_count,
            vertices.len()
        )));
    }

    let face_start = header_end + vertex_count;
    for face_line in lines.iter().skip(face_start).take(face_count) {
        let mut parts = face_line.split_whitespace();
        let face_vertex_count = parse_u64(
            parts.next().ok_or_else(|| {
                GeometryImportError::ParseFailed("PLY face line is empty".to_string())
            })?,
            "face vertex count",
        )? as usize;
        if face_vertex_count < 3 {
            return Err(GeometryImportError::ParseFailed(
                "PLY face requires at least 3 vertices".to_string(),
            ));
        }

        let mut indices = Vec::<usize>::with_capacity(face_vertex_count);
        for _ in 0..face_vertex_count {
            let raw = parts.next().ok_or_else(|| {
                GeometryImportError::ParseFailed("PLY face missing index entry".to_string())
            })?;
            let index = parse_u64(raw, "face index")? as usize;
            if index >= vertices.len() {
                return Err(GeometryImportError::ParseFailed(format!(
                    "PLY face index {} out of bounds for {} vertices",
                    index,
                    vertices.len()
                )));
            }
            indices.push(index);
        }

        let pivot = indices[0];
        for i in 1..(indices.len() - 1) {
            capacity_guard(triangle_count + 1, &options)?;
            let tri = [
                vertices[pivot],
                vertices[indices[i]],
                vertices[indices[i + 1]],
            ];
            if is_degenerate_triangle(&tri) {
                diagnostics.push(ImportDiagnostic {
                    code: "GEOMETRY_NORMALIZE_DEGENERATE_REMOVED".to_string(),
                    severity: ImportDiagnosticSeverity::Warning,
                    message: "Removed degenerate PLY face triangle during import".to_string(),
                });
            } else {
                triangle_count += 1;
            }
        }
    }

    let asset = build_asset(
        path,
        "ply/v1",
        options.units,
        vertices.len() as u64,
        triangle_count,
        diagnostics.clone(),
    );
    Ok(build_result(asset, diagnostics))
}

fn parse_u64(value: &str, field: &str) -> Result<u64, GeometryImportError> {
    value.parse::<u64>().map_err(|_| {
        GeometryImportError::ParseFailed(format!("invalid PLY {} value: {}", field, value))
    })
}
