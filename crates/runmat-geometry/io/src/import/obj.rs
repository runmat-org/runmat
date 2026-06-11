use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};
use runmat_geometry_core::SurfaceMesh;

use super::{
    build_asset, build_result, capacity_guard, is_degenerate_triangle, parse_f64,
    push_mesh_count_diagnostics, push_utf8_bom_stripped_diagnostic, strip_utf8_bom_text,
    GeometryImportError, GeometryImportOptions,
};

pub(super) fn import_obj(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| GeometryImportError::ParseFailed("invalid UTF-8 OBJ payload".to_string()))?;

    let mut diagnostics = Vec::<ImportDiagnostic>::new();
    let (text, stripped_bom) = strip_utf8_bom_text(text);
    if stripped_bom {
        push_utf8_bom_stripped_diagnostic(&mut diagnostics, "obj");
    }
    let mut vertex_pool = Vec::<[f64; 3]>::new();
    let mut triangles = Vec::<[u32; 3]>::new();
    let mut triangle_count = 0u64;

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("v ") {
            let mut parts = trimmed.split_whitespace();
            let _ = parts.next();
            let x = parse_f64(parts.next().ok_or_else(|| {
                GeometryImportError::ParseFailed(format!(
                    "missing OBJ vertex x component at line {}",
                    line_idx + 1
                ))
            })?)?;
            let y = parse_f64(parts.next().ok_or_else(|| {
                GeometryImportError::ParseFailed(format!(
                    "missing OBJ vertex y component at line {}",
                    line_idx + 1
                ))
            })?)?;
            let z = parse_f64(parts.next().ok_or_else(|| {
                GeometryImportError::ParseFailed(format!(
                    "missing OBJ vertex z component at line {}",
                    line_idx + 1
                ))
            })?)?;
            vertex_pool.push([x, y, z]);
            continue;
        }
        if !trimmed.starts_with("f ") {
            continue;
        }

        let mut face_indices = Vec::<usize>::new();
        for token in trimmed.split_whitespace().skip(1) {
            let index = parse_face_index(token, vertex_pool.len()).map_err(|reason| {
                GeometryImportError::ParseFailed(format!(
                    "invalid OBJ face index at line {}: {}",
                    line_idx + 1,
                    reason
                ))
            })?;
            face_indices.push(index);
        }

        if face_indices.len() < 3 {
            return Err(GeometryImportError::ParseFailed(format!(
                "OBJ face requires at least 3 vertices at line {}",
                line_idx + 1
            )));
        }

        let pivot = face_indices[0];
        for i in 1..(face_indices.len() - 1) {
            let tri = [pivot, face_indices[i], face_indices[i + 1]];
            capacity_guard(triangle_count + 1, &options)?;

            let vertices = [
                vertex_pool[tri[0]],
                vertex_pool[tri[1]],
                vertex_pool[tri[2]],
            ];
            if is_degenerate_triangle(&vertices) {
                diagnostics.push(ImportDiagnostic {
                    code: "GEOMETRY_NORMALIZE_DEGENERATE_REMOVED".to_string(),
                    severity: ImportDiagnosticSeverity::Warning,
                    message: "Removed degenerate OBJ face triangle during import".to_string(),
                });
            } else {
                triangles.push([
                    u32::try_from(tri[0]).map_err(|_| {
                        GeometryImportError::ParseFailed(
                            "OBJ vertex index exceeds render mesh index range".to_string(),
                        )
                    })?,
                    u32::try_from(tri[1]).map_err(|_| {
                        GeometryImportError::ParseFailed(
                            "OBJ vertex index exceeds render mesh index range".to_string(),
                        )
                    })?,
                    u32::try_from(tri[2]).map_err(|_| {
                        GeometryImportError::ParseFailed(
                            "OBJ vertex index exceeds render mesh index range".to_string(),
                        )
                    })?,
                ]);
                triangle_count += 1;
            }
        }
    }

    push_mesh_count_diagnostics(
        &mut diagnostics,
        "obj",
        vertex_pool.len() as u64,
        triangle_count,
    );
    let asset = build_asset(
        path,
        "obj/v1",
        options.units,
        vertex_pool.len() as u64,
        triangle_count,
        vec![SurfaceMesh::new("mesh_1", vertex_pool, triangles)],
        diagnostics.clone(),
    );
    Ok(build_result(asset, diagnostics))
}

fn parse_face_index(token: &str, vertex_count: usize) -> Result<usize, String> {
    if vertex_count == 0 {
        return Err("face declared before vertices".to_string());
    }
    let raw_index = token
        .split('/')
        .next()
        .ok_or_else(|| "missing face index token".to_string())?;
    let index = raw_index
        .parse::<isize>()
        .map_err(|_| format!("invalid face index '{}'", raw_index))?;
    if index == 0 {
        return Err("OBJ indices are 1-based and cannot be zero".to_string());
    }

    let resolved = if index > 0 {
        index - 1
    } else {
        vertex_count as isize + index
    };
    if resolved < 0 || resolved >= vertex_count as isize {
        return Err(format!(
            "face index '{}' resolves out of bounds for {} vertices",
            raw_index, vertex_count
        ));
    }
    Ok(resolved as usize)
}

#[cfg(test)]
mod tests {
    use super::parse_face_index;

    #[test]
    fn parse_face_index_supports_positive_and_negative_indices() {
        assert_eq!(parse_face_index("1/2/3", 4).expect("positive index"), 0);
        assert_eq!(parse_face_index("-1", 4).expect("negative index"), 3);
    }
}
