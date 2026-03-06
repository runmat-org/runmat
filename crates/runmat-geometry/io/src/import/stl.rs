use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};

use super::{
    build_asset, build_result, capacity_guard, is_degenerate_triangle, parse_f64,
    GeometryImportError, GeometryImportOptions,
};

pub(super) fn import_stl(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        GeometryImportError::ParseFailed("binary STL not supported yet".to_string())
    })?;

    let mut diagnostics = Vec::<ImportDiagnostic>::new();
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut triangle_count = 0u64;
    let mut current = Vec::<[f64; 3]>::new();

    for line in text.lines() {
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
                vertices.extend_from_slice(&tri);
                triangle_count += 1;
            }
            current.clear();
        }
    }

    let asset = build_asset(
        path,
        "stl/v1",
        options.units,
        vertices.len() as u64,
        triangle_count,
        diagnostics.clone(),
    );
    Ok(build_result(asset, diagnostics))
}
