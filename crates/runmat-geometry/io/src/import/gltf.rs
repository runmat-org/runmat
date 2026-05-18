use serde_json::Value;

use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};

use super::{
    build_asset, build_result, capacity_guard, is_degenerate_triangle, GeometryImportError,
    GeometryImportOptions,
};

pub(super) fn import_gltf(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    let value: Value = serde_json::from_slice(bytes).map_err(|err| {
        GeometryImportError::ParseFailed(format!("failed to decode GLTF JSON payload: {err}"))
    })?;

    let version = value
        .get("asset")
        .and_then(|asset| asset.get("version"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed("GLTF asset.version is required".to_string())
        })?;
    if !version.starts_with('2') {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF version '{}' is not supported; expected 2.x",
            version
        )));
    }

    let meshes = value
        .get("meshes")
        .and_then(Value::as_array)
        .ok_or_else(|| GeometryImportError::ParseFailed("GLTF meshes[] is required".to_string()))?;
    if meshes.is_empty() {
        return Err(GeometryImportError::ParseFailed(
            "GLTF meshes[] must not be empty".to_string(),
        ));
    }

    let mut diagnostics = Vec::<ImportDiagnostic>::new();
    let mut all_positions = Vec::<[f64; 3]>::new();
    let mut triangle_count = 0u64;

    for mesh in meshes {
        let primitives = mesh
            .get("primitives")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                GeometryImportError::ParseFailed("GLTF mesh.primitives[] is required".to_string())
            })?;
        for primitive in primitives {
            let positions = parse_inline_positions(primitive)?;
            let base_vertex = all_positions.len();
            all_positions.extend_from_slice(&positions);

            let indices = parse_inline_indices(primitive).map_err(|reason| {
                GeometryImportError::ParseFailed(format!(
                    "GLTF inline indices parse failed: {reason}"
                ))
            })?;
            if indices.len() % 3 != 0 {
                return Err(GeometryImportError::ParseFailed(
                    "GLTF inline indices must be a multiple of 3 for triangle primitives"
                        .to_string(),
                ));
            }
            for tri in indices.chunks_exact(3) {
                capacity_guard(triangle_count + 1, &options)?;
                let a = base_vertex + tri[0];
                let b = base_vertex + tri[1];
                let c = base_vertex + tri[2];
                if a >= all_positions.len() || b >= all_positions.len() || c >= all_positions.len()
                {
                    return Err(GeometryImportError::ParseFailed(
                        "GLTF inline index out of bounds for primitive positions".to_string(),
                    ));
                }
                let vertices = [all_positions[a], all_positions[b], all_positions[c]];
                if is_degenerate_triangle(&vertices) {
                    diagnostics.push(ImportDiagnostic {
                        code: "GEOMETRY_NORMALIZE_DEGENERATE_REMOVED".to_string(),
                        severity: ImportDiagnosticSeverity::Warning,
                        message: "Removed degenerate GLTF triangle during import".to_string(),
                    });
                } else {
                    triangle_count += 1;
                }
            }
        }
    }

    let asset = build_asset(
        path,
        "gltf/v1",
        options.units,
        all_positions.len() as u64,
        triangle_count,
        diagnostics.clone(),
    );
    Ok(build_result(asset, diagnostics))
}

fn parse_inline_positions(primitive: &Value) -> Result<Vec<[f64; 3]>, GeometryImportError> {
    let position_values = primitive
        .get("attributes")
        .and_then(|attributes| attributes.get("POSITION"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "GLTF inline POSITION array is required (accessor-backed payloads are not yet supported)"
                    .to_string(),
            )
        })?;
    if position_values.len() < 3 {
        return Err(GeometryImportError::ParseFailed(
            "GLTF POSITION must contain at least 3 vertices".to_string(),
        ));
    }

    position_values
        .iter()
        .map(|entry| {
            let coords = entry.as_array().ok_or_else(|| {
                GeometryImportError::ParseFailed("GLTF POSITION entry must be [x,y,z]".to_string())
            })?;
            if coords.len() < 3 {
                return Err(GeometryImportError::ParseFailed(
                    "GLTF POSITION entry missing coordinate components".to_string(),
                ));
            }
            let x = coords[0].as_f64().ok_or_else(|| {
                GeometryImportError::ParseFailed("GLTF POSITION x must be numeric".to_string())
            })?;
            let y = coords[1].as_f64().ok_or_else(|| {
                GeometryImportError::ParseFailed("GLTF POSITION y must be numeric".to_string())
            })?;
            let z = coords[2].as_f64().ok_or_else(|| {
                GeometryImportError::ParseFailed("GLTF POSITION z must be numeric".to_string())
            })?;
            Ok([x, y, z])
        })
        .collect()
}

fn parse_inline_indices(primitive: &Value) -> Result<Vec<usize>, String> {
    let Some(indices) = primitive.get("indices") else {
        let position_count = primitive
            .get("attributes")
            .and_then(|attributes| attributes.get("POSITION"))
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        return Ok((0..position_count).collect());
    };
    let values = indices
        .as_array()
        .ok_or_else(|| "GLTF inline indices must be an array".to_string())?;
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|index| index as usize)
                .ok_or_else(|| "GLTF inline index must be unsigned integer".to_string())
        })
        .collect()
}
