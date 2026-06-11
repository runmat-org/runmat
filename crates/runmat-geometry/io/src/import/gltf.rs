use base64::engine::general_purpose::STANDARD as BASE64_ENGINE;
use base64::Engine;
use runmat_geometry_core::SurfaceMesh;
use serde_json::Value;

use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};

use super::{
    build_asset, build_result, capacity_guard, is_degenerate_triangle, push_mesh_count_diagnostics,
    push_utf8_bom_stripped_diagnostic, strip_utf8_bom_bytes, GeometryImportError,
    GeometryImportOptions,
};

pub(super) fn import_gltf(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    let (bytes, stripped_bom) = strip_utf8_bom_bytes(bytes);
    if bytes.len() >= 4 && &bytes[0..4] == b"glTF" {
        return Err(GeometryImportError::ParseFailed(
            "binary GLB payloads are not supported yet; provide JSON GLTF inline payload"
                .to_string(),
        ));
    }
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
    if stripped_bom {
        push_utf8_bom_stripped_diagnostic(&mut diagnostics, "gltf");
    }
    let mut all_positions = Vec::<[f64; 3]>::new();
    let mut triangles = Vec::<[u32; 3]>::new();
    let mut triangle_count = 0u64;

    for mesh in meshes {
        let primitives = mesh
            .get("primitives")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                GeometryImportError::ParseFailed("GLTF mesh.primitives[] is required".to_string())
            })?;
        for primitive in primitives {
            let mode = primitive.get("mode").and_then(Value::as_u64).unwrap_or(4);
            if mode != 4 {
                return Err(GeometryImportError::ParseFailed(format!(
                    "GLTF primitive mode {} is not supported; only mode 4 (TRIANGLES) is allowed",
                    mode
                )));
            }
            let (positions, uses_accessor_data_uri) = parse_positions(&value, primitive)?;
            let uses_implicit_indices = primitive.get("indices").is_none();
            let base_vertex = all_positions.len();
            all_positions.extend_from_slice(&positions);

            let (indices, indices_use_accessor_data_uri) =
                parse_indices(&value, primitive, positions.len())?;
            if uses_accessor_data_uri || indices_use_accessor_data_uri {
                diagnostics.push(ImportDiagnostic {
                    code: "GEOMETRY_GLTF_ACCESSOR_DATA_URI_USED".to_string(),
                    severity: ImportDiagnosticSeverity::Info,
                    message: "GLTF primitive used accessor-backed data-URI buffers for deterministic mesh import".to_string(),
                });
            }
            if uses_implicit_indices {
                diagnostics.push(ImportDiagnostic {
                    code: "GEOMETRY_GLTF_IMPLICIT_INDICES_USED".to_string(),
                    severity: ImportDiagnosticSeverity::Info,
                    message: format!(
                        "GLTF primitive omitted indices; generated deterministic 0..{} index sequence",
                        positions.len().saturating_sub(1)
                    ),
                });
            }
            if indices.len() % 3 != 0 {
                return Err(GeometryImportError::ParseFailed(
                    "GLTF indices must be a multiple of 3 for triangle primitives".to_string(),
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
                        "GLTF index out of bounds for primitive positions".to_string(),
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
                    triangles.push([
                        u32::try_from(a).map_err(|_| {
                            GeometryImportError::ParseFailed(
                                "GLTF vertex index exceeds render mesh index range".to_string(),
                            )
                        })?,
                        u32::try_from(b).map_err(|_| {
                            GeometryImportError::ParseFailed(
                                "GLTF vertex index exceeds render mesh index range".to_string(),
                            )
                        })?,
                        u32::try_from(c).map_err(|_| {
                            GeometryImportError::ParseFailed(
                                "GLTF vertex index exceeds render mesh index range".to_string(),
                            )
                        })?,
                    ]);
                    triangle_count += 1;
                }
            }
        }
    }

    push_mesh_count_diagnostics(
        &mut diagnostics,
        "gltf",
        all_positions.len() as u64,
        triangle_count,
    );
    let asset = build_asset(
        path,
        "gltf/v1",
        options.units,
        all_positions.len() as u64,
        triangle_count,
        vec![SurfaceMesh::new("mesh_1", all_positions, triangles)],
        diagnostics.clone(),
    );
    Ok(build_result(asset, diagnostics))
}

fn parse_positions(
    root: &Value,
    primitive: &Value,
) -> Result<(Vec<[f64; 3]>, bool), GeometryImportError> {
    let position_ref = primitive
        .get("attributes")
        .and_then(|attributes| attributes.get("POSITION"))
        .ok_or_else(|| {
            GeometryImportError::ParseFailed("GLTF POSITION attribute is required".to_string())
        })?;
    if let Some(position_values) = position_ref.as_array() {
        return Ok((parse_inline_positions(position_values)?, false));
    }
    if let Some(accessor_index) = position_ref.as_u64() {
        return Ok((parse_accessor_positions(root, accessor_index)?, true));
    }
    Err(GeometryImportError::ParseFailed(
        "GLTF POSITION attribute must be an inline array or accessor index".to_string(),
    ))
}

fn parse_inline_positions(position_values: &[Value]) -> Result<Vec<[f64; 3]>, GeometryImportError> {
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

fn parse_indices(
    root: &Value,
    primitive: &Value,
    position_count: usize,
) -> Result<(Vec<usize>, bool), GeometryImportError> {
    let Some(indices) = primitive.get("indices") else {
        return Ok(((0..position_count).collect(), false));
    };
    if let Some(values) = indices.as_array() {
        return Ok((parse_inline_indices(values)?, false));
    }
    if let Some(accessor_index) = indices.as_u64() {
        return Ok((parse_accessor_indices(root, accessor_index)?, true));
    }
    Err(GeometryImportError::ParseFailed(
        "GLTF indices must be an inline array or accessor index".to_string(),
    ))
}

fn parse_inline_indices(values: &[Value]) -> Result<Vec<usize>, GeometryImportError> {
    values
        .iter()
        .map(|value| {
            let index = value.as_u64().ok_or_else(|| {
                GeometryImportError::ParseFailed(
                    "GLTF inline index must be unsigned integer".to_string(),
                )
            })?;
            usize::try_from(index).map_err(|_| {
                GeometryImportError::ParseFailed(
                    "GLTF inline index exceeds host addressable range".to_string(),
                )
            })
        })
        .collect()
}

fn parse_accessor_positions(
    root: &Value,
    accessor_index: u64,
) -> Result<Vec<[f64; 3]>, GeometryImportError> {
    let decoded = resolve_accessor_decode(root, accessor_index)?;
    if decoded.accessor_type != "VEC3" {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF POSITION accessor type '{}' is not supported; expected VEC3",
            decoded.accessor_type
        )));
    }
    if decoded.component_type != 5126 {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF POSITION accessor componentType {} is not supported; expected 5126 (FLOAT)",
            decoded.component_type
        )));
    }
    if decoded.count < 3 {
        return Err(GeometryImportError::ParseFailed(
            "GLTF POSITION accessor must contain at least 3 vertices".to_string(),
        ));
    }
    if decoded.stride < 12 {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF POSITION accessor byteStride {} is too small for VEC3 float payloads",
            decoded.stride
        )));
    }

    let mut positions = Vec::<[f64; 3]>::with_capacity(decoded.count);
    for i in 0..decoded.count {
        let offset = decoded.base_offset + i.saturating_mul(decoded.stride);
        let x = read_f32_le_as_f64(&decoded.bytes, decoded.view_end, offset, "POSITION x")?;
        let y = read_f32_le_as_f64(&decoded.bytes, decoded.view_end, offset + 4, "POSITION y")?;
        let z = read_f32_le_as_f64(&decoded.bytes, decoded.view_end, offset + 8, "POSITION z")?;
        positions.push([x, y, z]);
    }
    Ok(positions)
}

fn parse_accessor_indices(
    root: &Value,
    accessor_index: u64,
) -> Result<Vec<usize>, GeometryImportError> {
    let decoded = resolve_accessor_decode(root, accessor_index)?;
    if decoded.accessor_type != "SCALAR" {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF index accessor type '{}' is not supported; expected SCALAR",
            decoded.accessor_type
        )));
    }

    let component_size = match decoded.component_type {
        5121 => 1usize,
        5123 => 2usize,
        5125 => 4usize,
        other => {
            return Err(GeometryImportError::ParseFailed(format!(
                "GLTF index accessor componentType {} is not supported; expected 5121/5123/5125",
                other
            )));
        }
    };
    if decoded.stride < component_size {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF index accessor byteStride {} is too small for component size {}",
            decoded.stride, component_size
        )));
    }

    let mut indices = Vec::<usize>::with_capacity(decoded.count);
    for i in 0..decoded.count {
        let offset = decoded.base_offset + i.saturating_mul(decoded.stride);
        let index = match decoded.component_type {
            5121 => {
                if offset >= decoded.view_end || offset >= decoded.bytes.len() {
                    return Err(GeometryImportError::ParseFailed(
                        "GLTF index accessor byte offset is out of bounds".to_string(),
                    ));
                }
                decoded.bytes[offset] as u64
            }
            5123 => {
                let bytes = get_slice(
                    &decoded.bytes,
                    decoded.view_end,
                    offset,
                    2,
                    "index accessor u16",
                )?;
                u16::from_le_bytes([bytes[0], bytes[1]]) as u64
            }
            5125 => {
                let bytes = get_slice(
                    &decoded.bytes,
                    decoded.view_end,
                    offset,
                    4,
                    "index accessor u32",
                )?;
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64
            }
            _ => unreachable!(),
        };
        indices.push(usize::try_from(index).map_err(|_| {
            GeometryImportError::ParseFailed(
                "GLTF index accessor value exceeds host addressable range".to_string(),
            )
        })?);
    }
    Ok(indices)
}

struct AccessorDecode {
    bytes: Vec<u8>,
    base_offset: usize,
    view_end: usize,
    count: usize,
    stride: usize,
    component_type: u64,
    accessor_type: String,
}

fn component_size_bytes(component_type: u64) -> Option<usize> {
    match component_type {
        5121 => Some(1),
        5123 => Some(2),
        5125 => Some(4),
        5126 => Some(4),
        _ => None,
    }
}

fn resolve_accessor_decode(
    root: &Value,
    accessor_index: u64,
) -> Result<AccessorDecode, GeometryImportError> {
    let accessors = root
        .get("accessors")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "GLTF accessor-backed payload requires top-level accessors[]".to_string(),
            )
        })?;
    let accessor = accessors
        .get(usize::try_from(accessor_index).map_err(|_| {
            GeometryImportError::ParseFailed("GLTF accessor index exceeds host range".to_string())
        })?)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(format!(
                "GLTF accessor index {} is out of bounds",
                accessor_index
            ))
        })?;
    if accessor
        .get("normalized")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(GeometryImportError::ParseFailed(
            "GLTF normalized accessors are not supported yet".to_string(),
        ));
    }
    if accessor.get("sparse").is_some() {
        return Err(GeometryImportError::ParseFailed(
            "GLTF sparse accessors are not supported yet".to_string(),
        ));
    }
    let buffer_view_index = accessor
        .get("bufferView")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "GLTF accessor-backed payload requires accessor.bufferView".to_string(),
            )
        })?;
    let accessor_byte_offset = parse_usize(accessor.get("byteOffset").and_then(Value::as_u64), 0)?;
    let component_type = accessor
        .get("componentType")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "GLTF accessor-backed payload requires accessor.componentType".to_string(),
            )
        })?;
    let count = parse_usize(accessor.get("count").and_then(Value::as_u64), 0)?;
    let accessor_type = accessor
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "GLTF accessor-backed payload requires accessor.type".to_string(),
            )
        })?
        .to_string();

    let buffer_views = root
        .get("bufferViews")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "GLTF accessor-backed payload requires top-level bufferViews[]".to_string(),
            )
        })?;
    let buffer_view = buffer_views
        .get(usize::try_from(buffer_view_index).map_err(|_| {
            GeometryImportError::ParseFailed("GLTF bufferView index exceeds host range".to_string())
        })?)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(format!(
                "GLTF bufferView index {} is out of bounds",
                buffer_view_index
            ))
        })?;
    let buffer_index = buffer_view
        .get("buffer")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "GLTF accessor-backed payload requires bufferView.buffer".to_string(),
            )
        })?;
    let buffer_view_offset = parse_usize(buffer_view.get("byteOffset").and_then(Value::as_u64), 0)?;
    let buffer_view_byte_length = parse_usize(
        buffer_view.get("byteLength").and_then(Value::as_u64),
        usize::MAX,
    )?;
    if buffer_view_byte_length == usize::MAX {
        return Err(GeometryImportError::ParseFailed(
            "GLTF accessor-backed payload requires bufferView.byteLength".to_string(),
        ));
    }
    let byte_stride = parse_usize(buffer_view.get("byteStride").and_then(Value::as_u64), 0)?;

    let buffers = root
        .get("buffers")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "GLTF accessor-backed payload requires top-level buffers[]".to_string(),
            )
        })?;
    let buffer = buffers
        .get(usize::try_from(buffer_index).map_err(|_| {
            GeometryImportError::ParseFailed("GLTF buffer index exceeds host range".to_string())
        })?)
        .ok_or_else(|| {
            GeometryImportError::ParseFailed(format!(
                "GLTF buffer index {} is out of bounds",
                buffer_index
            ))
        })?;
    let uri = buffer.get("uri").and_then(Value::as_str).ok_or_else(|| {
        GeometryImportError::ParseFailed(
            "GLTF accessor-backed payload requires buffer.uri data URI (external/GLB buffers are not supported yet)"
                .to_string(),
        )
    })?;
    let bytes = decode_data_uri(uri)?;
    let declared_buffer_length =
        parse_usize(buffer.get("byteLength").and_then(Value::as_u64), usize::MAX)?;
    if declared_buffer_length == usize::MAX {
        return Err(GeometryImportError::ParseFailed(
            "GLTF accessor-backed payload requires buffer.byteLength".to_string(),
        ));
    }
    if declared_buffer_length > bytes.len() {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF buffer byteLength {} exceeds decoded data URI payload size {}",
            declared_buffer_length,
            bytes.len()
        )));
    }
    let buffer_limit = declared_buffer_length;

    let buffer_view_end = buffer_view_offset.saturating_add(buffer_view_byte_length);
    if buffer_view_offset > buffer_limit || buffer_view_end > buffer_limit {
        return Err(GeometryImportError::ParseFailed(
            "GLTF accessor-backed bufferView byte range is out of bounds for declared buffer length"
                .to_string(),
        ));
    }
    if accessor_byte_offset > buffer_view_byte_length {
        return Err(GeometryImportError::ParseFailed(
            "GLTF accessor byteOffset exceeds bufferView byteLength".to_string(),
        ));
    }

    let base_offset = buffer_view_offset.saturating_add(accessor_byte_offset);
    if base_offset > bytes.len() || base_offset > buffer_view_end {
        return Err(GeometryImportError::ParseFailed(
            "GLTF accessor-backed payload offset is out of bounds".to_string(),
        ));
    }
    let stride = if byte_stride == 0 {
        match accessor_type.as_str() {
            "VEC3" => 12,
            "SCALAR" => match component_type {
                5121 => 1,
                5123 => 2,
                5125 => 4,
                _ => 1,
            },
            _ => 1,
        }
    } else {
        byte_stride
    };

    let component_size = component_size_bytes(component_type).unwrap_or(1);
    if stride % component_size != 0 {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF accessor byteStride {} is not aligned to component size {}",
            stride, component_size
        )));
    }
    let element_size = match accessor_type.as_str() {
        "VEC3" => component_size.saturating_mul(3),
        "SCALAR" => component_size,
        _ => 1,
    };
    if stride < element_size {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF accessor byteStride {} is too small for accessor element size {}",
            stride, element_size
        )));
    }
    let required_span = if count == 0 {
        0usize
    } else {
        (count - 1)
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(element_size))
            .ok_or_else(|| {
                GeometryImportError::ParseFailed(
                    "GLTF accessor declared count/stride overflows host range".to_string(),
                )
            })?
    };
    let accessor_end = base_offset.checked_add(required_span).ok_or_else(|| {
        GeometryImportError::ParseFailed(
            "GLTF accessor declared count/stride overflows host range".to_string(),
        )
    })?;
    if accessor_end > buffer_view_end || accessor_end > bytes.len() {
        return Err(GeometryImportError::ParseFailed(
            "GLTF accessor declared count/stride exceeds bufferView byte range".to_string(),
        ));
    }

    Ok(AccessorDecode {
        bytes,
        base_offset,
        view_end: buffer_view_end,
        count,
        stride,
        component_type,
        accessor_type,
    })
}

fn decode_data_uri(uri: &str) -> Result<Vec<u8>, GeometryImportError> {
    if !uri.starts_with("data:") {
        return Err(GeometryImportError::ParseFailed(
            "GLTF accessor-backed buffer.uri must be a data URI (external file URIs are not supported yet)"
                .to_string(),
        ));
    }
    let (metadata, payload) = uri.split_once(',').ok_or_else(|| {
        GeometryImportError::ParseFailed(
            "GLTF accessor-backed data URI is missing payload separator".to_string(),
        )
    })?;
    if !metadata.contains(";base64") {
        return Err(GeometryImportError::ParseFailed(
            "GLTF accessor-backed data URI must use base64 encoding".to_string(),
        ));
    };
    BASE64_ENGINE.decode(payload.as_bytes()).map_err(|err| {
        GeometryImportError::ParseFailed(format!(
            "failed to decode GLTF accessor-backed data URI payload: {err}"
        ))
    })
}

fn parse_usize(value: Option<u64>, default: usize) -> Result<usize, GeometryImportError> {
    match value {
        Some(raw) => usize::try_from(raw).map_err(|_| {
            GeometryImportError::ParseFailed("GLTF integer value exceeds host range".to_string())
        }),
        None => Ok(default),
    }
}

fn read_f32_le_as_f64(
    bytes: &[u8],
    view_end: usize,
    offset: usize,
    label: &str,
) -> Result<f64, GeometryImportError> {
    let value = get_slice(bytes, view_end, offset, 4, label)?;
    Ok(f32::from_le_bytes([value[0], value[1], value[2], value[3]]) as f64)
}

fn get_slice<'a>(
    bytes: &'a [u8],
    view_end: usize,
    offset: usize,
    len: usize,
    label: &str,
) -> Result<&'a [u8], GeometryImportError> {
    let end = offset.saturating_add(len);
    if end > view_end || end > bytes.len() {
        return Err(GeometryImportError::ParseFailed(format!(
            "GLTF accessor-backed {} payload is out of bounds",
            label
        )));
    };
    Ok(&bytes[offset..end])
}
