use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};

use super::{
    build_asset, build_result, capacity_guard, is_degenerate_triangle, parse_f64,
    push_mesh_count_diagnostics, push_utf8_bom_stripped_diagnostic, strip_utf8_bom_bytes,
    GeometryImportError, GeometryImportOptions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    Ascii10,
    BinaryLittleEndian10,
}

#[derive(Debug, Clone)]
struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    face_count: usize,
    header_text: String,
}

pub(super) fn import_ply(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    let mut diagnostics = Vec::<ImportDiagnostic>::new();
    let (bytes, stripped_bom) = strip_utf8_bom_bytes(bytes);
    if stripped_bom {
        push_utf8_bom_stripped_diagnostic(&mut diagnostics, "ply");
    }
    let (header, body) = parse_ply_header(bytes)?;

    let mut vertices = Vec::<[f64; 3]>::with_capacity(header.vertex_count);
    let mut triangle_count = 0u64;

    match header.format {
        PlyFormat::Ascii10 => {
            parse_ascii_body(
                body,
                header.vertex_count,
                header.face_count,
                &mut vertices,
                &mut triangle_count,
                &mut diagnostics,
                &options,
            )?;
        }
        PlyFormat::BinaryLittleEndian10 => {
            validate_binary_layout_support(&header.header_text)?;
            parse_binary_little_endian_body(
                body,
                header.vertex_count,
                header.face_count,
                &mut vertices,
                &mut triangle_count,
                &mut diagnostics,
                &options,
            )?;
        }
    }

    push_mesh_count_diagnostics(
        &mut diagnostics,
        "ply",
        vertices.len() as u64,
        triangle_count,
    );
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

fn parse_ply_header(bytes: &[u8]) -> Result<(PlyHeader, &[u8]), GeometryImportError> {
    const END_HEADER_LF: &[u8] = b"end_header\n";
    const END_HEADER_CRLF: &[u8] = b"end_header\r\n";

    let (header_end, marker_len) = if let Some(index) = bytes
        .windows(END_HEADER_CRLF.len())
        .position(|window| window == END_HEADER_CRLF)
    {
        (index, END_HEADER_CRLF.len())
    } else if let Some(index) = bytes
        .windows(END_HEADER_LF.len())
        .position(|window| window == END_HEADER_LF)
    {
        (index, END_HEADER_LF.len())
    } else {
        return Err(GeometryImportError::ParseFailed(
            "PLY missing end_header marker".to_string(),
        ));
    };

    let split = header_end + marker_len;
    let header_bytes = &bytes[..split];
    let body = &bytes[split..];
    let header_text = std::str::from_utf8(header_bytes)
        .map_err(|_| GeometryImportError::ParseFailed("invalid UTF-8 PLY header".to_string()))?;
    let lines = header_text.lines().collect::<Vec<_>>();
    if lines.is_empty() || lines[0].trim() != "ply" {
        return Err(GeometryImportError::ParseFailed(
            "missing PLY header".to_string(),
        ));
    }

    let mut vertex_count = None;
    let mut face_count = None;
    let mut format = None;
    for line in lines.iter().skip(1) {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("format ") {
            format = Some(match value {
                "ascii 1.0" => PlyFormat::Ascii10,
                "binary_little_endian 1.0" => PlyFormat::BinaryLittleEndian10,
                _ => {
                    return Err(GeometryImportError::ParseFailed(format!(
                        "unsupported PLY format '{}'; expected ascii 1.0 or binary_little_endian 1.0",
                        value
                    )));
                }
            });
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

    let format = format.ok_or_else(|| {
        GeometryImportError::ParseFailed("PLY missing format declaration".to_string())
    })?;
    let vertex_count = vertex_count.ok_or_else(|| {
        GeometryImportError::ParseFailed("PLY missing vertex element count".to_string())
    })?;
    let face_count = face_count.ok_or_else(|| {
        GeometryImportError::ParseFailed("PLY missing face element count".to_string())
    })?;

    Ok((
        PlyHeader {
            format,
            vertex_count,
            face_count,
            header_text: header_text.to_string(),
        },
        body,
    ))
}

fn validate_binary_layout_support(header_text: &str) -> Result<(), GeometryImportError> {
    let requires = [
        "property float x",
        "property float y",
        "property float z",
        "property list uchar int vertex_indices",
    ];
    if requires.iter().any(|token| !header_text.contains(token)) {
        return Err(GeometryImportError::ParseFailed(
            "binary little-endian PLY currently requires vertex float x/y/z and face list uchar int vertex_indices properties".to_string(),
        ));
    }
    Ok(())
}

fn parse_ascii_body(
    body: &[u8],
    vertex_count: usize,
    face_count: usize,
    vertices: &mut Vec<[f64; 3]>,
    triangle_count: &mut u64,
    diagnostics: &mut Vec<ImportDiagnostic>,
    options: &GeometryImportOptions,
) -> Result<(), GeometryImportError> {
    let body_text = std::str::from_utf8(body).map_err(|_| {
        GeometryImportError::ParseFailed("invalid UTF-8 PLY ASCII body".to_string())
    })?;
    let mut lines = body_text.lines();
    for _ in 0..vertex_count {
        let vertex_line = lines.next().ok_or_else(|| {
            GeometryImportError::ParseFailed(format!(
                "PLY declared {} vertices but parsed {}",
                vertex_count,
                vertices.len()
            ))
        })?;
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

    for _ in 0..face_count {
        let face_line = lines
            .next()
            .ok_or_else(|| GeometryImportError::ParseFailed("PLY missing face line".to_string()))?;
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

        emit_face_triangles(vertices, &indices, triangle_count, diagnostics, options)?;
    }
    Ok(())
}

fn parse_binary_little_endian_body(
    body: &[u8],
    vertex_count: usize,
    face_count: usize,
    vertices: &mut Vec<[f64; 3]>,
    triangle_count: &mut u64,
    diagnostics: &mut Vec<ImportDiagnostic>,
    options: &GeometryImportOptions,
) -> Result<(), GeometryImportError> {
    let mut cursor = 0usize;
    for _ in 0..vertex_count {
        let x = read_f32_le(body, cursor, "PLY binary vertex x")?;
        let y = read_f32_le(body, cursor + 4, "PLY binary vertex y")?;
        let z = read_f32_le(body, cursor + 8, "PLY binary vertex z")?;
        vertices.push([x, y, z]);
        cursor = cursor.saturating_add(12);
    }

    for _ in 0..face_count {
        let face_vertex_count = *body.get(cursor).ok_or_else(|| {
            GeometryImportError::ParseFailed(
                "PLY binary face data is truncated at list count".to_string(),
            )
        })? as usize;
        cursor += 1;
        if face_vertex_count < 3 {
            return Err(GeometryImportError::ParseFailed(
                "PLY face requires at least 3 vertices".to_string(),
            ));
        }

        let mut indices = Vec::<usize>::with_capacity(face_vertex_count);
        for _ in 0..face_vertex_count {
            let raw = read_i32_le(body, cursor, "PLY binary face index")?;
            cursor = cursor.saturating_add(4);
            if raw < 0 {
                return Err(GeometryImportError::ParseFailed(
                    "PLY binary face index must be non-negative".to_string(),
                ));
            }
            let index = raw as usize;
            if index >= vertices.len() {
                return Err(GeometryImportError::ParseFailed(format!(
                    "PLY face index {} out of bounds for {} vertices",
                    index,
                    vertices.len()
                )));
            }
            indices.push(index);
        }

        emit_face_triangles(vertices, &indices, triangle_count, diagnostics, options)?;
    }
    Ok(())
}

fn emit_face_triangles(
    vertices: &[[f64; 3]],
    indices: &[usize],
    triangle_count: &mut u64,
    diagnostics: &mut Vec<ImportDiagnostic>,
    options: &GeometryImportOptions,
) -> Result<(), GeometryImportError> {
    let pivot = indices[0];
    for i in 1..(indices.len() - 1) {
        capacity_guard(*triangle_count + 1, options)?;
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
            *triangle_count += 1;
        }
    }
    Ok(())
}

fn read_f32_le(bytes: &[u8], offset: usize, label: &str) -> Result<f64, GeometryImportError> {
    let end = offset.saturating_add(4);
    if end > bytes.len() {
        return Err(GeometryImportError::ParseFailed(format!(
            "{} is out of bounds",
            label
        )));
    }
    Ok(f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]) as f64)
}

fn read_i32_le(bytes: &[u8], offset: usize, label: &str) -> Result<i32, GeometryImportError> {
    let end = offset.saturating_add(4);
    if end > bytes.len() {
        return Err(GeometryImportError::ParseFailed(format!(
            "{} is out of bounds",
            label
        )));
    }
    Ok(i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

fn parse_u64(value: &str, field: &str) -> Result<u64, GeometryImportError> {
    value.parse::<u64>().map_err(|_| {
        GeometryImportError::ParseFailed(format!("invalid PLY {} value: {}", field, value))
    })
}
