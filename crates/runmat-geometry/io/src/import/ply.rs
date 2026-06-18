use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};
use runmat_geometry_core::SurfaceMesh;

use super::{
    build_asset, build_result, capacity_guard, check_cancelled_periodic, is_degenerate_triangle,
    parse_f64, push_mesh_count_diagnostics, push_utf8_bom_stripped_diagnostic,
    strip_utf8_bom_bytes, BuildAssetInput, GeometryImportContext, GeometryImportError,
    GeometryImportOptions,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryFaceIndexType {
    I32,
    U32,
}

struct PlyMeshAccumulator<'a> {
    vertices: &'a mut Vec<[f64; 3]>,
    triangles: &'a mut Vec<[u32; 3]>,
    triangle_count: &'a mut u64,
    diagnostics: &'a mut Vec<ImportDiagnostic>,
}

pub(super) fn import_ply(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    context.check_cancelled()?;
    let mut diagnostics = Vec::<ImportDiagnostic>::new();
    let (bytes, stripped_bom) = strip_utf8_bom_bytes(bytes);
    if stripped_bom {
        push_utf8_bom_stripped_diagnostic(&mut diagnostics, "ply");
    }
    let (header, body) = parse_ply_header(bytes)?;

    let mut vertices = Vec::<[f64; 3]>::with_capacity(header.vertex_count);
    let mut triangles = Vec::<[u32; 3]>::new();
    let mut triangle_count = 0u64;

    {
        let mut mesh = PlyMeshAccumulator {
            vertices: &mut vertices,
            triangles: &mut triangles,
            triangle_count: &mut triangle_count,
            diagnostics: &mut diagnostics,
        };
        match header.format {
            PlyFormat::Ascii10 => {
                parse_ascii_body(
                    body,
                    header.vertex_count,
                    header.face_count,
                    &mut mesh,
                    &options,
                    context,
                )?;
            }
            PlyFormat::BinaryLittleEndian10 => {
                let face_index_type = validate_binary_layout_support(&header.header_text)?;
                parse_binary_little_endian_body(
                    body,
                    &header,
                    face_index_type,
                    &mut mesh,
                    &options,
                    context,
                )?;
            }
        }
    }

    push_mesh_count_diagnostics(
        &mut diagnostics,
        "ply",
        vertices.len() as u64,
        triangle_count,
    );
    let asset = build_asset(BuildAssetInput {
        path,
        importer_version: "ply/v1",
        units: options.units,
        tessellation_profile: options.tessellation_profile.clone(),
        vertex_count: vertices.len() as u64,
        element_count: triangle_count,
        surface_meshes: vec![SurfaceMesh::new("mesh_1", vertices, triangles)],
        diagnostics: diagnostics.clone(),
    });
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

fn validate_binary_layout_support(
    header_text: &str,
) -> Result<BinaryFaceIndexType, GeometryImportError> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ElementKind {
        Vertex,
        Face,
        Other,
    }

    let mut element = ElementKind::Other;
    let mut vertex_props = Vec::<String>::new();
    let mut face_props = Vec::<String>::new();
    for line in header_text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("element vertex ") {
            element = ElementKind::Vertex;
            continue;
        }
        if trimmed.starts_with("element face ") {
            element = ElementKind::Face;
            continue;
        }
        if trimmed.starts_with("element ") {
            element = ElementKind::Other;
            continue;
        }
        if !trimmed.starts_with("property ") {
            continue;
        }
        match element {
            ElementKind::Vertex => vertex_props.push(trimmed.to_string()),
            ElementKind::Face => face_props.push(trimmed.to_string()),
            ElementKind::Other => {}
        }
    }

    let expected_vertex = ["property float x", "property float y", "property float z"];
    if vertex_props.len() != expected_vertex.len()
        || vertex_props
            .iter()
            .map(String::as_str)
            .zip(expected_vertex)
            .any(|(actual, expected)| actual != expected)
    {
        return Err(GeometryImportError::ParseFailed(
            "binary little-endian PLY currently requires vertex properties exactly: property float x/y/z"
                .to_string(),
        ));
    }

    if face_props.as_slice() == ["property list uchar int vertex_indices"] {
        return Ok(BinaryFaceIndexType::I32);
    }
    if face_props.as_slice() == ["property list uchar uint vertex_indices"] {
        return Ok(BinaryFaceIndexType::U32);
    }
    Err(GeometryImportError::ParseFailed(
        "binary little-endian PLY currently requires face properties exactly: property list uchar int|uint vertex_indices"
            .to_string(),
    ))
}

fn parse_ascii_body(
    body: &[u8],
    vertex_count: usize,
    face_count: usize,
    mesh: &mut PlyMeshAccumulator<'_>,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<(), GeometryImportError> {
    let body_text = std::str::from_utf8(body).map_err(|_| {
        GeometryImportError::ParseFailed("invalid UTF-8 PLY ASCII body".to_string())
    })?;
    let mut lines = body_text.lines();
    for index in 0..vertex_count {
        check_cancelled_periodic(context, index)?;
        let vertex_line = lines.next().ok_or_else(|| {
            GeometryImportError::ParseFailed(format!(
                "PLY declared {} vertices but parsed {}",
                vertex_count,
                mesh.vertices.len()
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
        mesh.vertices.push([x, y, z]);
    }

    for face_index in 0..face_count {
        check_cancelled_periodic(context, face_index)?;
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
            if index >= mesh.vertices.len() {
                return Err(GeometryImportError::ParseFailed(format!(
                    "PLY face index {} out of bounds for {} vertices",
                    index,
                    mesh.vertices.len()
                )));
            }
            indices.push(index);
        }

        emit_face_triangles(&indices, mesh, options, context)?;
    }
    Ok(())
}

fn parse_binary_little_endian_body(
    body: &[u8],
    header: &PlyHeader,
    face_index_type: BinaryFaceIndexType,
    mesh: &mut PlyMeshAccumulator<'_>,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<(), GeometryImportError> {
    let mut cursor = 0usize;
    for index in 0..header.vertex_count {
        check_cancelled_periodic(context, index)?;
        let x = read_f32_le(body, cursor, "PLY binary vertex x")?;
        let y = read_f32_le(body, cursor + 4, "PLY binary vertex y")?;
        let z = read_f32_le(body, cursor + 8, "PLY binary vertex z")?;
        mesh.vertices.push([x, y, z]);
        cursor = cursor.saturating_add(12);
    }

    for face_index in 0..header.face_count {
        check_cancelled_periodic(context, face_index)?;
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
            let raw = match face_index_type {
                BinaryFaceIndexType::I32 => {
                    let value = read_i32_le(body, cursor, "PLY binary face index")?;
                    if value < 0 {
                        return Err(GeometryImportError::ParseFailed(
                            "PLY binary face index must be non-negative".to_string(),
                        ));
                    }
                    value as u64
                }
                BinaryFaceIndexType::U32 => {
                    read_u32_le(body, cursor, "PLY binary face index")? as u64
                }
            };
            cursor = cursor.saturating_add(4);
            let index = usize::try_from(raw).map_err(|_| {
                GeometryImportError::ParseFailed(
                    "PLY binary face index exceeds host addressable range".to_string(),
                )
            })?;
            if index >= mesh.vertices.len() {
                return Err(GeometryImportError::ParseFailed(format!(
                    "PLY face index {} out of bounds for {} vertices",
                    index,
                    mesh.vertices.len()
                )));
            }
            indices.push(index);
        }

        emit_face_triangles(&indices, mesh, options, context)?;
    }
    Ok(())
}

fn emit_face_triangles(
    indices: &[usize],
    mesh: &mut PlyMeshAccumulator<'_>,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<(), GeometryImportError> {
    let pivot = indices[0];
    for i in 1..(indices.len() - 1) {
        check_cancelled_periodic(context, i)?;
        capacity_guard(*mesh.triangle_count + 1, options)?;
        let tri = [
            mesh.vertices[pivot],
            mesh.vertices[indices[i]],
            mesh.vertices[indices[i + 1]],
        ];
        if is_degenerate_triangle(&tri) {
            mesh.diagnostics.push(ImportDiagnostic {
                code: "GEOMETRY_NORMALIZE_DEGENERATE_REMOVED".to_string(),
                severity: ImportDiagnosticSeverity::Warning,
                message: "Removed degenerate PLY face triangle during import".to_string(),
            });
        } else {
            mesh.triangles.push([
                u32::try_from(pivot).map_err(|_| {
                    GeometryImportError::ParseFailed(
                        "PLY vertex index exceeds render mesh index range".to_string(),
                    )
                })?,
                u32::try_from(indices[i]).map_err(|_| {
                    GeometryImportError::ParseFailed(
                        "PLY vertex index exceeds render mesh index range".to_string(),
                    )
                })?,
                u32::try_from(indices[i + 1]).map_err(|_| {
                    GeometryImportError::ParseFailed(
                        "PLY vertex index exceeds render mesh index range".to_string(),
                    )
                })?,
            ]);
            *mesh.triangle_count += 1;
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

fn read_u32_le(bytes: &[u8], offset: usize, label: &str) -> Result<u32, GeometryImportError> {
    let end = offset.saturating_add(4);
    if end > bytes.len() {
        return Err(GeometryImportError::ParseFailed(format!(
            "{} is out of bounds",
            label
        )));
    }
    Ok(u32::from_le_bytes([
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
