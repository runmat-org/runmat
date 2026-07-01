//! Progressive CAD preview sessions.
//!
//! This API is intentionally scoped to interactive preview. Full geometry loads
//! still go through `import_geometry*`, which keeps canonical analysis assets
//! independent from renderer-driven chunking.

use crate::{
    import::{cad, GeometryImportContext, GeometryImportError, GeometryImportOptions},
    occt::{self, OcctCadFormat},
    sniff::{detect_geometry_format, GeometryFormat},
    ImportResult,
};

#[derive(Debug, Clone)]
pub struct CadPreviewSessionStart {
    pub session_id: u64,
    pub path: String,
    pub format: GeometryFormat,
    pub face_count: u64,
}

#[derive(Debug, Clone)]
pub struct CadPreviewSessionChunk {
    pub session_id: u64,
    pub done: bool,
    pub face_cursor: u64,
    pub face_count: u64,
    pub result: ImportResult,
}

pub fn start_cad_preview_session(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<CadPreviewSessionStart, GeometryImportError> {
    context.check_cancelled()?;
    let format = detect_geometry_format(path, bytes);
    let Some(occt_format) = OcctCadFormat::from_geometry_format(format) else {
        return Err(GeometryImportError::UnsupportedFormat);
    };
    let session = occt::start_cad_preview_session(path, bytes, occt_format, &options, context)?;
    context.check_cancelled()?;
    Ok(CadPreviewSessionStart {
        session_id: session.session_id,
        path: path.to_string(),
        format,
        face_count: session.face_count,
    })
}

pub fn read_cad_preview_session_chunk(
    session_id: u64,
    path: &str,
    format: GeometryFormat,
    target_triangles: u64,
    max_faces: u64,
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<CadPreviewSessionChunk, GeometryImportError> {
    context.check_cancelled()?;
    let Some(occt_format) = OcctCadFormat::from_geometry_format(format) else {
        return Err(GeometryImportError::UnsupportedFormat);
    };
    let chunk = occt::read_cad_preview_session_chunk(
        session_id,
        target_triangles,
        max_faces,
        &options,
        context,
    )?;
    context.check_cancelled()?;
    let result = cad::build_topology_result(path, occt_format, chunk.topology, None, options)?;
    Ok(CadPreviewSessionChunk {
        session_id,
        done: chunk.done,
        face_cursor: chunk.face_cursor,
        face_count: chunk.face_count,
        result,
    })
}

pub fn close_cad_preview_session(session_id: u64) {
    occt::close_cad_preview_session(session_id);
}
