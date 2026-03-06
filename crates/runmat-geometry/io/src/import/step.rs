use super::{GeometryImportError, GeometryImportOptions};

pub(super) fn import_step(
    _path: &str,
    _bytes: &[u8],
    _options: GeometryImportOptions,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    Err(GeometryImportError::ParseFailed(
        "STEP import adapter not implemented yet (planned in io/src/cad)".to_string(),
    ))
}
