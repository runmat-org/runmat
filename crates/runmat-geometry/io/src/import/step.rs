use crate::cad::parse_step_summary;

use super::{build_asset, build_result, GeometryImportError, GeometryImportOptions};

pub(super) fn import_step(
    path: &str,
    bytes: &[u8],
    options: GeometryImportOptions,
) -> Result<crate::report::ImportResult, GeometryImportError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| GeometryImportError::ParseFailed("invalid UTF-8 STEP payload".to_string()))?;

    let summary = parse_step_summary(path, text).map_err(|reason| {
        GeometryImportError::ParseFailed(format!("STEP parse failed: {reason}"))
    })?;

    let mut asset = build_asset(
        path,
        "step/v1",
        options.units,
        0,
        0,
        Vec::new(),
        summary.diagnostics.clone(),
    );
    asset.source_geometry.kind = summary.source_kind;
    asset.source_geometry.assembly = summary.assembly;
    asset.source_geometry.material_evidence = summary.material_evidence;
    asset.regions = summary.regions;

    Ok(build_result(asset, summary.diagnostics))
}
