//! The inspect payload every host returns for an imported geometry asset.
use super::{
    geometry_asset_summary, geometry_compute_stats_op, GeometryInspectPayload, GeometryStatsPayload,
};
use crate::operations::OperationContext;
use runmat_geometry_core::GeometryAsset;

/// Describes an imported asset: its statistics, summary, regions and diagnostics.
/// `format` is the source format label (for example `"step"` or `"brep"`).
pub fn geometry_inspect_payload_from_asset(
    path: &str,
    format: &str,
    byte_count: u64,
    asset: &GeometryAsset,
) -> Result<GeometryInspectPayload, String> {
    let stats = geometry_compute_stats_op(asset, OperationContext::new(None, None))
        .map_err(|err| format!("{}: {}", err.error_code, err.message))?
        .data;
    Ok(GeometryInspectPayload {
        path: path.to_string(),
        format: format.to_string(),
        byte_count,
        supported: true,
        stats: Some(GeometryStatsPayload {
            mesh_count: stats.mesh_count,
            total_vertices: stats.total_vertices,
            total_elements: stats.total_elements,
            region_count: stats.region_count,
        }),
        geometry_summary: serde_json::to_value(geometry_asset_summary(asset)).ok(),
        regions: asset
            .regions
            .iter()
            .filter_map(|region| serde_json::to_value(region).ok())
            .collect(),
        diagnostics: asset
            .diagnostics
            .iter()
            .filter_map(|diagnostic| serde_json::to_value(diagnostic).ok())
            .collect(),
        degraded_reason: None,
    })
}
