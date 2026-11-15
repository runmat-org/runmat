//! A disposable view of a run drawn from its persisted render LOD tiles.

use super::{
    legend, load_field, scene_from_topology_with_stats, storage, AnalysisFieldLocation,
    AnalysisFieldSelection, AnalysisRenderTopology, FieldSceneError, SceneStats,
};
use crate::analysis::render_tiles::RenderViewRequest;
use crate::analysis::AnalysisRenderMesh;
use glam::Vec3;
use runmat_plot::core::BoundingBox;
use runmat_plot::geometry_scene::GeometrySceneCompleteness;
use runmat_plot::GeometryScene;

/// Materialize only the LOD nodes chosen for this view. The returned scene is
/// disposable; replacing it releases the previous view's geometry buffers.
pub fn build_scene_view(
    run_id: &str,
    field_id: Option<&str>,
    shape_mode: &str,
    exaggeration: f64,
    show_undeformed_outline: bool,
    view: Option<RenderViewRequest>,
    terms: &legend::FieldLegendTerms,
) -> Result<(GeometryScene, f64, bool), FieldSceneError> {
    if !matches!(shape_mode, "undeformed" | "true" | "exaggerated") {
        return Err(format!("unknown FEA shape mode '{shape_mode}'").into());
    }
    if !exaggeration.is_finite() || !(0.0..=100.0).contains(&exaggeration) {
        return Err("FEA exaggeration must be finite and between 0 and 100".into());
    }
    // The run record states the same fact: its steps' `render_topology` is `absent` exactly
    // when publication saved no render tiles for the run.
    let selection = storage::plan_render_view(run_id, view)?
        .ok_or_else(|| FieldSceneError::render_topology_missing(run_id))?;
    let mut meshes = Vec::with_capacity(selection.nodes.len());
    for node in &selection.nodes {
        let tile = storage::load_render_lod_node(run_id, node.mesh_index, node.node_index)?
            .ok_or("FEA render LOD tile is missing")?;
        meshes.push(AnalysisRenderMesh {
            mesh_id: format!("{}:lod:{}", node.mesh_id, node.node_index),
            vertices: tile.vertices,
            triangles: tile.triangles,
            regions: Vec::new(),
            vertex_volume_node_indices: tile.vertex_volume_node_indices,
            triangle_volume_element_indices: tile.triangle_volume_element_indices,
        });
    }
    if meshes.is_empty() {
        return Err("FEA run has no renderable surface mesh".into());
    }
    let topology = AnalysisRenderTopology {
        schema_version: selection.topology_schema_version.clone(),
        source: selection.source,
        volume_element_count: selection.volume_element_count,
        meshes,
    };
    let field = field_id
        .map(|id| load_field(run_id, &AnalysisFieldSelection::new(id, None)))
        .transpose()?;
    let additional_deformation =
        if shape_mode == "undeformed" || field.as_ref().is_some_and(|field| field.deforms) {
            None
        } else {
            load_field(
                run_id,
                &AnalysisFieldSelection::new("structural.displacement", None),
            )
            .ok()
        };
    let deformation = field
        .as_ref()
        .filter(|field| field.deforms)
        .or(additional_deformation.as_ref())
        .filter(|field| field.location == AnalysisFieldLocation::Node && field.components >= 3);
    let stats = SceneStats::from_selection(&selection, field.as_ref(), deformation);
    if field.is_some() && stats.scale.extremes.is_none() {
        return Err("the selected field has no persisted global plot range".into());
    }
    if shape_mode == "exaggerated" && deformation.is_some() && stats.automatic_scale.is_none() {
        return Err("the displacement field has no persisted deformation range".into());
    }
    let (mut scene, scale) = scene_from_topology_with_stats(
        run_id,
        &topology,
        field.as_ref(),
        deformation,
        shape_mode,
        exaggeration,
        show_undeformed_outline,
        Some(stats.with_terms(terms)),
    )?;
    let mut model_bounds = BoundingBox::default();
    let bounds = selection.model_bounds;
    model_bounds.expand_by_box(&BoundingBox::new(
        Vec3::from_array(bounds[0].map(|value| value as f32)),
        Vec3::from_array(bounds[1].map(|value| value as f32)),
    ));
    model_bounds.expand_by_box(&scene.bounds);
    scene.bounds = model_bounds;
    if let Some(overlay) = scene.overlay.as_mut() {
        legend::present_overlay(overlay, field.as_ref().map(|f| f.unit.as_str()), terms)?;
        overlay.mesh_count = selection.mesh_count;
        overlay.status = if selection.exact {
            GeometrySceneCompleteness::Complete
        } else {
            GeometrySceneCompleteness::BoundedPreview
        };
        overlay.quality_label = (!selection.exact).then(|| "Adaptive mesh detail".into());
    }
    Ok((scene, scale, selection.exact))
}
