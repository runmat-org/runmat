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
    // A mode shape's relative amplitude has no unit, study range or limit, as in the snapshot
    // posters: the study's display and assessments don't apply to it.
    let no_terms = legend::FieldLegendTerms::default();
    let terms = if field.as_ref().is_some_and(|field| field.mode.is_some()) {
        &no_terms
    } else {
        terms
    };
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
    let stats =
        SceneStats::from_selection(&selection, field.as_ref(), deformation).with_terms(terms);
    if field.is_some() && stats.scale.extremes.is_none() {
        return Err("the selected field has no persisted global plot range".into());
    }
    if shape_mode == "exaggerated"
        && deformation.is_some()
        && stats.exaggerated_scale(1.0).is_none()
    {
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
        Some(stats),
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
        if let Some(field) = field.as_ref().filter(|field| field.mode.is_some()) {
            present_mode_amplitude(overlay, field)?;
        } else {
            present_field_legend(overlay, field.as_ref(), terms, run_id)?;
        }
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

/// The legend of a field in the study's terms: its display unit, range and assessment line.
fn present_field_legend(
    overlay: &mut runmat_plot::geometry_scene::GeometrySceneOverlay,
    field: Option<&super::FieldData>,
    terms: &legend::FieldLegendTerms,
    run_id: &str,
) -> Result<(), FieldSceneError> {
    let shown = field.map(|field| legend::ShownField {
        field_id: &field.label,
        location: field.location,
    });
    let locate = |field_id: &str| {
        super::load_field_page(run_id, &AnalysisFieldSelection::new(field_id, None), 0)
            .ok()
            .map(|page| page.descriptor.location)
    };
    legend::present_overlay(
        overlay,
        field.map(|f| f.unit.as_str()),
        terms,
        shown,
        &locate,
    )?;
    Ok(())
}

/// The legend of a mode shape: its amplitude relative to the shape's peak, titled by its mode
/// like the snapshot posters ("Mode 1 relative amplitude"). A field recovered from the mode
/// (its stress, say) keeps its name in the title.
fn present_mode_amplitude(
    overlay: &mut runmat_plot::geometry_scene::GeometrySceneOverlay,
    field: &super::FieldData,
) -> Result<(), FieldSceneError> {
    let Some(legend) = overlay.scalar_legend.as_mut() else {
        return Ok(());
    };
    let number = field.mode.map_or(1, |index| index + 1);
    let label = if field.deforms {
        legend::mode_amplitude_label(number)
    } else {
        format!("Mode {number} {} relative amplitude", field.label)
    };
    legend::present_relative_amplitude(legend, label)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{scene_from_topology, AnalysisFieldLocation, FieldData};
    use super::present_mode_amplitude;
    use crate::analysis::{
        AnalysisRenderMesh, AnalysisRenderTopology, AnalysisRenderTopologySource,
    };

    fn triangle() -> AnalysisRenderTopology {
        AnalysisRenderTopology {
            schema_version: "test".into(),
            source: AnalysisRenderTopologySource::AnalysisMesh,
            volume_element_count: 1,
            meshes: vec![AnalysisRenderMesh {
                mesh_id: "mesh".into(),
                vertices: vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.1]],
                triangles: vec![[0, 1, 2]],
                regions: vec![],
                vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
                triangle_volume_element_indices: vec![Some(0)],
            }],
        }
    }

    // A mass-normalized eigenvector (published with unit "1") peaks at 7.0; the live view's
    // legend reads its relative amplitude, 0 to 1, as the snapshot poster does, never
    // "0 – 7.0" or a length.
    #[test]
    fn a_mode_shape_legend_reads_relative_amplitude_not_metres() {
        let field = FieldData::inline(
            AnalysisFieldLocation::Node,
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0],
            "structural.displacement",
            "1",
        )
        .mode_shape();
        let (mut scene, _) = scene_from_topology(
            "run",
            &triangle(),
            Some(&field),
            Some(&field),
            "exaggerated",
            1.0,
            false,
        )
        .unwrap();
        let overlay = scene.overlay.as_mut().unwrap();
        present_mode_amplitude(overlay, &field).unwrap();
        let legend = overlay.scalar_legend.as_ref().unwrap();
        assert_eq!(legend.label, "Mode 1 relative amplitude");
        assert_eq!(legend.unit, "relative");
        assert_eq!((legend.minimum, legend.maximum), (0.0, 1.0));
        assert!(legend.limit.is_none() && legend.note.is_none());
    }
}
