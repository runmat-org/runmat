//! Build a view from the persisted solver topology and one paged result field.
//! The view is disposable; the run artifacts remain the source of truth.

use super::render_tiles::{RenderViewRequest, RenderViewSelection, MAX_TRIANGLES_PER_TILE};
use super::{
    analysis_field_page_by_run_id_op, storage, AnalysisFieldLocation, AnalysisFieldPageValues,
    AnalysisFieldRequestOptions, AnalysisFieldSelection, AnalysisRenderMesh,
    AnalysisRenderTopology,
};
use crate::operations::OperationContext;
use glam::{Vec3, Vec4};
use runmat_plot::core::{
    AlphaMode, BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex,
};
use runmat_plot::geometry_scene::{
    GeometrySceneCompleteness, GeometrySceneOverlay, GeometrySceneScalarLegend,
};
use runmat_plot::{GeometryScene, GeometrySceneChunk};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};

const FIELD_PAGE_VALUES: usize = 4096;
// 256 × 4,096 f64 values cap each view's decoded field cache at 8 MiB.
const FIELD_CACHE_PAGES: usize = 256;

struct FieldData {
    location: AnalysisFieldLocation,
    components: usize,
    value_count: usize,
    values: FieldValues,
    label: String,
    unit: String,
}

enum FieldValues {
    Paged {
        run_id: String,
        cache: RefCell<FieldPageCache>,
    },
    #[cfg(test)]
    Inline(Vec<f64>),
}

#[derive(Default)]
struct FieldPageCache {
    pages: HashMap<usize, Vec<f64>>,
    order: VecDeque<usize>,
}

impl FieldData {
    fn range(&self, start: usize, len: usize) -> Result<Vec<f64>, String> {
        let end = start.checked_add(len).ok_or("field range overflow")?;
        if end > self.value_count {
            return Err(format!(
                "field '{}' is missing values {start}..{end}",
                self.label
            ));
        }
        match &self.values {
            #[cfg(test)]
            FieldValues::Inline(values) => Ok(values[start..end].to_vec()),
            FieldValues::Paged { run_id, cache } => {
                let mut result = Vec::with_capacity(len);
                let mut cursor = start;
                while cursor < end {
                    let page_start = cursor / FIELD_PAGE_VALUES * FIELD_PAGE_VALUES;
                    let mut cache = cache.borrow_mut();
                    if !cache.pages.contains_key(&page_start) {
                        let page = load_field_page(run_id, &self.label, page_start)?;
                        if page.offset != page_start
                            || page.count != FIELD_PAGE_VALUES.min(self.value_count - page_start)
                        {
                            return Err(format!("field '{}' page is incomplete", self.label));
                        }
                        let values = page_real_values(page.values);
                        if values.len() != page.count {
                            return Err(format!("field '{}' page has invalid length", self.label));
                        }
                        if cache.pages.len() == FIELD_CACHE_PAGES {
                            if let Some(oldest) = cache.order.pop_front() {
                                cache.pages.remove(&oldest);
                            }
                        }
                        cache.order.push_back(page_start);
                        cache.pages.insert(page_start, values);
                    }
                    let page = cache.pages.get(&page_start).expect("page just loaded");
                    let from = cursor - page_start;
                    let count = (end - cursor).min(page.len() - from);
                    if count == 0 {
                        return Err(format!("field '{}' page made no progress", self.label));
                    }
                    result.extend_from_slice(&page[from..from + count]);
                    cursor += count;
                }
                Ok(result)
            }
        }
    }

    #[cfg(test)]
    fn inline(
        location: AnalysisFieldLocation,
        components: usize,
        values: Vec<f64>,
        label: &str,
        unit: &str,
    ) -> Self {
        Self {
            location,
            components,
            value_count: values.len(),
            values: FieldValues::Inline(values),
            label: label.into(),
            unit: unit.into(),
        }
    }
}

fn load_field_page(
    run_id: &str,
    field_id: &str,
    offset: usize,
) -> Result<super::AnalysisFieldPageResult, String> {
    analysis_field_page_by_run_id_op(
        run_id,
        AnalysisFieldSelection::new(field_id, None),
        AnalysisFieldRequestOptions {
            offset: Some(offset),
            limit: Some(FIELD_PAGE_VALUES),
        },
        OperationContext::new(None, None),
    )
    .map(|result| result.data)
    .map_err(|error| format!("{}: {}", error.error_code, error.message))
}

fn page_real_values(values: AnalysisFieldPageValues) -> Vec<f64> {
    match values {
        AnalysisFieldPageValues::Real(values) => values,
        AnalysisFieldPageValues::ComplexInterleaved(values) => values
            .chunks_exact(2)
            .map(|pair| pair[0].hypot(pair[1]))
            .collect(),
    }
}

mod deformation;

fn is_deformation_field(id: &str) -> bool {
    id == "structural.displacement" || deformation::is_modal_field(id)
}

pub fn build_scene(
    run_id: &str,
    field_id: Option<&str>,
    shape_mode: &str,
    exaggeration: f64,
    show_undeformed_outline: bool,
) -> Result<(GeometryScene, f64), String> {
    if !matches!(shape_mode, "undeformed" | "true" | "exaggerated") {
        return Err(format!("unknown FEA shape mode '{shape_mode}'"));
    }
    if !exaggeration.is_finite() || !(0.0..=100.0).contains(&exaggeration) {
        return Err("FEA exaggeration must be finite and between 0 and 100".into());
    }
    let topology = storage::load_render_topology(run_id)?
        .ok_or_else(|| format!("FEA run '{run_id}' has no persisted render topology"))?;
    let field = field_id.map(|id| load_field(run_id, id)).transpose()?;
    let additional_deformation = if shape_mode == "undeformed"
        || field
            .as_ref()
            .is_some_and(|field| is_deformation_field(&field.label))
    {
        None
    } else {
        load_field(run_id, "structural.displacement").ok()
    };
    let deformation = field
        .as_ref()
        .filter(|field| is_deformation_field(&field.label))
        .or(additional_deformation.as_ref())
        .filter(|field| field.location == AnalysisFieldLocation::Node && field.components >= 3);
    scene_from_topology(
        run_id,
        &topology,
        field.as_ref(),
        deformation,
        shape_mode,
        exaggeration,
        show_undeformed_outline,
    )
}

/// Materialize only the LOD nodes chosen for this view. The returned scene is
/// disposable; replacing it releases the previous view's geometry buffers.
pub fn build_scene_view(
    run_id: &str,
    field_id: Option<&str>,
    shape_mode: &str,
    exaggeration: f64,
    show_undeformed_outline: bool,
    view: Option<RenderViewRequest>,
) -> Result<(GeometryScene, f64, bool), String> {
    if !matches!(shape_mode, "undeformed" | "true" | "exaggerated") {
        return Err(format!("unknown FEA shape mode '{shape_mode}'"));
    }
    if !exaggeration.is_finite() || !(0.0..=100.0).contains(&exaggeration) {
        return Err("FEA exaggeration must be finite and between 0 and 100".into());
    }
    let selection = storage::plan_render_view(run_id, view)?
        .ok_or_else(|| format!("FEA run '{run_id}' has no persisted render topology"))?;
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
    let field = field_id.map(|id| load_field(run_id, id)).transpose()?;
    let additional_deformation = if shape_mode == "undeformed"
        || field
            .as_ref()
            .is_some_and(|field| is_deformation_field(&field.label))
    {
        None
    } else {
        load_field(run_id, "structural.displacement").ok()
    };
    let deformation = field
        .as_ref()
        .filter(|field| is_deformation_field(&field.label))
        .or(additional_deformation.as_ref())
        .filter(|field| field.location == AnalysisFieldLocation::Node && field.components >= 3);
    let stats = SceneStats::from_selection(&selection, field.as_ref(), deformation);
    if field.is_some() && stats.scalar_range.is_none() {
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

struct SceneStats {
    element_count: usize,
    scalar_range: Option<(f64, f64)>,
    automatic_scale: Option<f64>,
}

impl SceneStats {
    fn from_selection(
        selection: &RenderViewSelection,
        field: Option<&FieldData>,
        deformation: Option<&FieldData>,
    ) -> Self {
        let scalar_range = field.and_then(|field| {
            selection
                .field_summaries
                .iter()
                .find(|summary| summary.field_id == field.label)
                .map(|summary| (summary.minimum, summary.maximum))
        });
        let [minimum, maximum] = selection.model_bounds;
        let diagonal = (0..3)
            .map(|axis| (maximum[axis] - minimum[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        let automatic_scale = deformation.and_then(|field| {
            selection
                .field_summaries
                .iter()
                .find(|summary| summary.field_id == field.label)
                .map(|summary| {
                    deformation::automatic_scale(diagonal, summary.maximum, &field.label)
                })
        });
        Self {
            element_count: selection.volume_element_count,
            scalar_range,
            automatic_scale,
        }
    }
}

fn load_field(run_id: &str, field_id: &str) -> Result<FieldData, String> {
    let page = load_field_page(run_id, field_id, 0)?;
    let descriptor = &page.descriptor;
    if !matches!(
        descriptor.location,
        AnalysisFieldLocation::Node | AnalysisFieldLocation::Element
    ) {
        return Err(format!(
            "field '{field_id}' does not map to mesh nodes or elements"
        ));
    }
    let components = descriptor.component_count.unwrap_or(1);
    if components == 0 || page.total_count == 0 {
        return Err(format!("field '{field_id}' has no values or components"));
    }
    let first = page_real_values(page.values);
    if page.offset != 0
        || page.count != FIELD_PAGE_VALUES.min(page.total_count)
        || first.len() != page.count
    {
        return Err(format!("field '{field_id}' first page has invalid length"));
    }
    let mut cache = FieldPageCache::default();
    cache.order.push_back(0);
    cache.pages.insert(0, first);
    Ok(FieldData {
        location: descriptor.location,
        components,
        value_count: page.total_count,
        values: FieldValues::Paged {
            run_id: run_id.into(),
            cache: RefCell::new(cache),
        },
        label: field_id.to_string(),
        unit: descriptor.unit.clone().unwrap_or_default(),
    })
}

mod geometry;
use geometry::*;

#[cfg(test)]
mod tests;

/// Preserve explicit topology lines so shaded, x-ray and wireframe modes share one scene.
pub fn append_solver_mesh_edges(scene: &mut GeometryScene) {
    let mut edge_chunks = Vec::new();
    for chunk in &scene.chunks {
        if chunk.render_data.pipeline_type != PipelineType::Triangles {
            continue;
        }
        let indices = chunk
            .indices
            .clone()
            .unwrap_or_else(|| (0..chunk.vertices.len() as u32).collect());
        let mut seen = HashSet::new();
        let mut vertices = Vec::new();
        for triangle in indices.chunks_exact(3) {
            for (a, b) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                let (Some(mut start), Some(mut end)) = (
                    chunk.vertices.get(a as usize).copied(),
                    chunk.vertices.get(b as usize).copied(),
                ) else {
                    continue;
                };
                let start_key = start.position.map(f32::to_bits);
                let end_key = end.position.map(f32::to_bits);
                let edge = if start_key < end_key {
                    (start_key, end_key)
                } else {
                    (end_key, start_key)
                };
                if !seen.insert(edge) {
                    continue;
                }
                start.color = [0.58, 0.72, 0.88, 0.95];
                end.color = start.color;
                vertices.extend([start, end]);
            }
        }
        if vertices.is_empty() {
            continue;
        }
        let vertex_count = vertices.len();
        let material = Material {
            albedo: Vec4::ONE,
            ..Material::default()
        };
        let mut edge_chunk = GeometrySceneChunk::from_render_data(
            format!("{}:solver-mesh-edges", chunk.chunk_id),
            RenderData {
                pipeline_type: PipelineType::Lines,
                vertices,
                indices: None,
                gpu_vertices: None,
                bounds: None,
                material,
                draw_calls: vec![DrawCall {
                    vertex_offset: 0,
                    vertex_count,
                    index_offset: None,
                    index_count: None,
                    instance_count: 1,
                }],
                image: None,
            },
        );
        edge_chunk.mesh_id = chunk.mesh_id.clone();
        edge_chunk.label = Some("Solver mesh edges".into());
        edge_chunk.owner_node_ids = chunk.owner_node_ids.clone();
        edge_chunks.push(edge_chunk);
    }
    scene.chunks.extend(edge_chunks);
}
