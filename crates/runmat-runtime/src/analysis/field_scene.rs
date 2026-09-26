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
            .is_some_and(|field| field.label == "structural.displacement")
    {
        None
    } else {
        load_field(run_id, "structural.displacement").ok()
    };
    let deformation = field
        .as_ref()
        .filter(|field| field.label == "structural.displacement")
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
            .is_some_and(|field| field.label == "structural.displacement")
    {
        None
    } else {
        load_field(run_id, "structural.displacement").ok()
    };
    let deformation = field
        .as_ref()
        .filter(|field| field.label == "structural.displacement")
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
                    if diagonal.is_finite() && summary.maximum > f64::EPSILON {
                        (0.12 * diagonal / summary.maximum).max(1.0)
                    } else {
                        0.0
                    }
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

fn scene_from_topology(
    run_id: &str,
    topology: &AnalysisRenderTopology,
    field: Option<&FieldData>,
    deformation: Option<&FieldData>,
    shape_mode: &str,
    exaggeration: f64,
    show_undeformed_outline: bool,
) -> Result<(GeometryScene, f64), String> {
    scene_from_topology_with_stats(
        run_id,
        topology,
        field,
        deformation,
        shape_mode,
        exaggeration,
        show_undeformed_outline,
        None,
    )
}

fn scene_from_topology_with_stats(
    run_id: &str,
    topology: &AnalysisRenderTopology,
    field: Option<&FieldData>,
    deformation: Option<&FieldData>,
    shape_mode: &str,
    exaggeration: f64,
    show_undeformed_outline: bool,
    stats: Option<SceneStats>,
) -> Result<(GeometryScene, f64), String> {
    let deformation_scale = match (shape_mode, deformation) {
        ("true", Some(_)) => 1.0,
        ("exaggerated", Some(field)) => {
            let automatic_scale = match stats.as_ref().and_then(|stats| stats.automatic_scale) {
                Some(scale) => scale,
                None => automatic_deformation_scale(topology, field)?,
            };
            automatic_scale * exaggeration
        }
        _ => 0.0,
    };
    let element_count = stats
        .as_ref()
        .map(|stats| stats.element_count)
        .unwrap_or(topology.volume_element_count);
    let element_stride = if let Some(field) =
        field.filter(|field| field.location == AnalysisFieldLocation::Element)
    {
        let entity_count = field.value_count / field.components;
        if element_count == 0 || entity_count % element_count != 0 {
            return Err(format!(
                "field '{}' cannot be mapped to the surface elements",
                field.label
            ));
        }
        entity_count / element_count
    } else {
        1
    };
    let mut minimum = stats
        .as_ref()
        .and_then(|stats| stats.scalar_range)
        .map_or(f64::INFINITY, |range| range.0);
    let mut maximum = stats
        .as_ref()
        .and_then(|stats| stats.scalar_range)
        .map_or(f64::NEG_INFINITY, |range| range.1);
    let mut sample_count = 0_usize;
    if let Some(field) = field {
        for mesh in &topology.meshes {
            match field.location {
                AnalysisFieldLocation::Node => {
                    for index in &mesh.vertex_volume_node_indices {
                        if let Some(index) = index {
                            let sample = field_magnitude(field, *index, 1)?;
                            minimum = minimum.min(sample);
                            maximum = maximum.max(sample);
                            sample_count += 1;
                        }
                    }
                }
                AnalysisFieldLocation::Element => {
                    for index in &mesh.triangle_volume_element_indices {
                        if let Some(index) = index {
                            let sample = field_magnitude(field, *index, element_stride)?;
                            minimum = minimum.min(sample);
                            maximum = maximum.max(sample);
                            sample_count += 1;
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        if sample_count == 0 {
            return Err(format!(
                "field '{}' has no mapping onto the persisted surface",
                field.label
            ));
        }
    }
    let mut chunks = Vec::new();
    let mut vertex_count = 0usize;
    let mut triangle_count = 0usize;
    for mesh in &topology.meshes {
        for (batch_index, triangles) in mesh.triangles.chunks(MAX_TRIANGLES_PER_TILE).enumerate() {
            let mut vertices = Vec::with_capacity(triangles.len() * 3);
            for (local_index, triangle) in triangles.iter().enumerate() {
                let triangle_index = batch_index * MAX_TRIANGLES_PER_TILE + local_index;
                let positions = triangle.map(|index| mesh.vertices.get(index as usize));
                let [Some(a), Some(b), Some(c)] = positions else {
                    return Err(format!(
                        "render topology '{}' contains an invalid triangle",
                        mesh.mesh_id
                    ));
                };
                let positions = [a, b, c].map(|position| cast_position(*position));
                let deformed = [0, 1, 2].map(|corner| {
                    let node = mesh
                        .vertex_volume_node_indices
                        .get(triangle[corner] as usize)
                        .copied()
                        .flatten();
                    deform_position(positions[corner], node, deformation, deformation_scale)
                });
                let [first, second, third] = deformed;
                let deformed = [first?, second?, third?];
                let normal = (Vec3::from_array(deformed[1]) - Vec3::from_array(deformed[0]))
                    .cross(Vec3::from_array(deformed[2]) - Vec3::from_array(deformed[0]))
                    .normalize_or_zero()
                    .to_array();
                for corner in 0..3 {
                    let sample = if let Some(field) = field {
                        let entity = match field.location {
                            AnalysisFieldLocation::Node => mesh
                                .vertex_volume_node_indices
                                .get(triangle[corner] as usize)
                                .copied()
                                .flatten(),
                            AnalysisFieldLocation::Element => mesh
                                .triangle_volume_element_indices
                                .get(triangle_index)
                                .copied()
                                .flatten(),
                            _ => None,
                        };
                        entity
                            .map(|index| field_magnitude(field, index, element_stride))
                            .transpose()?
                    } else {
                        None
                    };
                    let color = sample
                        .map(|value| scalar_color(value, minimum, maximum))
                        .unwrap_or([0.66, 0.73, 0.83, 1.0]);
                    vertices.push(Vertex {
                        position: deformed[corner],
                        color,
                        normal,
                        tex_coords: [0.0, 0.0],
                    });
                }
                triangle_count += 1;
            }
            vertex_count += vertices.len();
            let indices = (0..vertices.len() as u32).collect();
            let mut material = Material::default();
            material.albedo = Vec4::ONE;
            material.double_sided = true;
            let mut chunk = GeometrySceneChunk::indexed_triangles(
                format!("{}:surface:{batch_index}", mesh.mesh_id),
                vertices,
                indices,
                material,
            );
            chunk.mesh_id = Some(mesh.mesh_id.clone());
            chunks.push(chunk);
        }
    }
    if chunks.is_empty() {
        return Err("FEA run has no renderable surface mesh".into());
    }
    if show_undeformed_outline && deformation_scale != 0.0 {
        append_undeformed_outline(&mut chunks, topology);
    }
    let mut scene = GeometryScene::new(
        format!("fea:{run_id}:{}", field.map_or("mesh", |f| &f.label)),
        1,
        chunks,
    );
    append_solver_mesh_edges(&mut scene);
    scene.overlay = Some(GeometrySceneOverlay {
        source_name: None,
        status: GeometrySceneCompleteness::Complete,
        quality_label: None,
        format: None,
        source_label: None,
        allow_create_fea_study: false,
        byte_count: None,
        mesh_count: topology.meshes.len(),
        vertex_count,
        triangle_count,
        progress_percent: None,
        scalar_legend: field.map(|field| GeometrySceneScalarLegend {
            label: if element_stride > 1 {
                format!("{} mean magnitude", field.label)
            } else if field.components > 1 {
                format!("{} magnitude", field.label)
            } else {
                field.label.clone()
            },
            unit: field.unit.clone(),
            minimum,
            maximum,
        }),
        region_count: 0,
        mapped_region_count: 0,
        assembly_nodes: Vec::new(),
        regions: Vec::new(),
        warnings: Vec::new(),
    });
    Ok((scene, deformation_scale))
}

fn cast_position(position: [f64; 3]) -> [f32; 3] {
    position.map(|value| value as f32)
}

fn append_undeformed_outline(
    chunks: &mut Vec<GeometrySceneChunk>,
    topology: &AnalysisRenderTopology,
) {
    for mesh in &topology.meshes {
        let mut seen = HashSet::new();
        let mut vertices = Vec::new();
        for triangle in &mesh.triangles {
            for (a, b) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                for index in [a, b] {
                    let position = cast_position(mesh.vertices[index as usize]);
                    vertices.push(Vertex {
                        position,
                        color: [0.92, 0.92, 1.0, 0.35],
                        normal: [0.0, 0.0, 0.0],
                        tex_coords: [0.0, 0.0],
                    });
                }
            }
        }
        if vertices.is_empty() {
            continue;
        }
        let vertex_count = vertices.len();
        let mut material = Material::default();
        material.alpha_mode = AlphaMode::Blend;
        let mut chunk = GeometrySceneChunk::from_render_data(
            format!("{}:undeformed-outline", mesh.mesh_id),
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
        chunk.mesh_id = Some(mesh.mesh_id.clone());
        chunk.label = Some("Undeformed outline".into());
        chunks.push(chunk);
    }
}

fn automatic_deformation_scale(
    topology: &AnalysisRenderTopology,
    field: &FieldData,
) -> Result<f64, String> {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    let mut max_displacement = 0.0_f64;
    for mesh in &topology.meshes {
        for (vertex_index, position) in mesh.vertices.iter().enumerate() {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(position[axis]);
                maximum[axis] = maximum[axis].max(position[axis]);
            }
            if let Some(node) = mesh
                .vertex_volume_node_indices
                .get(vertex_index)
                .copied()
                .flatten()
            {
                let start = node
                    .checked_mul(field.components)
                    .ok_or("displacement index overflow")?;
                let values = field.range(start, 3)?;
                let magnitude = values.iter().map(|value| value * value).sum::<f64>().sqrt();
                if !magnitude.is_finite() {
                    return Err("displacement contains a non-finite value".into());
                }
                max_displacement = max_displacement.max(magnitude);
            }
        }
    }
    let diagonal = (0..3)
        .map(|axis| (maximum[axis] - minimum[axis]).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(if diagonal.is_finite() && max_displacement > f64::EPSILON {
        (0.12 * diagonal / max_displacement).max(1.0)
    } else {
        0.0
    })
}

fn deform_position(
    original: [f32; 3],
    node: Option<usize>,
    field: Option<&FieldData>,
    scale: f64,
) -> Result<[f32; 3], String> {
    let (Some(node), Some(field)) = (node, field) else {
        return Ok(original);
    };
    let start = node
        .checked_mul(field.components)
        .ok_or("displacement index overflow")?;
    let values = field.range(start, 3)?;
    Ok([0, 1, 2].map(|axis| original[axis] + (values[axis] * scale) as f32))
}

fn field_magnitude(field: &FieldData, entity: usize, stride: usize) -> Result<f64, String> {
    let mut total = 0.0;
    for sample in 0..stride {
        let start = entity
            .checked_mul(stride)
            .and_then(|value| value.checked_add(sample))
            .and_then(|value| value.checked_mul(field.components))
            .ok_or("field index overflow")?;
        let values = field.range(start, field.components)?;
        total += if field.components == 1 {
            values[0]
        } else {
            values.iter().map(|value| value * value).sum::<f64>().sqrt()
        };
    }
    let magnitude = total / stride as f64;
    if !magnitude.is_finite() {
        return Err(format!("field '{}' has a non-finite value", field.label));
    }
    Ok(magnitude)
}

fn scalar_color(value: f64, minimum: f64, maximum: f64) -> [f32; 4] {
    let t = ((value - minimum) / (maximum - minimum).max(f64::EPSILON)).clamp(0.0, 1.0) as f32;
    let stops = [
        [0.10, 0.20, 0.72],
        [0.02, 0.68, 0.90],
        [0.12, 0.82, 0.48],
        [0.98, 0.82, 0.12],
        [0.88, 0.10, 0.12],
    ];
    let index = (t * 4.0).floor().min(3.0) as usize;
    let fraction = t * 4.0 - index as f32;
    let mut color = [0.0; 4];
    for axis in 0..3 {
        color[axis] = stops[index][axis] * (1.0 - fraction) + stops[index + 1][axis] * fraction;
    }
    color[3] = 1.0;
    color
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisRenderMesh, AnalysisRenderTopologySource};
    use runmat_plot::{GeometrySceneDisplayMode, GeometryScenePresentation};

    #[test]
    fn large_surface_is_split_into_bounded_scene_chunks() {
        let topology = AnalysisRenderTopology {
            schema_version: "test".into(),
            source: AnalysisRenderTopologySource::AnalysisMesh,
            volume_element_count: 1,
            meshes: vec![AnalysisRenderMesh {
                mesh_id: "mesh".into(),
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                triangles: vec![[0, 1, 2]; MAX_TRIANGLES_PER_TILE + 1],
                regions: vec![],
                vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
                triangle_volume_element_indices: vec![Some(0); MAX_TRIANGLES_PER_TILE + 1],
            }],
        };
        let (scene, _) =
            scene_from_topology("run", &topology, None, None, "undeformed", 1.0, false).unwrap();
        let surfaces = scene
            .chunks
            .iter()
            .filter(|chunk| chunk.render_data.pipeline_type == PipelineType::Triangles)
            .collect::<Vec<_>>();
        assert_eq!(surfaces.len(), 2);
        assert!(surfaces
            .iter()
            .all(|chunk| chunk.triangle_count() <= MAX_TRIANGLES_PER_TILE));
    }

    #[test]
    fn selected_nodal_field_recolors_surface_and_preserves_wireframe_edges() {
        let topology = AnalysisRenderTopology {
            schema_version: "test".into(),
            source: AnalysisRenderTopologySource::AnalysisMesh,
            volume_element_count: 1,
            meshes: vec![AnalysisRenderMesh {
                mesh_id: "mesh".into(),
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                triangles: vec![[0, 1, 2]],
                regions: vec![],
                vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
                triangle_volume_element_indices: vec![Some(0)],
            }],
        };
        let field = FieldData::inline(
            AnalysisFieldLocation::Node,
            1,
            vec![0.0, 1.0, 2.0],
            "temperature",
            "K",
        );
        let (scene, scale) = scene_from_topology(
            "run",
            &topology,
            Some(&field),
            None,
            "undeformed",
            1.0,
            false,
        )
        .unwrap();
        assert_eq!(scale, 0.0);
        assert_eq!(scene.chunks.len(), 2);
        assert_ne!(
            scene.chunks[0].vertices[0].color,
            scene.chunks[0].vertices[2].color
        );
        assert_eq!(
            scene
                .overlay
                .as_ref()
                .unwrap()
                .scalar_legend
                .as_ref()
                .unwrap()
                .unit,
            "K"
        );
        assert!(scene.title.is_none());
        let wireframe = scene.nodes_with_presentation(&GeometryScenePresentation {
            display_mode: GeometrySceneDisplayMode::Wireframe,
            ..GeometryScenePresentation::default()
        });
        assert!(wireframe[0]
            .render_data
            .as_ref()
            .unwrap()
            .draw_calls
            .is_empty());
        assert!(!wireframe[1]
            .render_data
            .as_ref()
            .unwrap()
            .draw_calls
            .is_empty());
        let displacement = FieldData::inline(
            AnalysisFieldLocation::Node,
            3,
            vec![0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
            "structural.displacement",
            "m",
        );
        let (deformed, true_scale) = scene_from_topology(
            "run",
            &topology,
            Some(&field),
            Some(&displacement),
            "true",
            1.0,
            false,
        )
        .unwrap();
        assert_eq!(true_scale, 1.0);
        assert_ne!(
            deformed.chunks[0].vertices[1].position,
            scene.chunks[0].vertices[1].position
        );
        let (undeformed, _) = scene_from_topology(
            "run",
            &topology,
            Some(&field),
            Some(&displacement),
            "undeformed",
            1.0,
            false,
        )
        .unwrap();
        assert_eq!(
            undeformed.chunks[0].vertices[1].position,
            scene.chunks[0].vertices[1].position
        );
        assert_eq!(
            undeformed.chunks[0].vertices[1].color,
            deformed.chunks[0].vertices[1].color
        );
        let (exaggerated, exaggerated_scale) = scene_from_topology(
            "run",
            &topology,
            Some(&field),
            Some(&displacement),
            "exaggerated",
            2.0,
            false,
        )
        .unwrap();
        assert!(exaggerated_scale >= 2.0);
        assert_eq!(
            exaggerated.chunks[0].vertices[1].color,
            scene.chunks[0].vertices[1].color
        );
        let (outlined, _) = scene_from_topology(
            "run",
            &topology,
            Some(&field),
            Some(&displacement),
            "true",
            1.0,
            true,
        )
        .unwrap();
        assert_eq!(outlined.chunks.len(), deformed.chunks.len() + 1);
        assert_eq!(
            outlined
                .chunks
                .iter()
                .find(|chunk| chunk.label.as_deref() == Some("Undeformed outline"))
                .unwrap()
                .vertices[0]
                .position,
            [0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn integration_point_values_are_averaged_per_element() {
        let field = FieldData::inline(
            AnalysisFieldLocation::Element,
            1,
            vec![1.0, 3.0, 5.0, 7.0],
            "integration point temperature",
            "K",
        );
        assert_eq!(field_magnitude(&field, 0, 2).unwrap(), 2.0);
        assert_eq!(field_magnitude(&field, 1, 2).unwrap(), 6.0);
    }
}

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
        let mut material = Material::default();
        material.albedo = Vec4::ONE;
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
