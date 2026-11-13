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
    /// A mode shape: its amplitude is an arbitrary eigenvector normalization, not metres.
    normalized: bool,
    /// A nodal displacement (or displacement mode shape) that can displace the drawn shape.
    deforms: bool,
}

enum FieldValues {
    Paged {
        run_id: String,
        selection: AnalysisFieldSelection,
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
            FieldValues::Paged {
                run_id,
                selection,
                cache,
            } => {
                let mut result = Vec::with_capacity(len);
                let mut cursor = start;
                while cursor < end {
                    let page_start = cursor / FIELD_PAGE_VALUES * FIELD_PAGE_VALUES;
                    let mut cache = cache.borrow_mut();
                    if !cache.pages.contains_key(&page_start) {
                        let page = load_field_page(run_id, selection, page_start)?;
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
            normalized: false,
            deforms: location == AnalysisFieldLocation::Node && components == 3,
        }
    }

    /// The same values as a published mode shape (arbitrary normalization).
    #[cfg(test)]
    fn mode_shape(mut self) -> Self {
        self.normalized = true;
        self
    }
}

fn load_field_page(
    run_id: &str,
    selection: &AnalysisFieldSelection,
    offset: usize,
) -> Result<super::AnalysisFieldPageResult, String> {
    analysis_field_page_by_run_id_op(
        run_id,
        selection.clone(),
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
pub mod extrema;
pub mod volume;

/// How [`build_scene`] draws the body's shape: `mode` is `"undeformed"`, `"true"` or
/// `"exaggerated"`, `exaggeration` is the displacement factor (finite, 0 to 100), and
/// `show_undeformed_outline` overlays the undeformed outline.
#[derive(Clone, Copy, Debug)]
pub struct SceneShape<'a> {
    pub mode: &'a str,
    pub exaggeration: f64,
    pub show_undeformed_outline: bool,
}

/// Build one complete view of a run: `field` colors the surface and `deformation`, a nodal
/// vector field of the same run, displaces it. Both are explicit selections of published
/// fields (with their sample), so the caller decides from the field descriptors.
///
/// `scalar_range` colors the field over a known range, such as the field's true extrema from
/// [`extrema::field_extrema`]; without it the range is taken from the drawn surface.
/// `solver_mesh` is the run's canonical solver mesh artifact: a run whose publication keeps no
/// render tiles (frequency-sampled runs) is drawn from the boundary of that mesh, the same
/// projection the tiles store.
///
/// `detail` [`SurfaceDetail::Corners`] draws each quadratic boundary face as one corner
/// triangle (from the solver mesh), for a scene that must fit a payload budget; the returned
/// [`SurfaceApproximation`] states what that leaves out.
pub fn build_scene(
    run_id: &str,
    solver_mesh: Option<&str>,
    field: Option<&AnalysisFieldSelection>,
    deformation: Option<&AnalysisFieldSelection>,
    scalar_range: Option<(f64, f64)>,
    shape: SceneShape<'_>,
    detail: super::SurfaceDetail,
) -> Result<(GeometryScene, f64, SurfaceApproximation), String> {
    let SceneShape {
        mode: shape_mode,
        exaggeration,
        show_undeformed_outline,
    } = shape;
    if !matches!(shape_mode, "undeformed" | "true" | "exaggerated") {
        return Err(format!("unknown FEA shape mode '{shape_mode}'"));
    }
    if !exaggeration.is_finite() || !(0.0..=100.0).contains(&exaggeration) {
        return Err("FEA exaggeration must be finite and between 0 and 100".into());
    }
    let load_from_solver_mesh = |detail| -> Result<_, String> {
        solver_mesh
            .map(|path| {
                super::solver_mesh_artifact::load_solver_mesh_artifact(std::path::Path::new(path))
                    .map_err(|error| error.to_string())
            })
            .transpose()?
            .and_then(|mesh| super::render_topology_from_solver_mesh_at(Some(&mesh), detail))
            .ok_or_else(|| format!("FEA run '{run_id}' has no persisted render topology"))
    };
    let (topology, dropped) = match detail {
        super::SurfaceDetail::Corners => load_from_solver_mesh(detail)?,
        super::SurfaceDetail::Full => match storage::load_render_topology(run_id)? {
            Some(topology) => (topology, Vec::new()),
            None => load_from_solver_mesh(detail)?,
        },
    };
    let field_data = field
        .map(|selection| load_field(run_id, selection))
        .transpose()?;
    let separate_deformation = match deformation {
        Some(selection) if shape_mode != "undeformed" && field != Some(selection) => {
            Some(load_field(run_id, selection)?)
        }
        _ => None,
    };
    let deformation_data = match deformation {
        None => None,
        Some(_) if shape_mode == "undeformed" => None,
        Some(selection) if field == Some(selection) => field_data.as_ref(),
        Some(_) => separate_deformation.as_ref(),
    };
    if let Some(data) = deformation_data {
        if data.location != AnalysisFieldLocation::Node || data.components < 3 {
            return Err(format!(
                "deformation field '{}' is not a nodal vector field",
                data.label
            ));
        }
    }
    let stats = scalar_range.map(|range| SceneStats {
        element_count: topology.volume_element_count,
        scalar_range: Some(range),
        automatic_scale: None,
    });
    let (scene, scale) = scene_from_topology_with_stats(
        run_id,
        &topology,
        field_data.as_ref(),
        deformation_data,
        shape_mode,
        exaggeration,
        show_undeformed_outline,
        stats,
    )?;
    let approximation = SurfaceApproximation::of_dropped(
        detail,
        &topology,
        &dropped,
        field_data.as_ref(),
        deformation_data,
        scale,
    )?;
    Ok((scene, scale, approximation))
}

/// What a drawn surface leaves out of the solution on the boundary.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SurfaceApproximation {
    /// `full`: every boundary node is drawn where it is. `corners`: each quadratic face is one
    /// triangle through its corners.
    pub detail: &'static str,
    /// Largest distance between a dropped midside node (as drawn, deformation included) and the
    /// midpoint of the drawn edge, in metres.
    pub position_error_m: f64,
    /// Largest difference between the colored value at a dropped midside node and the linear
    /// color drawn there; zero for element fields, absent without a field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_error: Option<f64>,
}

impl SurfaceApproximation {
    fn of_dropped(
        detail: super::SurfaceDetail,
        topology: &AnalysisRenderTopology,
        dropped: &[[u32; 3]],
        field: Option<&FieldData>,
        deformation: Option<&FieldData>,
        scale: f64,
    ) -> Result<Self, String> {
        let mut approximation = Self {
            detail: match detail {
                super::SurfaceDetail::Full => "full",
                super::SurfaceDetail::Corners => "corners",
            },
            position_error_m: 0.0,
            value_error: field.map(|_| 0.0),
        };
        let Some(mesh) = topology.meshes.first() else {
            return Ok(approximation);
        };
        for &[a, b, midside] in dropped {
            let drawn = |node: u32| {
                deform_position(
                    cast_position(mesh.vertices[node as usize]),
                    Some(node as usize),
                    deformation,
                    scale,
                )
                .map(|p| p.map(f64::from))
            };
            let (pa, pb, pm) = (drawn(a)?, drawn(b)?, drawn(midside)?);
            let gap = (0..3)
                .map(|axis| (pm[axis] - 0.5 * (pa[axis] + pb[axis])).powi(2))
                .sum::<f64>()
                .sqrt();
            approximation.position_error_m = approximation.position_error_m.max(gap);
            if let Some(field) = field.filter(|field| field.location == AnalysisFieldLocation::Node)
            {
                let value = |node: u32| field_magnitude(field, node as usize, 1);
                let error = (value(midside)? - 0.5 * (value(a)? + value(b)?)).abs();
                if let Some(worst) = approximation.value_error.as_mut() {
                    *worst = worst.max(error);
                }
            }
        }
        Ok(approximation)
    }
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
                    deformation::automatic_scale(diagonal, summary.maximum, field.normalized)
                })
        });
        Self {
            element_count: selection.volume_element_count,
            scalar_range,
            automatic_scale,
        }
    }
}

fn load_field(run_id: &str, selection: &AnalysisFieldSelection) -> Result<FieldData, String> {
    let field_id = selection.field_id.as_str();
    let page = load_field_page(run_id, selection, 0)?;
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
    let normalized = deformation::is_mode_shape(descriptor);
    let deforms = deformation::displaces(descriptor);
    Ok(FieldData {
        location: descriptor.location,
        components,
        value_count: page.total_count,
        values: FieldValues::Paged {
            run_id: run_id.into(),
            selection: selection.clone(),
            cache: RefCell::new(cache),
        },
        label: field_id.to_string(),
        unit: descriptor.unit.clone().unwrap_or_default(),
        normalized,
        deforms,
    })
}

mod geometry;
use geometry::*;

#[cfg(test)]
mod tests;

/// Surface triangle count above which a scene carries feature edges instead of every mesh edge:
/// a full wireframe on a dense mesh buries the field colors and is drawn only when asked for.
pub const FULL_MESH_EDGE_TRIANGLE_LIMIT: usize = 20_000;

/// Dihedral angle above which an edge between two surface triangles is a feature (crease) edge.
pub const FEATURE_EDGE_CREASE_DEGREES: f32 = 30.0;

/// Preserve explicit topology lines so shaded, x-ray and wireframe modes share one scene. Dense
/// scenes get feature edges (creases and open boundary loops) instead of the full wireframe.
pub fn append_solver_mesh_edges(scene: &mut GeometryScene) {
    let triangle_count = scene
        .chunks
        .iter()
        .filter(|chunk| chunk.render_data.pipeline_type == PipelineType::Triangles)
        .map(|chunk| {
            chunk
                .indices
                .as_ref()
                .map_or(chunk.vertices.len(), Vec::len)
                / 3
        })
        .sum::<usize>();
    if triangle_count > FULL_MESH_EDGE_TRIANGLE_LIMIT {
        append_feature_edges(scene, FEATURE_EDGE_CREASE_DEGREES);
    } else {
        append_full_mesh_edges(scene);
    }
}

/// Suffix of the chunk ids of a full solver-mesh wireframe.
const FULL_MESH_EDGES_SUFFIX: &str = ":solver-mesh-edges";

/// Edges for a view that colors a result field: a full wireframe drawn over the field reads as a
/// grid of dots at poster size, so the full mesh edges give way to feature edges (creases and
/// open boundary loops). Scenes of the mesh itself keep their full wireframe.
pub fn use_result_view_edges(scene: &mut GeometryScene) {
    let before = scene.chunks.len();
    scene
        .chunks
        .retain(|chunk| !chunk.chunk_id.ends_with(FULL_MESH_EDGES_SUFFIX));
    if scene.chunks.len() != before {
        append_feature_edges(scene, FEATURE_EDGE_CREASE_DEGREES);
        scene.revision = scene.revision.saturating_add(1);
    }
}

/// Edges where adjacent surface triangles meet at more than `crease_degrees`, plus edges with
/// a single adjacent triangle (open boundary loops), taken across every surface chunk so tile
/// seams are not mistaken for boundaries.
pub fn append_feature_edges(scene: &mut GeometryScene, crease_degrees: f32) {
    type Key = [u32; 3];
    let mut edges: HashMap<(Key, Key), (Vertex, Vertex, Vec<Vec3>)> = HashMap::new();
    for chunk in &scene.chunks {
        if chunk.render_data.pipeline_type != PipelineType::Triangles {
            continue;
        }
        let indices = chunk
            .indices
            .clone()
            .unwrap_or_else(|| (0..chunk.vertices.len() as u32).collect());
        for triangle in indices.chunks_exact(3) {
            let corners =
                [0, 1, 2].map(|corner| chunk.vertices.get(triangle[corner] as usize).copied());
            let [Some(a), Some(b), Some(c)] = corners else {
                continue;
            };
            let normal = (Vec3::from_array(b.position) - Vec3::from_array(a.position))
                .cross(Vec3::from_array(c.position) - Vec3::from_array(a.position))
                .normalize_or_zero();
            for (start, end) in [(a, b), (b, c), (c, a)] {
                let start_key = start.position.map(f32::to_bits);
                let end_key = end.position.map(f32::to_bits);
                let key = if start_key < end_key {
                    (start_key, end_key)
                } else {
                    (end_key, start_key)
                };
                edges
                    .entry(key)
                    .or_insert_with(|| (start, end, Vec::new()))
                    .2
                    .push(normal);
            }
        }
    }
    let cosine = crease_degrees.to_radians().cos();
    let mut keys = edges.keys().copied().collect::<Vec<_>>();
    keys.sort_unstable();
    let mut vertices = Vec::new();
    for key in keys {
        let (mut start, mut end, normals) = edges.remove(&key).expect("collected edge");
        let feature = normals.len() == 1
            || normals.iter().enumerate().any(|(i, left)| {
                normals[i + 1..]
                    .iter()
                    .any(|right| left.dot(*right) < cosine)
            });
        if !feature {
            continue;
        }
        start.color = [0.12, 0.14, 0.18, 0.9];
        end.color = start.color;
        vertices.extend([start, end]);
    }
    if vertices.is_empty() {
        return;
    }
    let vertex_count = vertices.len();
    let mut edge_chunk = GeometrySceneChunk::from_render_data(
        "feature-edges",
        RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: None,
            material: Material {
                albedo: Vec4::ONE,
                ..Material::default()
            },
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
    edge_chunk.label = Some("Feature edges".into());
    scene.chunks.push(edge_chunk);
}

fn append_full_mesh_edges(scene: &mut GeometryScene) {
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
            format!("{}{FULL_MESH_EDGES_SUFFIX}", chunk.chunk_id),
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
