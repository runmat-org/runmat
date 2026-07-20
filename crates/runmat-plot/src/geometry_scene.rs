//! Chunked geometry scenes for CAD and FEA visualization.
//!
//! This is intentionally rendering-domain data. CAD import, semantic ownership,
//! and FEA result storage stay in their own crates; this module describes the
//! mesh chunks that the plot renderer can keep resident and redraw efficiently.

use crate::core::{
    AlphaMode, BoundingBox, Camera, DrawCall, Material, PipelineType, RenderData, SceneNode, Vertex,
};
use glam::{Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;

const SECTION_DISTANCE_EPSILON: f32 = 1.0e-6;
const GEOMETRY_SELECTED_REGION_COLOR: [f32; 4] = [0.98, 0.45, 0.12, 1.0];
const GEOMETRY_HOVER_REGION_COLOR: [f32; 4] = [0.43, 0.78, 1.0, 1.0];
const GEOMETRY_MATERIAL_REGION_COLOR: [f32; 4] = [0.38, 0.58, 0.96, 0.92];
const GEOMETRY_BOUNDARY_REGION_COLOR: [f32; 4] = [0.22, 0.82, 0.52, 0.96];
const GEOMETRY_DRIVING_REGION_COLOR: [f32; 4] = [0.98, 0.38, 0.28, 0.96];

#[derive(Debug, Clone)]
pub struct GeometryScene {
    pub scene_id: String,
    pub revision: u64,
    pub title: Option<String>,
    pub overlay: Option<GeometrySceneOverlay>,
    pub chunks: Vec<GeometrySceneChunk>,
    pub bounds: BoundingBox,
    pub show_grid: bool,
    pub axis_equal: bool,
}

impl GeometryScene {
    pub fn new(
        scene_id: impl Into<String>,
        revision: u64,
        chunks: Vec<GeometrySceneChunk>,
    ) -> Self {
        let bounds = combined_chunk_bounds(&chunks);
        Self {
            scene_id: scene_id.into(),
            revision,
            title: None,
            overlay: None,
            chunks,
            bounds,
            show_grid: false,
            axis_equal: true,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_overlay(mut self, overlay: GeometrySceneOverlay) -> Self {
        self.overlay = Some(overlay);
        self
    }

    pub fn append_chunks(&mut self, chunks: impl IntoIterator<Item = GeometrySceneChunk>) {
        self.chunks.extend(chunks);
        self.revision = self.revision.saturating_add(1);
        self.bounds = combined_chunk_bounds(&self.chunks);
    }

    pub fn set_overlay(&mut self, overlay: GeometrySceneOverlay) {
        self.overlay = Some(overlay);
        self.revision = self.revision.saturating_add(1);
    }

    pub fn cache_key(&self) -> GeometrySceneCacheKey {
        GeometrySceneCacheKey {
            scene_id: self.scene_id.clone(),
            revision: self.revision,
            chunk_count: self.chunks.len(),
            vertex_count: self.vertex_count(),
            index_count: self.index_count(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.render_data.vertex_count())
            .sum()
    }

    pub fn index_count(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.indices.as_ref().map(Vec::len).unwrap_or(0))
            .sum()
    }

    pub fn triangle_count(&self) -> usize {
        self.chunks
            .iter()
            .map(GeometrySceneChunk::triangle_count)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn nodes(&self) -> Vec<SceneNode> {
        self.nodes_with_presentation(&GeometryScenePresentation::default())
    }

    pub fn nodes_with_presentation(
        &self,
        presentation: &GeometryScenePresentation,
    ) -> Vec<SceneNode> {
        let mut nodes: Vec<SceneNode> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| SceneNode {
                id: self.chunk_node_id(index, &chunk.chunk_id),
                name: chunk
                    .label
                    .clone()
                    .unwrap_or_else(|| format!("Geometry chunk {}", index + 1)),
                transform: Mat4::IDENTITY,
                visible: chunk.visible,
                cast_shadows: false,
                receive_shadows: false,
                axes_index: 0,
                parent: None,
                children: Vec::new(),
                render_data: Some(chunk.render_data_with_presentation(presentation)),
                bounds: chunk.bounds,
                lod_levels: Vec::new(),
                current_lod: 0,
            })
            .collect();
        nodes.extend(self.annotation_nodes(presentation));
        nodes
    }

    pub fn chunk_node_id(&self, index: usize, chunk_id: &str) -> u64 {
        stable_node_id(&self.scene_id, self.revision, index, chunk_id)
    }

    fn annotation_nodes(&self, presentation: &GeometryScenePresentation) -> Vec<SceneNode> {
        if presentation.region_annotations.is_empty() {
            return Vec::new();
        }

        let mut point_vertices = Vec::new();
        let mut line_vertices = Vec::new();
        let arrow_length = annotation_arrow_length(self.bounds);

        for annotation in &presentation.region_annotations {
            for chunk in &self.chunks {
                if !chunk.visible || chunk.render_data.pipeline_type != PipelineType::Triangles {
                    continue;
                }
                let Some(anchor) = chunk.region_anchor(&annotation.region_id) else {
                    continue;
                };
                let color = annotation
                    .color
                    .unwrap_or_else(|| geometry_region_role_color(annotation.role.as_deref()));
                let mut marker = vertex(
                    anchor.to_array(),
                    color,
                    [0.0, 0.0, annotation.size.unwrap_or(15.0)],
                );
                marker.tex_coords = [1.0, 1.0];
                point_vertices.push(marker);

                if let Some(direction) = annotation
                    .direction
                    .and_then(normalized_annotation_direction)
                {
                    append_annotation_arrow(
                        &mut line_vertices,
                        anchor,
                        direction,
                        arrow_length,
                        color,
                    );
                }
            }
        }

        let mut nodes = Vec::new();
        if !point_vertices.is_empty() {
            nodes.push(self.annotation_node(
                "FEA region markers",
                "__fea_annotations:markers",
                self.chunks.len(),
                PipelineType::Points,
                point_vertices,
            ));
        }
        if !line_vertices.is_empty() {
            nodes.push(self.annotation_node(
                "FEA load vectors",
                "__fea_annotations:vectors",
                self.chunks.len() + 1,
                PipelineType::Lines,
                line_vertices,
            ));
        }
        nodes
    }

    fn annotation_node(
        &self,
        name: &str,
        chunk_id: &str,
        index: usize,
        pipeline_type: PipelineType,
        vertices: Vec<Vertex>,
    ) -> SceneNode {
        let vertex_count = vertices.len();
        let bounds = bounds_from_vertices(&vertices);
        SceneNode {
            id: stable_node_id(&self.scene_id, self.revision, index, chunk_id),
            name: name.to_string(),
            transform: Mat4::IDENTITY,
            visible: true,
            cast_shadows: false,
            receive_shadows: false,
            axes_index: 0,
            parent: None,
            children: Vec::new(),
            render_data: Some(RenderData {
                pipeline_type,
                vertices,
                indices: None,
                gpu_vertices: None,
                bounds: Some(bounds),
                material: Material {
                    albedo: Vec4::ONE,
                    alpha_mode: AlphaMode::Blend,
                    double_sided: true,
                    ..Default::default()
                },
                draw_calls: vec![DrawCall {
                    vertex_offset: 0,
                    vertex_count,
                    index_offset: None,
                    index_count: None,
                    instance_count: 1,
                }],
                image: None,
            }),
            bounds,
            lod_levels: Vec::new(),
            current_lod: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryScenePresentation {
    pub selected_region_id: Option<String>,
    #[serde(default)]
    pub selected_region_ids: Vec<String>,
    pub hovered_region_id: Option<String>,
    #[serde(default)]
    pub region_highlights: Vec<GeometrySceneRegionHighlight>,
    #[serde(default)]
    pub region_annotations: Vec<GeometrySceneRegionAnnotation>,
    #[serde(default)]
    pub display_mode: GeometrySceneDisplayMode,
    #[serde(default = "default_edge_overlay_enabled")]
    pub edge_overlay_enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_owner_node_ids: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub isolated_owner_node_ids: Option<Vec<String>>,
    #[serde(
        default,
        deserialize_with = "deserialize_explicit_optional_section",
        skip_serializing_if = "Option::is_none"
    )]
    pub section: Option<Option<GeometrySceneSection>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view_preset: Option<GeometrySceneViewPreset>,
}

impl Default for GeometryScenePresentation {
    fn default() -> Self {
        Self {
            selected_region_id: None,
            selected_region_ids: Vec::new(),
            hovered_region_id: None,
            region_highlights: Vec::new(),
            region_annotations: Vec::new(),
            display_mode: GeometrySceneDisplayMode::Shaded,
            edge_overlay_enabled: true,
            hidden_owner_node_ids: None,
            isolated_owner_node_ids: None,
            section: None,
            view_preset: None,
        }
    }
}

impl GeometryScenePresentation {
    pub(crate) fn rewrites_geometry_vertices(&self) -> bool {
        self.hovered_region_id.is_some()
            || self.selected_region_id.is_some()
            || !self.selected_region_ids.is_empty()
            || !self.region_highlights.is_empty()
            || self.active_section().is_some()
    }

    pub(crate) fn resolves_owner_visibility(&self) -> bool {
        self.hidden_owner_node_ids.is_some() || self.isolated_owner_node_ids.is_some()
    }

    pub(crate) fn resolved_hidden_owner_node_ids<'a, I>(
        &self,
        all_owner_node_ids: I,
        current_hidden_owner_node_ids: &BTreeSet<String>,
    ) -> BTreeSet<String>
    where
        I: IntoIterator<Item = &'a str>,
    {
        if !self.resolves_owner_visibility() {
            return current_hidden_owner_node_ids.clone();
        }

        let mut hidden = BTreeSet::new();
        if let Some(isolated_owner_node_ids) = self.isolated_owner_node_ids.as_ref() {
            let isolated =
                normalized_owner_node_id_set(isolated_owner_node_ids.iter().map(String::as_str));
            for owner_id in all_owner_node_ids {
                let normalized = owner_id.trim();
                if !normalized.is_empty() && !isolated.contains(normalized) {
                    hidden.insert(normalized.to_string());
                }
            }
        }

        if let Some(hidden_owner_node_ids) = self.hidden_owner_node_ids.as_ref() {
            hidden.extend(normalized_owner_node_id_set(
                hidden_owner_node_ids.iter().map(String::as_str),
            ));
        }
        hidden
    }

    pub(crate) fn resolves_section(&self) -> bool {
        self.section.is_some()
    }

    pub(crate) fn active_section(&self) -> Option<&GeometrySceneSection> {
        self.section.as_ref().and_then(Option::as_ref)
    }
}

fn normalized_owner_node_id_set<'a, I>(ids: I) -> BTreeSet<String>
where
    I: IntoIterator<Item = &'a str>,
{
    ids.into_iter()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn deserialize_explicit_optional_section<'de, D>(
    deserializer: D,
) -> Result<Option<Option<GeometrySceneSection>>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<GeometrySceneSection>::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySceneSection {
    pub plane: GeometrySceneSectionPlane,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySceneSectionPlane {
    pub normal: [f32; 3],
    #[serde(default)]
    pub origin: Option<[f32; 3]>,
    #[serde(default)]
    pub offset: Option<f32>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometrySceneViewPreset {
    Perspective,
    Isometric,
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySceneRegionHighlight {
    pub region_id: String,
    #[serde(default)]
    pub color: Option<[f32; 4]>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySceneRegionAnnotation {
    pub region_id: String,
    #[serde(default)]
    pub color: Option<[f32; 4]>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub direction: Option<[f32; 3]>,
    #[serde(default)]
    pub size: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GeometrySceneDisplayMode {
    Shaded,
    Edges,
    Wireframe,
}

impl Default for GeometrySceneDisplayMode {
    fn default() -> Self {
        Self::Shaded
    }
}

impl GeometrySceneDisplayMode {
    fn alpha(self, edge_overlay_enabled: bool) -> f32 {
        match self {
            Self::Shaded => 1.0,
            Self::Edges if edge_overlay_enabled => 0.84,
            Self::Edges => 0.94,
            Self::Wireframe => 0.16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeometryScenePickRequest {
    pub camera: Camera,
    pub surface_size: [f32; 2],
    pub position: [f32; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryScenePickResult {
    pub mesh_id: Option<String>,
    pub chunk_id: String,
    pub triangle_index: usize,
    pub region_id: Option<String>,
    pub region_label: Option<String>,
    pub region_tag: Option<String>,
    pub distance: f32,
    pub position: [f32; 3],
}

fn default_edge_overlay_enabled() -> bool {
    true
}

#[derive(Debug, Clone)]
pub struct GeometryScenePickIndex {
    scene_key: GeometrySceneCacheKey,
    triangles: Vec<IndexedTriangle>,
    nodes: Vec<PickBvhNode>,
    root: Option<usize>,
}

impl GeometryScenePickIndex {
    pub fn build(scene: &GeometryScene) -> Self {
        let mut triangles = Vec::with_capacity(scene.triangle_count());
        for chunk in &scene.chunks {
            if !chunk.visible || chunk.render_data.pipeline_type != PipelineType::Triangles {
                continue;
            }
            let Some(indices) = chunk.indices.as_ref() else {
                continue;
            };
            for (triangle_index, triangle) in indices.chunks_exact(3).enumerate() {
                let a = chunk.vertices.get(triangle[0] as usize);
                let b = chunk.vertices.get(triangle[1] as usize);
                let c = chunk.vertices.get(triangle[2] as usize);
                let (Some(a), Some(b), Some(c)) = (a, b, c) else {
                    continue;
                };
                let a = Vec3::from_array(a.position);
                let b = Vec3::from_array(b.position);
                let c = Vec3::from_array(c.position);
                let bounds = triangle_bounds(a, b, c);
                let region = chunk.region_for_triangle(triangle_index as u32);
                triangles.push(IndexedTriangle {
                    a,
                    b,
                    c,
                    bounds,
                    centroid: (a + b + c) / 3.0,
                    mesh_id: chunk.mesh_id.clone(),
                    chunk_id: chunk.chunk_id.clone(),
                    triangle_index,
                    region_id: region.map(|item| item.region_id.clone()),
                    region_label: region.and_then(|item| item.label.clone()),
                    region_tag: region.and_then(|item| item.tag.clone()),
                });
            }
        }

        let mut nodes = Vec::new();
        let triangle_count = triangles.len();
        let root = build_pick_bvh_node(&mut nodes, &mut triangles, 0, triangle_count);
        Self {
            scene_key: scene.cache_key(),
            triangles,
            nodes,
            root,
        }
    }

    pub fn scene_key(&self) -> &GeometrySceneCacheKey {
        &self.scene_key
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    pub fn pick(&self, request: GeometryScenePickRequest) -> Option<GeometryScenePickResult> {
        if request.surface_size[0] <= 0.0 || request.surface_size[1] <= 0.0 {
            return None;
        }
        let mut camera = request.camera;
        let screen_size = Vec2::new(request.surface_size[0], request.surface_size[1]);
        let screen_pos = Vec2::new(request.position[0], request.position[1]);
        let origin = camera.screen_to_world(screen_pos, screen_size, 0.0);
        let far = camera.screen_to_world(screen_pos, screen_size, 1.0);
        let direction = (far - origin).normalize_or_zero();
        if direction.length_squared() <= f32::EPSILON {
            return None;
        }
        let ray = PickRay { origin, direction };
        let mut best: Option<PickHit> = None;
        if let Some(root) = self.root {
            self.pick_node(root, &ray, &mut best);
        }
        let hit = best?;
        let triangle = self.triangles.get(hit.triangle_index)?;
        Some(GeometryScenePickResult {
            mesh_id: triangle.mesh_id.clone(),
            chunk_id: triangle.chunk_id.clone(),
            triangle_index: triangle.triangle_index,
            region_id: triangle.region_id.clone(),
            region_label: triangle.region_label.clone(),
            region_tag: triangle.region_tag.clone(),
            distance: hit.distance,
            position: (ray.origin + ray.direction * hit.distance).to_array(),
        })
    }

    fn pick_node(&self, node_index: usize, ray: &PickRay, best: &mut Option<PickHit>) {
        let Some(node) = self.nodes.get(node_index) else {
            return;
        };
        let max_distance = best
            .as_ref()
            .map(|hit| hit.distance)
            .unwrap_or(f32::INFINITY);
        let Some(bounds_distance) = ray_intersects_bounds(ray, node.bounds) else {
            return;
        };
        if bounds_distance > max_distance {
            return;
        }
        match node.kind {
            PickBvhNodeKind::Leaf { start, end } => {
                for triangle_index in start..end {
                    let Some(triangle) = self.triangles.get(triangle_index) else {
                        continue;
                    };
                    if let Some(hit_distance) =
                        ray_intersects_triangle(ray, triangle.a, triangle.b, triangle.c)
                    {
                        if hit_distance > 0.0
                            && hit_distance
                                < best
                                    .as_ref()
                                    .map(|hit| hit.distance)
                                    .unwrap_or(f32::INFINITY)
                        {
                            *best = Some(PickHit {
                                triangle_index,
                                distance: hit_distance,
                            });
                        }
                    }
                }
            }
            PickBvhNodeKind::Branch { left, right } => {
                self.pick_node(left, ray, best);
                self.pick_node(right, ray, best);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometrySceneCompleteness {
    Complete,
    Loading,
    BoundedPreview,
    FailedComplete,
}

#[derive(Debug, Clone)]
pub struct GeometrySceneOverlay {
    pub source_name: Option<String>,
    pub status: GeometrySceneCompleteness,
    pub quality_label: Option<String>,
    pub format: Option<String>,
    pub source_label: Option<String>,
    pub allow_create_fea_study: bool,
    pub byte_count: Option<u64>,
    pub mesh_count: usize,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub progress_percent: Option<f64>,
    pub region_count: usize,
    pub mapped_region_count: usize,
    pub assembly_nodes: Vec<GeometrySceneAssemblyNode>,
    pub regions: Vec<GeometrySceneRegionSummary>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GeometrySceneAssemblyNode {
    pub node_id: String,
    pub label: String,
    pub children: Vec<GeometrySceneAssemblyNode>,
}

#[derive(Debug, Clone)]
pub struct GeometrySceneRegionSummary {
    pub region_id: String,
    pub label: String,
    pub tag: Option<String>,
    pub kind: Option<String>,
    pub triangle_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometrySceneCacheKey {
    pub scene_id: String,
    pub revision: u64,
    pub chunk_count: usize,
    pub vertex_count: usize,
    pub index_count: usize,
}

#[derive(Debug, Clone)]
pub struct GeometrySceneChunk {
    pub chunk_id: String,
    pub mesh_id: Option<String>,
    pub label: Option<String>,
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
    pub render_data: RenderData,
    pub bounds: BoundingBox,
    pub material: Material,
    pub regions: Vec<GeometrySceneRegion>,
    pub owner_node_ids: Vec<String>,
    pub visible: bool,
}

impl GeometrySceneChunk {
    pub fn indexed_triangles(
        chunk_id: impl Into<String>,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        material: Material,
    ) -> Self {
        let bounds = bounds_from_vertices(&vertices);
        let vertex_count = vertices.len();
        let index_count = indices.len();
        let render_data = RenderData {
            pipeline_type: PipelineType::Triangles,
            vertices: vertices.clone(),
            indices: Some(indices.clone()),
            gpu_vertices: None,
            bounds: Some(bounds),
            material: material.clone(),
            draw_calls: vec![DrawCall {
                vertex_offset: 0,
                vertex_count,
                index_offset: Some(0),
                index_count: Some(index_count),
                instance_count: 1,
            }],
            image: None,
        };
        Self {
            chunk_id: chunk_id.into(),
            mesh_id: None,
            label: None,
            vertices,
            indices: Some(indices),
            render_data,
            bounds,
            material,
            regions: Vec::new(),
            owner_node_ids: Vec::new(),
            visible: true,
        }
    }

    pub fn from_render_data(chunk_id: impl Into<String>, render_data: RenderData) -> Self {
        let material = render_data.material.clone();
        let vertices = render_data.vertices.clone();
        let indices = render_data.indices.clone();
        let bounds = render_data
            .bounds
            .unwrap_or_else(|| bounds_from_vertices(&vertices));
        Self {
            chunk_id: chunk_id.into(),
            mesh_id: None,
            label: None,
            vertices,
            indices,
            render_data,
            bounds,
            material,
            regions: Vec::new(),
            owner_node_ids: Vec::new(),
            visible: true,
        }
    }

    pub fn with_mesh_id(mut self, mesh_id: impl Into<String>) -> Self {
        self.mesh_id = Some(mesh_id.into());
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_regions(mut self, regions: Vec<GeometrySceneRegion>) -> Self {
        self.regions = regions;
        self
    }

    pub fn with_owner_node_ids(mut self, owner_node_ids: Vec<String>) -> Self {
        self.owner_node_ids = owner_node_ids;
        self
    }

    pub fn triangle_count(&self) -> usize {
        if self.render_data.pipeline_type != PipelineType::Triangles {
            return 0;
        }
        self.indices
            .as_ref()
            .map(|indices| indices.len() / 3)
            .unwrap_or_else(|| self.render_data.vertex_count() / 3)
    }

    pub fn render_data(&self) -> RenderData {
        self.render_data.clone()
    }

    pub fn render_data_with_presentation(
        &self,
        presentation: &GeometryScenePresentation,
    ) -> RenderData {
        let mut render_data = self.render_data.clone();
        if presentation.rewrites_geometry_vertices() {
            // Presentation overlays rewrite CPU-side vertices/indices. A stale
            // GPU vertex source would otherwise bypass the rewritten data.
            render_data.gpu_vertices = None;
        }
        let is_edge_chunk = self.is_edge_chunk();
        match presentation.display_mode {
            GeometrySceneDisplayMode::Wireframe if !is_edge_chunk => {
                render_data.material.alpha_mode = AlphaMode::Blend;
                render_data.material.albedo.w = presentation
                    .display_mode
                    .alpha(presentation.edge_overlay_enabled);
            }
            GeometrySceneDisplayMode::Wireframe => {}
            GeometrySceneDisplayMode::Edges
                if is_edge_chunk && !presentation.edge_overlay_enabled =>
            {
                for vertex in &mut render_data.vertices {
                    vertex.color[3] = 0.0;
                }
            }
            GeometrySceneDisplayMode::Edges | GeometrySceneDisplayMode::Shaded => {
                if !is_edge_chunk {
                    let alpha = presentation
                        .display_mode
                        .alpha(presentation.edge_overlay_enabled)
                        .min(render_data.material.albedo.w);
                    render_data.material.albedo.w = alpha;
                    render_data.material.alpha_mode = if alpha < 0.98 {
                        AlphaMode::Blend
                    } else {
                        render_data.material.alpha_mode
                    };
                    for vertex in &mut render_data.vertices {
                        vertex.color[3] = vertex.color[3].min(alpha);
                    }
                }
            }
        }

        for highlight in &presentation.region_highlights {
            self.apply_region_color(
                &mut render_data,
                &highlight.region_id,
                highlight
                    .color
                    .unwrap_or_else(|| geometry_region_role_color(highlight.role.as_deref())),
            );
        }
        if let Some(region_id) = presentation.hovered_region_id.as_deref() {
            self.apply_region_color(&mut render_data, region_id, GEOMETRY_HOVER_REGION_COLOR);
        }
        for region_id in &presentation.selected_region_ids {
            let colored = self.apply_region_color(
                &mut render_data,
                region_id,
                GEOMETRY_SELECTED_REGION_COLOR,
            );
            log::info!(
                target: "runmat_plot",
                "geometry_scene.selection_color chunk_id={} region_id={} colored_triangles={}",
                self.chunk_id,
                region_id,
                colored
            );
        }
        if let Some(region_id) = presentation.selected_region_id.as_deref() {
            let colored = self.apply_region_color(
                &mut render_data,
                region_id,
                GEOMETRY_SELECTED_REGION_COLOR,
            );
            log::info!(
                target: "runmat_plot",
                "geometry_scene.selection_color chunk_id={} region_id={} colored_triangles={}",
                self.chunk_id,
                region_id,
                colored
            );
        }
        if let Some(section) = presentation.active_section() {
            render_data = self.render_data_with_section(render_data, section);
        }
        render_data
    }

    fn render_data_with_section(
        &self,
        render_data: RenderData,
        section: &GeometrySceneSection,
    ) -> RenderData {
        let Some(plane) = section_plane(&section.plane) else {
            return render_data;
        };
        match render_data.pipeline_type {
            PipelineType::Triangles => clip_triangle_render_data(render_data, plane),
            PipelineType::Lines => clip_line_render_data(render_data, plane),
            _ => render_data,
        }
    }

    fn is_edge_chunk(&self) -> bool {
        self.render_data.pipeline_type == PipelineType::Lines
            || self.chunk_id.contains(":edges")
            || self
                .label
                .as_ref()
                .map(|label| label.to_ascii_lowercase().contains("edge"))
                .unwrap_or(false)
    }

    fn region_for_triangle(&self, triangle_index: u32) -> Option<&GeometrySceneRegion> {
        self.regions.iter().find(|region| {
            region.triangle_ranges.iter().any(|range| {
                triangle_index >= range.start
                    && triangle_index < range.start.saturating_add(range.count)
            })
        })
    }

    fn region_anchor(&self, region_id: &str) -> Option<Vec3> {
        if self.render_data.pipeline_type != PipelineType::Triangles {
            return None;
        }
        let region = self
            .regions
            .iter()
            .find(|item| item.region_id == region_id)?;
        let mut weighted_centroid = Vec3::ZERO;
        let mut total_area = 0.0_f32;
        let mut fallback_centroid = Vec3::ZERO;
        let mut fallback_count = 0_usize;

        for range in &region.triangle_ranges {
            let start = range.start as usize;
            let end = start.saturating_add(range.count as usize);
            for triangle_index in start..end {
                let Some((a, b, c)) = self.triangle_vertices(triangle_index) else {
                    continue;
                };
                let centroid = (a + b + c) / 3.0;
                let area = (b - a).cross(c - a).length() * 0.5;
                if area.is_finite() && area > 1.0e-8 {
                    weighted_centroid += centroid * area;
                    total_area += area;
                } else {
                    fallback_centroid += centroid;
                    fallback_count += 1;
                }
            }
        }

        if total_area > 0.0 {
            Some(weighted_centroid / total_area)
        } else if fallback_count > 0 {
            Some(fallback_centroid / fallback_count as f32)
        } else {
            None
        }
    }

    fn triangle_vertices(&self, triangle_index: usize) -> Option<(Vec3, Vec3, Vec3)> {
        let vertex_at = |index: u32| {
            self.render_data
                .vertices
                .get(index as usize)
                .map(|vertex| Vec3::from_array(vertex.position))
        };

        if let Some(indices) = self.indices.as_ref() {
            let base = triangle_index.checked_mul(3)?;
            let triangle = indices.get(base..base + 3)?;
            Some((
                vertex_at(triangle[0])?,
                vertex_at(triangle[1])?,
                vertex_at(triangle[2])?,
            ))
        } else {
            let base = triangle_index.checked_mul(3)?;
            let a = self.render_data.vertices.get(base)?;
            let b = self.render_data.vertices.get(base + 1)?;
            let c = self.render_data.vertices.get(base + 2)?;
            Some((
                Vec3::from_array(a.position),
                Vec3::from_array(b.position),
                Vec3::from_array(c.position),
            ))
        }
    }

    fn apply_region_color(
        &self,
        render_data: &mut RenderData,
        region_id: &str,
        color: [f32; 4],
    ) -> usize {
        if self.render_data.pipeline_type != PipelineType::Triangles {
            return 0;
        }
        let Some(region) = self.regions.iter().find(|item| item.region_id == region_id) else {
            log::info!(
                target: "runmat_plot",
                "geometry_scene.selection_color_missing chunk_id={} region_id={} available_regions={}",
                self.chunk_id,
                region_id,
                self.regions.len()
            );
            return 0;
        };
        let (Some(indices), Some(render_indices)) =
            (self.indices.as_ref(), render_data.indices.as_mut())
        else {
            return self.apply_direct_region_color(render_data, region, color);
        };

        let mut colored = 0usize;
        for range in &region.triangle_ranges {
            let start = range.start as usize;
            let end = start.saturating_add(range.count as usize);
            for triangle_index in start..end {
                let base = triangle_index.saturating_mul(3);
                let Some(triangle) = indices.get(base..base + 3) else {
                    continue;
                };
                if render_indices.get(base..base + 3).is_none() {
                    continue;
                }
                let mut isolated_indices = [0_u32; 3];
                let mut isolated = true;
                for (slot, vertex_index) in triangle.iter().copied().enumerate() {
                    let Some(vertex) = render_data.vertices.get(vertex_index as usize).copied()
                    else {
                        isolated = false;
                        break;
                    };
                    let mut vertex = vertex;
                    vertex.color = color;
                    let next_index = render_data.vertices.len();
                    if next_index > u32::MAX as usize {
                        isolated = false;
                        break;
                    }
                    render_data.vertices.push(vertex);
                    isolated_indices[slot] = next_index as u32;
                }
                if isolated {
                    render_indices[base..base + 3].copy_from_slice(&isolated_indices);
                    colored += 1;
                }
            }
        }
        colored
    }

    fn apply_direct_region_color(
        &self,
        render_data: &mut RenderData,
        region: &GeometrySceneRegion,
        color: [f32; 4],
    ) -> usize {
        if render_data.indices.is_some() {
            return 0;
        }
        let mut colored = 0usize;
        for range in &region.triangle_ranges {
            let start = range.start as usize;
            let end = start.saturating_add(range.count as usize);
            for triangle_index in start..end {
                let base = triangle_index.saturating_mul(3);
                let Some(triangle) = render_data.vertices.get_mut(base..base + 3) else {
                    continue;
                };
                for vertex in triangle {
                    vertex.color = color;
                }
                colored += 1;
            }
        }
        colored
    }
}

fn section_plane(plane: &GeometrySceneSectionPlane) -> Option<(Vec3, Vec3)> {
    let normal = Vec3::from_array(plane.normal).normalize_or_zero();
    if normal.length_squared() <= f32::EPSILON {
        return None;
    }
    let origin = plane
        .origin
        .map(Vec3::from_array)
        .unwrap_or_else(|| normal * plane.offset.unwrap_or(0.0));
    if !origin.is_finite() {
        return None;
    }
    Some((normal, origin))
}

fn geometry_region_role_color(role: Option<&str>) -> [f32; 4] {
    match role.unwrap_or_default() {
        "material" => GEOMETRY_MATERIAL_REGION_COLOR,
        "boundary" => GEOMETRY_BOUNDARY_REGION_COLOR,
        "driving" => GEOMETRY_DRIVING_REGION_COLOR,
        "selection" => GEOMETRY_SELECTED_REGION_COLOR,
        _ => GEOMETRY_HOVER_REGION_COLOR,
    }
}

fn clip_triangle_render_data(mut render_data: RenderData, plane: (Vec3, Vec3)) -> RenderData {
    let mut clipped_vertices = Vec::new();
    if let Some(indices) = render_data.indices.as_ref() {
        for triangle in indices.chunks_exact(3) {
            let Some(vertices) = triangle_vertices_from_indices(&render_data.vertices, triangle)
            else {
                continue;
            };
            push_clipped_triangle(vertices, plane, &mut clipped_vertices);
        }
    } else {
        for triangle in render_data.vertices.chunks_exact(3) {
            push_clipped_triangle(
                [triangle[0], triangle[1], triangle[2]],
                plane,
                &mut clipped_vertices,
            );
        }
    }
    let bounds = bounds_from_vertices(&clipped_vertices);
    let vertex_count = clipped_vertices.len();
    render_data.vertices = clipped_vertices;
    render_data.indices = None;
    render_data.bounds = Some(bounds);
    render_data.draw_calls = vec![DrawCall {
        vertex_offset: 0,
        vertex_count,
        index_offset: None,
        index_count: None,
        instance_count: 1,
    }];
    render_data
}

fn clip_line_render_data(mut render_data: RenderData, plane: (Vec3, Vec3)) -> RenderData {
    let mut clipped_vertices = Vec::new();
    if let Some(indices) = render_data.indices.as_ref() {
        for line in indices.chunks_exact(2) {
            let Some(a) = render_data.vertices.get(line[0] as usize).copied() else {
                continue;
            };
            let Some(b) = render_data.vertices.get(line[1] as usize).copied() else {
                continue;
            };
            push_clipped_line(a, b, plane, &mut clipped_vertices);
        }
    } else {
        for line in render_data.vertices.chunks_exact(2) {
            push_clipped_line(line[0], line[1], plane, &mut clipped_vertices);
        }
    }
    let bounds = bounds_from_vertices(&clipped_vertices);
    let vertex_count = clipped_vertices.len();
    render_data.vertices = clipped_vertices;
    render_data.indices = None;
    render_data.bounds = Some(bounds);
    render_data.draw_calls = vec![DrawCall {
        vertex_offset: 0,
        vertex_count,
        index_offset: None,
        index_count: None,
        instance_count: 1,
    }];
    render_data
}

fn triangle_vertices_from_indices(vertices: &[Vertex], triangle: &[u32]) -> Option<[Vertex; 3]> {
    Some([
        *vertices.get(*triangle.first()? as usize)?,
        *vertices.get(*triangle.get(1)? as usize)?,
        *vertices.get(*triangle.get(2)? as usize)?,
    ])
}

fn push_clipped_triangle(vertices: [Vertex; 3], plane: (Vec3, Vec3), output: &mut Vec<Vertex>) {
    let clipped = clip_polygon_to_plane(&vertices, plane);
    if clipped.len() < 3 {
        return;
    }
    for index in 1..clipped.len().saturating_sub(1) {
        output.push(clipped[0]);
        output.push(clipped[index]);
        output.push(clipped[index + 1]);
    }
}

fn push_clipped_line(a: Vertex, b: Vertex, plane: (Vec3, Vec3), output: &mut Vec<Vertex>) {
    let distance_a = signed_distance_to_plane(a, plane);
    let distance_b = signed_distance_to_plane(b, plane);
    let inside_a = distance_a >= -SECTION_DISTANCE_EPSILON;
    let inside_b = distance_b >= -SECTION_DISTANCE_EPSILON;
    match (inside_a, inside_b) {
        (true, true) => {
            output.push(a);
            output.push(b);
        }
        (false, false) => {}
        (true, false) => {
            output.push(a);
            output.push(interpolate_vertex(
                a,
                b,
                section_intersection_t(distance_a, distance_b),
            ));
        }
        (false, true) => {
            output.push(interpolate_vertex(
                a,
                b,
                section_intersection_t(distance_a, distance_b),
            ));
            output.push(b);
        }
    }
}

fn clip_polygon_to_plane(vertices: &[Vertex], plane: (Vec3, Vec3)) -> Vec<Vertex> {
    let Some(mut previous) = vertices.last().copied() else {
        return Vec::new();
    };
    let mut previous_distance = signed_distance_to_plane(previous, plane);
    let mut previous_inside = previous_distance >= -SECTION_DISTANCE_EPSILON;
    let mut output = Vec::with_capacity(vertices.len() + 1);
    for current in vertices.iter().copied() {
        let current_distance = signed_distance_to_plane(current, plane);
        let current_inside = current_distance >= -SECTION_DISTANCE_EPSILON;
        if current_inside != previous_inside {
            output.push(interpolate_vertex(
                previous,
                current,
                section_intersection_t(previous_distance, current_distance),
            ));
        }
        if current_inside {
            output.push(current);
        }
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    output
}

fn signed_distance_to_plane(vertex: Vertex, (normal, origin): (Vec3, Vec3)) -> f32 {
    (Vec3::from_array(vertex.position) - origin).dot(normal)
}

fn section_intersection_t(distance_a: f32, distance_b: f32) -> f32 {
    let denominator = distance_a - distance_b;
    if denominator.abs() <= f32::EPSILON {
        0.0
    } else {
        (distance_a / denominator).clamp(0.0, 1.0)
    }
}

fn interpolate_vertex(a: Vertex, b: Vertex, t: f32) -> Vertex {
    let position = Vec3::from_array(a.position).lerp(Vec3::from_array(b.position), t);
    let color = Vec4::from_array(a.color).lerp(Vec4::from_array(b.color), t);
    let normal = Vec3::from_array(a.normal)
        .lerp(Vec3::from_array(b.normal), t)
        .normalize_or_zero();
    let tex_coords_a = Vec2::from_array(a.tex_coords);
    let tex_coords_b = Vec2::from_array(b.tex_coords);
    Vertex {
        position: position.to_array(),
        color: color.to_array(),
        normal: if normal.length_squared() > f32::EPSILON {
            normal.to_array()
        } else {
            a.normal
        },
        tex_coords: tex_coords_a.lerp(tex_coords_b, t).to_array(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometrySceneRegion {
    pub region_id: String,
    pub label: Option<String>,
    pub tag: Option<String>,
    pub triangle_ranges: Vec<GeometrySceneTriangleRange>,
}

impl GeometrySceneRegion {
    pub fn new(
        region_id: impl Into<String>,
        label: Option<String>,
        tag: Option<String>,
        triangle_ranges: Vec<GeometrySceneTriangleRange>,
    ) -> Self {
        Self {
            region_id: region_id.into(),
            label,
            tag,
            triangle_ranges,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometrySceneTriangleRange {
    pub start: u32,
    pub count: u32,
}

impl GeometrySceneTriangleRange {
    pub fn new(start: u32, count: u32) -> Self {
        Self { start, count }
    }
}

pub fn cad_default_material() -> Material {
    Material {
        albedo: Vec4::new(0.46, 0.49, 0.48, 1.0),
        roughness: 0.72,
        metallic: 0.0,
        emissive: Vec4::ZERO,
        alpha_mode: AlphaMode::Opaque,
        double_sided: true,
    }
}

pub fn vertex(position: [f32; 3], color: [f32; 4], normal: [f32; 3]) -> Vertex {
    Vertex {
        position,
        color,
        normal,
        tex_coords: [0.0, 0.0],
    }
}

fn bounds_from_vertices(vertices: &[Vertex]) -> BoundingBox {
    if vertices.is_empty() {
        return BoundingBox::default();
    }
    let mut bounds = BoundingBox::new(
        Vec3::from_array(vertices[0].position),
        Vec3::from_array(vertices[0].position),
    );
    for item in vertices.iter().skip(1) {
        bounds.expand(Vec3::from_array(item.position));
    }
    bounds
}

fn combined_chunk_bounds(chunks: &[GeometrySceneChunk]) -> BoundingBox {
    let mut bounds = BoundingBox::default();
    for chunk in chunks {
        bounds.expand_by_box(&chunk.bounds);
    }
    bounds
}

fn annotation_arrow_length(bounds: BoundingBox) -> f32 {
    let size = bounds.size();
    let diagonal = size.length();
    if diagonal.is_finite() && diagonal > 1.0e-6 {
        diagonal * 0.075
    } else {
        1.0
    }
}

fn normalized_annotation_direction(direction: [f32; 3]) -> Option<Vec3> {
    let direction = Vec3::from_array(direction);
    let length = direction.length();
    (length.is_finite() && length > 1.0e-8).then_some(direction / length)
}

fn append_annotation_arrow(
    vertices: &mut Vec<Vertex>,
    anchor: Vec3,
    direction: Vec3,
    length: f32,
    color: [f32; 4],
) {
    let start = anchor;
    let end = anchor + direction * length;
    append_annotation_line(vertices, start, end, color);

    let side = perpendicular_unit(direction);
    let wing_base = end - direction * (length * 0.28);
    let wing_size = length * 0.12;
    append_annotation_line(vertices, end, wing_base + side * wing_size, color);
    append_annotation_line(vertices, end, wing_base - side * wing_size, color);
}

fn append_annotation_line(vertices: &mut Vec<Vertex>, start: Vec3, end: Vec3, color: [f32; 4]) {
    vertices.push(vertex(start.to_array(), color, [0.0, 0.0, 1.0]));
    vertices.push(vertex(end.to_array(), color, [0.0, 0.0, 1.0]));
}

fn perpendicular_unit(direction: Vec3) -> Vec3 {
    let reference = if direction.z.abs() < 0.9 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let side = direction.cross(reference);
    let length = side.length();
    if length > 1.0e-8 {
        side / length
    } else {
        Vec3::X
    }
}

fn stable_node_id(scene_id: &str, revision: u64, index: usize, chunk_id: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET_BASIS;
    for byte in scene_id
        .as_bytes()
        .iter()
        .chain(chunk_id.as_bytes())
        .copied()
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash ^= revision;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^ index as u64
}

#[derive(Debug, Clone)]
struct IndexedTriangle {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    bounds: BoundingBox,
    centroid: Vec3,
    mesh_id: Option<String>,
    chunk_id: String,
    triangle_index: usize,
    region_id: Option<String>,
    region_label: Option<String>,
    region_tag: Option<String>,
}

#[derive(Debug, Clone)]
struct PickBvhNode {
    bounds: BoundingBox,
    kind: PickBvhNodeKind,
}

#[derive(Debug, Clone, Copy)]
enum PickBvhNodeKind {
    Leaf { start: usize, end: usize },
    Branch { left: usize, right: usize },
}

#[derive(Debug, Clone, Copy)]
struct PickRay {
    origin: Vec3,
    direction: Vec3,
}

#[derive(Debug, Clone, Copy)]
struct PickHit {
    triangle_index: usize,
    distance: f32,
}

fn build_pick_bvh_node(
    nodes: &mut Vec<PickBvhNode>,
    triangles: &mut [IndexedTriangle],
    start: usize,
    end: usize,
) -> Option<usize> {
    if start >= end {
        return None;
    }
    let bounds = combined_triangle_bounds(&triangles[start..end]);
    let node_index = nodes.len();
    nodes.push(PickBvhNode {
        bounds,
        kind: PickBvhNodeKind::Leaf { start, end },
    });
    const LEAF_TRIANGLES: usize = 32;
    if end - start <= LEAF_TRIANGLES {
        return Some(node_index);
    }
    let centroid_bounds = combined_centroid_bounds(&triangles[start..end]);
    let extent = centroid_bounds.max - centroid_bounds.min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    triangles[start..end].sort_by(|a, b| {
        a.centroid[axis]
            .partial_cmp(&b.centroid[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + (end - start) / 2;
    let left = build_pick_bvh_node(nodes, triangles, start, mid);
    let right = build_pick_bvh_node(nodes, triangles, mid, end);
    if let (Some(left), Some(right)) = (left, right) {
        nodes[node_index].kind = PickBvhNodeKind::Branch { left, right };
    }
    Some(node_index)
}

fn triangle_bounds(a: Vec3, b: Vec3, c: Vec3) -> BoundingBox {
    let mut bounds = BoundingBox::new(a, a);
    bounds.expand(b);
    bounds.expand(c);
    bounds
}

fn combined_triangle_bounds(triangles: &[IndexedTriangle]) -> BoundingBox {
    let mut bounds = BoundingBox::default();
    for triangle in triangles {
        bounds.expand_by_box(&triangle.bounds);
    }
    bounds
}

fn combined_centroid_bounds(triangles: &[IndexedTriangle]) -> BoundingBox {
    let Some(first) = triangles.first() else {
        return BoundingBox::default();
    };
    let mut bounds = BoundingBox::new(first.centroid, first.centroid);
    for triangle in triangles.iter().skip(1) {
        bounds.expand(triangle.centroid);
    }
    bounds
}

fn ray_intersects_bounds(ray: &PickRay, bounds: BoundingBox) -> Option<f32> {
    let mut t_min: f32 = 0.0;
    let mut t_max = f32::INFINITY;
    for axis in 0..3 {
        let origin = ray.origin[axis];
        let direction = ray.direction[axis];
        let min = bounds.min[axis];
        let max = bounds.max[axis];
        if direction.abs() < 1e-8 {
            if origin < min || origin > max {
                return None;
            }
            continue;
        }
        let inv_direction = 1.0 / direction;
        let mut t0 = (min - origin) * inv_direction;
        let mut t1 = (max - origin) * inv_direction;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_min = t_min.max(t0);
        t_max = t_max.min(t1);
        if t_max < t_min {
            return None;
        }
    }
    Some(t_min.max(0.0))
}

fn ray_intersects_triangle(ray: &PickRay, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let pvec = ray.direction.cross(edge2);
    let det = edge1.dot(pvec);
    if det.abs() < 1e-7 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ray.origin - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(edge1);
    let v = ray.direction.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(qvec) * inv_det;
    (t > 1e-6).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProjectionType;

    #[test]
    fn pick_index_returns_region_for_triangle() {
        let material = cad_default_material();
        let chunk = GeometrySceneChunk::indexed_triangles(
            "face_chunk",
            vec![
                vertex([-1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2],
            material,
        )
        .with_regions(vec![GeometrySceneRegion::new(
            "face_a",
            Some("Face A".to_string()),
            Some("cad-face".to_string()),
            vec![GeometrySceneTriangleRange::new(0, 1)],
        )]);
        let scene = GeometryScene::new("scene", 1, vec![chunk]);
        let index = GeometryScenePickIndex::build(&scene);
        let mut camera = Camera::new();
        camera.position = Vec3::new(0.0, 0.0, 5.0);
        camera.target = Vec3::ZERO;
        camera.up = Vec3::Y;
        camera.aspect_ratio = 1.0;
        camera.projection = ProjectionType::Perspective {
            fov: 45.0_f32.to_radians(),
            near: 0.1,
            far: 100.0,
        };
        camera.mark_dirty();

        let hit = index.pick(GeometryScenePickRequest {
            camera,
            surface_size: [800.0, 800.0],
            position: [400.0, 400.0],
        });
        assert_eq!(
            hit.and_then(|hit| hit.region_id),
            Some("face_a".to_string())
        );
    }

    #[test]
    fn presentation_selection_colors_indexed_triangle_region() {
        let material = cad_default_material();
        let chunk = GeometrySceneChunk::indexed_triangles(
            "face_chunk",
            vec![
                vertex([-1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([2.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2, 1, 3, 2],
            material,
        )
        .with_regions(vec![GeometrySceneRegion::new(
            "face_b",
            Some("Face B".to_string()),
            Some("cad-face".to_string()),
            vec![GeometrySceneTriangleRange::new(1, 1)],
        )]);

        let render_data = chunk.render_data_with_presentation(&GeometryScenePresentation {
            selected_region_ids: vec!["face_b".to_string()],
            ..Default::default()
        });

        assert_eq!(render_data.vertices.len(), 7);
        assert_eq!(
            render_data.indices.as_deref(),
            Some(&[0, 1, 2, 4, 5, 6][..])
        );
        assert_eq!(
            render_data.vertices[4].color,
            GEOMETRY_SELECTED_REGION_COLOR
        );
        assert_eq!(
            render_data.vertices[5].color,
            GEOMETRY_SELECTED_REGION_COLOR
        );
        assert_eq!(
            render_data.vertices[6].color,
            GEOMETRY_SELECTED_REGION_COLOR
        );
    }

    #[test]
    fn presentation_selection_colors_direct_triangle_region() {
        let material = cad_default_material();
        let base_color = [0.5, 0.5, 0.5, 1.0];
        let vertices = vec![
            vertex([-1.0, -1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
            vertex([1.0, -1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
            vertex([1.0, -1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
            vertex([2.0, -1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
        ];
        let chunk = GeometrySceneChunk::from_render_data(
            "face_chunk",
            RenderData {
                pipeline_type: PipelineType::Triangles,
                vertices,
                indices: None,
                gpu_vertices: None,
                bounds: None,
                material,
                draw_calls: vec![DrawCall {
                    vertex_offset: 0,
                    vertex_count: 6,
                    index_offset: None,
                    index_count: None,
                    instance_count: 1,
                }],
                image: None,
            },
        )
        .with_regions(vec![GeometrySceneRegion::new(
            "face_b",
            Some("Face B".to_string()),
            Some("cad-face".to_string()),
            vec![GeometrySceneTriangleRange::new(1, 1)],
        )]);

        let render_data = chunk.render_data_with_presentation(&GeometryScenePresentation {
            selected_region_id: Some("face_b".to_string()),
            ..Default::default()
        });

        assert!(render_data.indices.is_none());
        assert_eq!(render_data.vertices.len(), 6);
        assert_eq!(render_data.vertices[0].color, base_color);
        assert_eq!(render_data.vertices[1].color, base_color);
        assert_eq!(render_data.vertices[2].color, base_color);
        assert_eq!(
            render_data.vertices[3].color,
            GEOMETRY_SELECTED_REGION_COLOR
        );
        assert_eq!(
            render_data.vertices[4].color,
            GEOMETRY_SELECTED_REGION_COLOR
        );
        assert_eq!(
            render_data.vertices[5].color,
            GEOMETRY_SELECTED_REGION_COLOR
        );
    }

    #[test]
    fn presentation_resolves_hidden_and_isolated_owner_visibility() {
        let current_hidden = BTreeSet::from(["panel_b".to_string()]);
        let all_owner_ids = ["panel_a", "panel_b", "panel_c"];

        assert_eq!(
            GeometryScenePresentation::default()
                .resolved_hidden_owner_node_ids(all_owner_ids, &current_hidden),
            current_hidden
        );

        assert_eq!(
            GeometryScenePresentation {
                hidden_owner_node_ids: Some(vec![
                    " panel_a ".to_string(),
                    String::new(),
                    "panel_a".to_string(),
                ]),
                ..Default::default()
            }
            .resolved_hidden_owner_node_ids(all_owner_ids, &BTreeSet::new()),
            BTreeSet::from(["panel_a".to_string()])
        );

        assert_eq!(
            GeometryScenePresentation {
                hidden_owner_node_ids: Some(vec!["panel_c".to_string()]),
                isolated_owner_node_ids: Some(vec!["panel_a".to_string()]),
                ..Default::default()
            }
            .resolved_hidden_owner_node_ids(all_owner_ids, &BTreeSet::new()),
            BTreeSet::from(["panel_b".to_string(), "panel_c".to_string()])
        );

        assert_eq!(
            GeometryScenePresentation {
                hidden_owner_node_ids: Some(Vec::new()),
                isolated_owner_node_ids: None,
                ..Default::default()
            }
            .resolved_hidden_owner_node_ids(all_owner_ids, &current_hidden),
            BTreeSet::new()
        );
    }

    #[test]
    fn presentation_section_distinguishes_preserve_clear_and_plane() {
        let preserve: GeometryScenePresentation =
            serde_json::from_value(serde_json::json!({})).expect("preserve presentation");
        assert_eq!(preserve.section, None);

        let clear: GeometryScenePresentation =
            serde_json::from_value(serde_json::json!({ "section": null }))
                .expect("clear presentation");
        assert_eq!(clear.section, Some(None));

        let sectioned: GeometryScenePresentation = serde_json::from_value(serde_json::json!({
            "section": {
                "plane": {
                    "normal": [1.0, 0.0, 0.0],
                    "origin": [0.0, 0.0, 0.0],
                    "label": "midplane"
                }
            }
        }))
        .expect("section presentation");
        let section = sectioned
            .section
            .as_ref()
            .and_then(Option::as_ref)
            .expect("section");
        assert_eq!(section.plane.normal, [1.0, 0.0, 0.0]);
        assert_eq!(section.plane.label.as_deref(), Some("midplane"));
    }

    #[test]
    fn presentation_accepts_bounded_view_presets() {
        let front: GeometryScenePresentation =
            serde_json::from_value(serde_json::json!({ "viewPreset": "front" }))
                .expect("front view presentation");
        assert_eq!(front.view_preset, Some(GeometrySceneViewPreset::Front));

        let isometric: GeometryScenePresentation =
            serde_json::from_value(serde_json::json!({ "viewPreset": "isometric" }))
                .expect("isometric view presentation");
        assert_eq!(
            isometric.view_preset,
            Some(GeometrySceneViewPreset::Isometric)
        );
    }

    #[test]
    fn presentation_section_clips_triangle_render_data() {
        let material = cad_default_material();
        let chunk = GeometrySceneChunk::indexed_triangles(
            "face_chunk",
            vec![
                vertex([-1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2],
            material,
        );
        let render_data = chunk.render_data_with_presentation(&GeometryScenePresentation {
            section: Some(Some(GeometrySceneSection {
                plane: GeometrySceneSectionPlane {
                    normal: [1.0, 0.0, 0.0],
                    origin: Some([0.0, 0.0, 0.0]),
                    offset: None,
                    label: Some("midplane".to_string()),
                },
            })),
            ..Default::default()
        });

        assert_eq!(render_data.pipeline_type, PipelineType::Triangles);
        assert!(render_data.indices.is_none());
        assert_eq!(render_data.vertices.len(), 6);
        assert_eq!(
            render_data.draw_calls[0].vertex_count,
            render_data.vertices.len()
        );
        assert!(render_data
            .vertices
            .iter()
            .all(|vertex| vertex.position[0] >= -SECTION_DISTANCE_EPSILON));
    }

    #[test]
    fn presentation_region_annotations_emit_marker_and_vector_nodes() {
        let material = cad_default_material();
        let chunk = GeometrySceneChunk::indexed_triangles(
            "face_chunk",
            vec![
                vertex([-1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2],
            material,
        )
        .with_regions(vec![GeometrySceneRegion::new(
            "loaded_face",
            Some("Loaded face".to_string()),
            Some("cad-face".to_string()),
            vec![GeometrySceneTriangleRange::new(0, 1)],
        )]);
        let scene = GeometryScene::new("scene", 1, vec![chunk]);
        let nodes = scene.nodes_with_presentation(&GeometryScenePresentation {
            region_annotations: vec![GeometrySceneRegionAnnotation {
                region_id: "loaded_face".to_string(),
                color: Some([0.9, 0.1, 0.1, 1.0]),
                role: Some("load".to_string()),
                label: Some("load".to_string()),
                direction: Some([0.0, 0.0, 1.0]),
                size: Some(18.0),
            }],
            ..Default::default()
        });

        let marker = nodes
            .iter()
            .find(|node| node.name == "FEA region markers")
            .and_then(|node| node.render_data.as_ref())
            .expect("marker annotation node");
        assert_eq!(marker.pipeline_type, PipelineType::Points);
        assert_eq!(marker.vertices.len(), 1);
        assert!((marker.vertices[0].position[1] - (-1.0 / 3.0)).abs() < 1.0e-6);
        assert_eq!(marker.vertices[0].normal[2], 18.0);

        let vector = nodes
            .iter()
            .find(|node| node.name == "FEA load vectors")
            .and_then(|node| node.render_data.as_ref())
            .expect("vector annotation node");
        assert_eq!(vector.pipeline_type, PipelineType::Lines);
        assert_eq!(vector.vertices.len(), 6);
        assert!(vector.vertices[1].position[2] > vector.vertices[0].position[2]);
    }
}
