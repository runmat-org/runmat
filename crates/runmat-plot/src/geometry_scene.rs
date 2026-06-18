//! Chunked geometry scenes for CAD and FEA visualization.
//!
//! This is intentionally rendering-domain data. CAD import, semantic ownership,
//! and FEA result storage stay in their own crates; this module describes the
//! mesh chunks that the plot renderer can keep resident and redraw efficiently.

use crate::core::{
    AlphaMode, BoundingBox, Camera, DrawCall, Material, PipelineType, RenderData, SceneNode, Vertex,
};
use glam::{Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

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
                let color = annotation.color;
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
    pub hovered_region_id: Option<String>,
    #[serde(default)]
    pub region_highlights: Vec<GeometrySceneRegionHighlight>,
    #[serde(default)]
    pub region_annotations: Vec<GeometrySceneRegionAnnotation>,
    #[serde(default)]
    pub display_mode: GeometrySceneDisplayMode,
    #[serde(default = "default_edge_overlay_enabled")]
    pub edge_overlay_enabled: bool,
}

impl Default for GeometryScenePresentation {
    fn default() -> Self {
        Self {
            selected_region_id: None,
            hovered_region_id: None,
            region_highlights: Vec::new(),
            region_annotations: Vec::new(),
            display_mode: GeometrySceneDisplayMode::Shaded,
            edge_overlay_enabled: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySceneRegionHighlight {
    pub region_id: String,
    pub color: [f32; 4],
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySceneRegionAnnotation {
    pub region_id: String,
    pub color: [f32; 4],
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
            self.apply_region_color(&mut render_data, &highlight.region_id, highlight.color);
        }
        if let Some(region_id) = presentation.hovered_region_id.as_deref() {
            self.apply_region_color(&mut render_data, region_id, [0.43, 0.78, 1.0, 1.0]);
        }
        if let Some(region_id) = presentation.selected_region_id.as_deref() {
            self.apply_region_color(&mut render_data, region_id, [0.98, 0.78, 0.22, 1.0]);
        }
        render_data
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

    fn apply_region_color(&self, render_data: &mut RenderData, region_id: &str, color: [f32; 4]) {
        if self.render_data.pipeline_type != PipelineType::Triangles {
            return;
        }
        let Some(region) = self.regions.iter().find(|item| item.region_id == region_id) else {
            return;
        };
        let Some(indices) = self.indices.as_ref() else {
            return;
        };
        let Some(render_indices) = render_data.indices.as_mut() else {
            return;
        };
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
                }
            }
        }
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
                color: [0.9, 0.1, 0.1, 1.0],
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
