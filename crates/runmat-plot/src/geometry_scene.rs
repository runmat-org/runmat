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
use std::collections::{BTreeSet, HashMap, HashSet};

mod annotations;
mod edge_stroke;

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
    /// Unit provenance is attached to the rendered representation, not supplied
    /// by a later measurement caller. Legacy/unitless scenes remain explicit.
    pub physical_context: Option<GeometryPhysicalContext>,
    /// Export-only labels on a request-local agent observation.
    pub observation_callouts: Vec<GeometrySceneCallout>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryPhysicalContext {
    pub coordinate_units:
        runmat_types::physical::CoordinateUnits<runmat_types::physical::PreviewCoordinates>,
    pub source_unit_evidence: runmat_types::physical::LengthUnitEvidence,
}

/// Source face triangles provide candidate leader endpoints. Visibility is
/// checked against the final clipped scene before placing a label.
#[derive(Debug, Clone)]
pub struct GeometrySceneCallout {
    pub region_id: String,
    pub color: [f32; 4],
    pub triangles: Vec<[[f32; 3]; 3]>,
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
            show_grid: true,
            axis_equal: true,
            observation_callouts: Vec::new(),
            physical_context: None,
        }
    }

    pub fn with_physical_context(mut self, context: GeometryPhysicalContext) -> Self {
        self.physical_context = Some(context);
        self
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
                render_data: Some(edge_stroke::widen(
                    chunk.render_data_with_presentation(presentation),
                    presentation.edge_width_px,
                )),
                bounds: chunk.bounds,
                lod_levels: Vec::new(),
                current_lod: 0,
            })
            .collect();
        nodes.extend(self.annotation_nodes(presentation));
        nodes.extend(self.topology_selection_nodes(presentation));
        nodes
    }

    pub fn chunk_node_id(&self, index: usize, chunk_id: &str) -> u64 {
        stable_node_id(&self.scene_id, self.revision, index, chunk_id)
    }

    #[cfg(feature = "egui-overlay")]
    pub(crate) fn region_anchor(&self, region_id: &str) -> Option<Vec3> {
        if let Some(mesh_id) = region_id.strip_prefix("mesh:") {
            let mut bounds = None::<BoundingBox>;
            for chunk in self
                .chunks
                .iter()
                .filter(|chunk| chunk.visible && chunk.mesh_id.as_deref() == Some(mesh_id))
            {
                bounds = Some(match bounds {
                    Some(bounds) => bounds.union(&chunk.bounds),
                    None => chunk.bounds,
                });
            }
            return bounds.map(|bounds| (bounds.min + bounds.max) * 0.5);
        }
        self.chunks
            .iter()
            .filter(|chunk| chunk.visible)
            .find_map(|chunk| chunk.region_anchor(region_id))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryScenePresentation {
    pub selected_region_id: Option<String>,
    #[serde(default)]
    pub selected_region_ids: Vec<String>,
    #[serde(default)]
    pub selected_topology: Vec<GeometrySceneTopologySelection>,
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
    pub show_grid: Option<bool>,
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
    /// Width of line chunks (edges, wireframes) in CSS px. Above 1 px they draw as screen-space
    /// strokes of that width; unset or 1 px or less draws GPU hairlines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_width_px: Option<f32>,
}

impl Default for GeometryScenePresentation {
    fn default() -> Self {
        Self {
            selected_region_id: None,
            selected_region_ids: Vec::new(),
            selected_topology: Vec::new(),
            hovered_region_id: None,
            region_highlights: Vec::new(),
            region_annotations: Vec::new(),
            display_mode: GeometrySceneDisplayMode::Shaded,
            edge_overlay_enabled: true,
            show_grid: None,
            hidden_owner_node_ids: None,
            isolated_owner_node_ids: None,
            section: None,
            view_preset: None,
            edge_width_px: None,
        }
    }
}

impl GeometryScenePresentation {
    pub(crate) fn rewrites_geometry_vertices(&self) -> bool {
        self.hovered_region_id.is_some()
            || self.selected_region_id.is_some()
            || !self.selected_region_ids.is_empty()
            || self
                .selected_topology
                .iter()
                .any(|selection| matches!(selection, GeometrySceneTopologySelection::Face { .. }))
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum GeometrySceneTopologySelection {
    Face {
        chunk_id: String,
        triangle_index: usize,
    },
    Edge {
        chunk_id: String,
        vertex_indices: [u32; 2],
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        segments: Vec<[u32; 2]>,
    },
    Point {
        chunk_id: String,
        vertex_index: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySceneRegionAnnotation {
    pub region_id: String,
    #[serde(default)]
    pub color: Option<[f32; 4]>,
    pub kind: GeometrySceneAnnotationKind,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub direction: Option<[f32; 3]>,
    #[serde(default)]
    pub size: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometrySceneAnnotationKind {
    /// Text-only selection identity; does not add a physical marker.
    Label,
    Force,
    Moment,
    Pressure,
    Traction,
    BodyForce,
    HeatFlux,
    CurrentDensity,
    NormalVelocity,
    FlowVelocity,
    MassFlow,
    WallMotion,
    FixedSupport,
    SymmetrySupport,
    PrescribedMotion,
    BoundaryCondition,
    ThermalCondition,
    ElectricalCondition,
    MagneticCondition,
    AcousticCondition,
    FluidCondition,
    Connector,
    Interface,
    InitialCondition,
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
    fn alpha(self, _edge_overlay_enabled: bool) -> f32 {
        match self {
            Self::Shaded => 1.0,
            Self::Edges => 0.38,
            Self::Wireframe => 0.16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeometryScenePickRequest {
    pub camera: Camera,
    pub surface_size: [f32; 2],
    pub position: [f32; 2],
    pub mode: GeometryScenePickMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryScenePickMode {
    #[default]
    Face,
    Edge,
    Point,
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
    /// Vertex indices of the hit tessellation triangle. These preserve stable mesh topology for
    /// point/node and edge selection without pretending that every click is a face selection.
    pub vertex_indices: [u32; 3],
    /// Barycentric coordinates of `position` within `vertex_indices`.
    pub barycentric_coordinates: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_selection: Option<GeometrySceneTopologySelection>,
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
    feature_edges: HashMap<PickEdgeKey, FeatureEdgeInfo>,
    feature_edge_neighbors: HashMap<PickVertexKey, Vec<PickEdgeKey>>,
    scene_diagonal: f32,
}

impl GeometryScenePickIndex {
    pub fn build(scene: &GeometryScene) -> Self {
        let mut triangles = Vec::with_capacity(scene.triangle_count());
        for (chunk_index, chunk) in scene.chunks.iter().enumerate() {
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
                    chunk_index,
                    chunk_id: chunk.chunk_id.clone(),
                    triangle_index,
                    vertex_indices: [triangle[0], triangle[1], triangle[2]],
                    region_id: region.map(|item| item.region_id.clone()),
                    region_label: region.and_then(|item| item.label.clone()),
                    region_tag: region.and_then(|item| item.tag.clone()),
                });
            }
        }

        let mut nodes = Vec::new();
        let triangle_count = triangles.len();
        let root = build_pick_bvh_node(&mut nodes, &mut triangles, 0, triangle_count);
        let (feature_edges, feature_edge_neighbors) = collect_feature_edges(&triangles);
        Self {
            scene_key: scene.cache_key(),
            triangles,
            nodes,
            root,
            feature_edges,
            feature_edge_neighbors,
            scene_diagonal: scene.bounds.size().length().max(1.0e-6),
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
        let screen_pos = Vec2::from_array(request.position);
        let screen_size = Vec2::from_array(request.surface_size);
        match request.mode {
            GeometryScenePickMode::Face => {
                let (hit, ray) = self.pick_at(&mut camera, screen_size, screen_pos)?;
                let position = ray.origin + ray.direction * hit.distance;
                self.result_for_triangle(hit.triangle_index, position, hit.distance, None)
            }
            GeometryScenePickMode::Edge | GeometryScenePickMode::Point => {
                self.pick_screen_topology(&mut camera, screen_size, screen_pos, request.mode)
            }
        }
    }

    fn result_for_triangle(
        &self,
        triangle_index: usize,
        position: Vec3,
        distance: f32,
        topology_selection: Option<GeometrySceneTopologySelection>,
    ) -> Option<GeometryScenePickResult> {
        let triangle = self.triangles.get(triangle_index)?;
        Some(GeometryScenePickResult {
            mesh_id: triangle.mesh_id.clone(),
            chunk_id: triangle.chunk_id.clone(),
            triangle_index: triangle.triangle_index,
            region_id: triangle.region_id.clone(),
            region_label: triangle.region_label.clone(),
            region_tag: triangle.region_tag.clone(),
            distance,
            position: position.to_array(),
            vertex_indices: triangle.vertex_indices,
            barycentric_coordinates: triangle_barycentric_coordinates(
                position, triangle.a, triangle.b, triangle.c,
            ),
            topology_selection,
        })
    }

    fn pick_at(
        &self,
        camera: &mut Camera,
        size: Vec2,
        position: Vec2,
    ) -> Option<(PickHit, PickRay)> {
        let origin = camera.screen_to_world(position, size, 0.0);
        let far = camera.screen_to_world(position, size, 1.0);
        let direction = (far - origin).normalize_or_zero();
        if direction.length_squared() <= f32::EPSILON {
            return None;
        }
        let ray = PickRay { origin, direction };
        let mut best = None;
        if let Some(root) = self.root {
            self.pick_node(root, &ray, &mut best);
        }
        best.map(|hit| (hit, ray))
    }

    fn pick_screen_topology(
        &self,
        camera: &mut Camera,
        size: Vec2,
        cursor: Vec2,
        mode: GeometryScenePickMode,
    ) -> Option<GeometryScenePickResult> {
        const OFFSETS: [[f32; 2]; 25] = [
            [0.0, 0.0],
            [3.0, 0.0],
            [-3.0, 0.0],
            [0.0, 3.0],
            [0.0, -3.0],
            [3.0, 3.0],
            [3.0, -3.0],
            [-3.0, 3.0],
            [-3.0, -3.0],
            [6.0, 0.0],
            [-6.0, 0.0],
            [0.0, 6.0],
            [0.0, -6.0],
            [6.0, 6.0],
            [6.0, -6.0],
            [-6.0, 6.0],
            [-6.0, -6.0],
            [12.0, 0.0],
            [-12.0, 0.0],
            [0.0, 12.0],
            [0.0, -12.0],
            [12.0, 12.0],
            [12.0, -12.0],
            [-12.0, 12.0],
            [-12.0, -12.0],
        ];
        let view_proj = camera.view_proj_matrix();
        let radius_px = match mode {
            GeometryScenePickMode::Edge => 10.0,
            GeometryScenePickMode::Point => 13.0,
            GeometryScenePickMode::Face => return None,
        };
        let mut visited_triangles = HashSet::new();
        let mut nearby_edges = HashSet::new();
        let mut best: Option<PickCandidate> = None;
        for offset in OFFSETS {
            let sample = cursor + Vec2::from_array(offset);
            if sample.x < 0.0 || sample.y < 0.0 || sample.x > size.x || sample.y > size.y {
                continue;
            }
            let Some((hit, _)) = self.pick_at(camera, size, sample) else {
                continue;
            };
            if !visited_triangles.insert(hit.triangle_index) {
                continue;
            }
            let triangle = &self.triangles[hit.triangle_index];
            let corners = [triangle.a, triangle.b, triangle.c];
            match mode {
                GeometryScenePickMode::Edge => {
                    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                        let edge_key = pick_edge_key(triangle, a, b);
                        if self.feature_edges.contains_key(&edge_key) {
                            nearby_edges.insert(edge_key);
                        }
                    }
                    // Rays just inside a face may never hit its neighbor across
                    // a crease. Include feature edges incident to its vertices.
                    for corner in corners {
                        let vertex_key = PickVertexKey {
                            chunk_index: triangle.chunk_index,
                            position: corner.to_array().map(f32::to_bits),
                        };
                        if let Some(neighbors) = self.feature_edge_neighbors.get(&vertex_key) {
                            nearby_edges.extend(neighbors.iter().cloned());
                        }
                    }
                }
                GeometryScenePickMode::Point => {
                    for (corner, position) in corners.into_iter().enumerate() {
                        let Some((screen_point, _)) = project_pick_point(view_proj, position, size)
                        else {
                            continue;
                        };
                        let distance_px_sq = cursor.distance_squared(screen_point);
                        if distance_px_sq > radius_px * radius_px {
                            continue;
                        }
                        let selection = GeometrySceneTopologySelection::Point {
                            chunk_id: triangle.chunk_id.clone(),
                            vertex_index: triangle.vertex_indices[corner],
                        };
                        self.consider_pick_candidate(
                            camera,
                            size,
                            screen_point,
                            PickCandidate {
                                triangle_index: hit.triangle_index,
                                position,
                                distance_px_sq,
                                endpoint_margin_px: 0.0,
                                projected_length_px: 0.0,
                                depth: camera.position.distance(position),
                                selection,
                                edge_key: None,
                            },
                            &mut best,
                        );
                    }
                }
                GeometryScenePickMode::Face => unreachable!(),
            }
        }
        if mode == GeometryScenePickMode::Edge {
            for edge_key in nearby_edges {
                let info = &self.feature_edges[&edge_key];
                let (Some(pa), Some(pb)) = (
                    project_pick_point(view_proj, info.positions[0], size),
                    project_pick_point(view_proj, info.positions[1], size),
                ) else {
                    continue;
                };
                let ab = pb.0 - pa.0;
                let length_px = ab.length();
                let t = ((cursor - pa.0).dot(ab) / ab.length_squared().max(1.0e-8)).clamp(0.0, 1.0);
                let screen_point = pa.0 + ab * t;
                let distance_px_sq = cursor.distance_squared(screen_point);
                if distance_px_sq > radius_px * radius_px {
                    continue;
                }
                let world_t = t * pa.1 / ((1.0 - t) * pb.1 + t * pa.1);
                let world_point = info.positions[0].lerp(info.positions[1], world_t);
                let triangle = &self.triangles[info.triangle_index];
                let selection = GeometrySceneTopologySelection::Edge {
                    chunk_id: triangle.chunk_id.clone(),
                    vertex_indices: info.vertex_indices,
                    segments: Vec::new(),
                };
                self.consider_pick_candidate(
                    camera,
                    size,
                    screen_point,
                    PickCandidate {
                        triangle_index: info.triangle_index,
                        position: world_point,
                        distance_px_sq,
                        endpoint_margin_px: t.min(1.0 - t) * length_px,
                        projected_length_px: length_px,
                        depth: camera.position.distance(world_point),
                        selection,
                        edge_key: Some(edge_key),
                    },
                    &mut best,
                );
            }
        }
        let mut best = best?;
        if let (Some(key), GeometrySceneTopologySelection::Edge { segments, .. }) =
            (&best.edge_key, &mut best.selection)
        {
            *segments = self.edge_chain(key);
        }
        self.result_for_triangle(
            best.triangle_index,
            best.position,
            best.depth,
            Some(best.selection),
        )
    }

    fn consider_pick_candidate(
        &self,
        camera: &mut Camera,
        size: Vec2,
        screen_point: Vec2,
        candidate: PickCandidate,
        best: &mut Option<PickCandidate>,
    ) {
        if let Some((front, ray)) = self.pick_at(camera, size, screen_point) {
            let candidate_distance = ray.origin.distance(candidate.position);
            let tolerance = self.scene_diagonal * 1.0e-3;
            if front.distance + tolerance < candidate_distance {
                return;
            }
        }
        if let Some(current) = best {
            // A line ending at the cursor is a weaker match than one running
            // under it. This resolves shared CAD corners without favoring a
            // short, foreshortened edge whose endpoint happens to be nearby.
            let score = |item: &PickCandidate| {
                let endpoint_penalty = if item.edge_key.is_some() {
                    (6.0 - item.endpoint_margin_px).max(0.0).powi(2)
                } else {
                    0.0
                };
                item.distance_px_sq + endpoint_penalty
            };
            let candidate_score = score(&candidate);
            let current_score = score(current);
            let tied_score = (candidate_score - current_score).abs() <= 1.0;
            let tied_depth =
                (candidate.depth - current.depth).abs() <= self.scene_diagonal * 1.0e-4;
            if candidate_score > current_score + 1.0
                || (tied_score && candidate.depth > current.depth && !tied_depth)
                || (tied_score
                    && tied_depth
                    && candidate.projected_length_px <= current.projected_length_px)
            {
                return;
            }
        }
        *best = Some(candidate);
    }

    fn edge_chain(&self, selected: &PickEdgeKey) -> Vec<[u32; 2]> {
        let Some(info) = self.feature_edges.get(selected) else {
            return Vec::new();
        };
        let mut segments = vec![info.vertex_indices];
        // Untagged mesh edges are individual finite elements. CAD face boundaries
        // can span many tessellation segments and should highlight as one edge.
        if info.regions.is_empty() {
            return segments;
        }
        let mut visited = HashSet::from([selected.clone()]);
        for (endpoint, other) in [(0, 1), (1, 0)] {
            let mut current = info.positions[endpoint];
            let mut direction = (current - info.positions[other]).normalize_or_zero();
            while segments.len() < 8192 {
                let vertex_key = PickVertexKey {
                    chunk_index: selected.chunk_index,
                    position: current.to_array().map(f32::to_bits),
                };
                let Some(neighbors) = self.feature_edge_neighbors.get(&vertex_key) else {
                    break;
                };
                let mut continuation: Option<(&PickEdgeKey, Vec3, Vec3, f32)> = None;
                for key in neighbors {
                    if visited.contains(key) {
                        continue;
                    }
                    let Some(candidate) = self.feature_edges.get(key) else {
                        continue;
                    };
                    if candidate.regions != info.regions {
                        continue;
                    }
                    let next = if candidate.positions[0].to_array().map(f32::to_bits)
                        == vertex_key.position
                    {
                        candidate.positions[1]
                    } else {
                        candidate.positions[0]
                    };
                    let next_direction = (next - current).normalize_or_zero();
                    let alignment = direction.dot(next_direction);
                    if alignment < 0.70 {
                        continue;
                    }
                    if continuation.as_ref().is_none_or(|item| alignment > item.3) {
                        continuation = Some((key, next, next_direction, alignment));
                    }
                }
                let Some((key, next, next_direction, _)) = continuation else {
                    break;
                };
                segments.push(self.feature_edges[key].vertex_indices);
                visited.insert(key.clone());
                current = next;
                direction = next_direction;
            }
        }
        segments
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeometrySceneCompleteness {
    Complete,
    Loading,
    BoundedPreview,
    FailedComplete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub scalar_legend: Option<GeometrySceneScalarLegend>,
    pub region_count: usize,
    pub mapped_region_count: usize,
    pub assembly_nodes: Vec<GeometrySceneAssemblyNode>,
    pub regions: Vec<GeometrySceneRegionSummary>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometrySceneScalarLegend {
    pub label: String,
    pub unit: String,
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometrySceneAssemblyNode {
    pub node_id: String,
    pub label: String,
    pub children: Vec<GeometrySceneAssemblyNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Topology lines generated for CAD display modes; shaded CAD stays smooth.
    pub wireframe_only: bool,
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
            wireframe_only: false,
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
            wireframe_only: false,
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

    /// Build CAD feature edges once. Tessellation diagonals within a smooth
    /// face are render details and should not appear as model edges.
    pub fn wireframe_edges(&self, color: [f32; 4]) -> Option<Self> {
        if self.render_data.pipeline_type != PipelineType::Triangles {
            return None;
        }
        struct EdgeIncidence {
            start: Vertex,
            end: Vertex,
            normal: Vec3,
            region_index: Option<usize>,
            count: usize,
            is_feature: bool,
        }
        let mut edge_indices = HashMap::new();
        let mut edges: Vec<EdgeIncidence> = Vec::new();
        let mut vertices = Vec::new();
        let indices = self.indices.as_deref();
        let vertex_count = indices.map_or(self.vertices.len(), |indices| indices.len());
        let triangle_count = vertex_count / 3;
        let mut triangle_regions = vec![None; triangle_count];
        let mut region_ids = HashMap::new();
        for region in &self.regions {
            let next_region_index = region_ids.len();
            let region_index = *region_ids
                .entry(region.region_id.as_str())
                .or_insert(next_region_index);
            for range in &region.triangle_ranges {
                let start = (range.start as usize).min(triangle_count);
                let end = start
                    .saturating_add(range.count as usize)
                    .min(triangle_count);
                for triangle_region in &mut triangle_regions[start..end] {
                    *triangle_region = Some(region_index);
                }
            }
        }
        for offset in (0..vertex_count).step_by(3) {
            if offset + 2 >= vertex_count {
                break;
            }
            let triangle = [0, 1, 2].map(|corner| {
                indices.map_or(offset + corner, |indices| indices[offset + corner] as usize)
            });
            let (Some(a), Some(b), Some(c)) = (
                self.vertices.get(triangle[0]),
                self.vertices.get(triangle[1]),
                self.vertices.get(triangle[2]),
            ) else {
                continue;
            };
            let normal = (Vec3::from_array(b.position) - Vec3::from_array(a.position))
                .cross(Vec3::from_array(c.position) - Vec3::from_array(a.position))
                .normalize_or_zero();
            let region_index = triangle_regions[offset / 3];
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                let (Some(mut start), Some(mut end)) = (
                    self.vertices.get(triangle[a]).copied(),
                    self.vertices.get(triangle[b]).copied(),
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
                if let Some(&index) = edge_indices.get(&edge) {
                    let existing: &mut EdgeIncidence = &mut edges[index];
                    existing.count += 1;
                    existing.is_feature |= existing.region_index != region_index
                        || existing.normal.dot(normal) < 0.966;
                } else {
                    start.color = color;
                    end.color = color;
                    edge_indices.insert(edge, edges.len());
                    edges.push(EdgeIncidence {
                        start,
                        end,
                        normal,
                        region_index,
                        count: 1,
                        is_feature: false,
                    });
                }
            }
        }
        for edge in edges {
            if edge.count == 1 || edge.is_feature || edge.region_index.is_none() {
                vertices.extend([edge.start, edge.end]);
            }
        }
        if vertices.is_empty() {
            return None;
        }
        let vertex_count = vertices.len();
        let material = Material {
            albedo: Vec4::ONE,
            ..Material::default()
        };
        let mut chunk = Self::from_render_data(
            format!("{}:wireframe-edges", self.chunk_id),
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
        chunk.mesh_id = self.mesh_id.clone();
        chunk.owner_node_ids = self.owner_node_ids.clone();
        chunk.visible = self.visible;
        chunk.wireframe_only = true;
        Some(chunk)
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
            GeometrySceneDisplayMode::Shaded
                if self.wireframe_only && !presentation.edge_overlay_enabled =>
            {
                render_data.draw_calls.clear();
            }
            GeometrySceneDisplayMode::Wireframe if !is_edge_chunk => {
                // A wireframe view is the topology, not a translucent shaded view. Keeping
                // triangle draw calls here both hid the mesh lines through depth writes and made
                // the mode look indistinguishable from shaded rendering on native surfaces.
                render_data.draw_calls.clear();
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
        if presentation.display_mode == GeometrySceneDisplayMode::Edges
            && is_edge_chunk
            && render_data.pipeline_type == PipelineType::Lines
        {
            render_data.pipeline_type = PipelineType::LinesNoDepth;
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
            log::debug!(
                target: "runmat_plot",
                "geometry_scene.selection_color chunk_id={} region_id={} colored_triangles={}",
                self.chunk_id,
                region_id,
                colored
            );
        }
        for selection in &presentation.selected_topology {
            if let GeometrySceneTopologySelection::Face {
                chunk_id,
                triangle_index,
            } = selection
            {
                if chunk_id == &self.chunk_id {
                    self.apply_triangle_color(
                        &mut render_data,
                        *triangle_index,
                        GEOMETRY_SELECTED_REGION_COLOR,
                    );
                }
            }
        }
        if let Some(region_id) = presentation.selected_region_id.as_deref() {
            let colored = self.apply_region_color(
                &mut render_data,
                region_id,
                GEOMETRY_SELECTED_REGION_COLOR,
            );
            log::debug!(
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
        matches!(
            self.render_data.pipeline_type,
            PipelineType::Lines | PipelineType::LinesNoDepth
        ) || self.chunk_id.contains(":edges")
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
            log::debug!(
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

    fn apply_triangle_color(
        &self,
        render_data: &mut RenderData,
        triangle_index: usize,
        color: [f32; 4],
    ) {
        if self.render_data.pipeline_type != PipelineType::Triangles {
            return;
        }
        let base = triangle_index.saturating_mul(3);
        if let (Some(source_indices), Some(render_indices)) =
            (self.indices.as_ref(), render_data.indices.as_mut())
        {
            let (Some(source), Some(target)) = (
                source_indices.get(base..base + 3),
                render_indices.get_mut(base..base + 3),
            ) else {
                return;
            };
            let mut isolated = [0_u32; 3];
            for (slot, index) in source.iter().copied().enumerate() {
                let Some(mut vertex) = render_data.vertices.get(index as usize).copied() else {
                    return;
                };
                vertex.color = color;
                let Ok(next_index) = u32::try_from(render_data.vertices.len()) else {
                    return;
                };
                render_data.vertices.push(vertex);
                isolated[slot] = next_index;
            }
            target.copy_from_slice(&isolated);
        } else if let Some(triangle) = render_data.vertices.get_mut(base..base + 3) {
            for vertex in triangle {
                vertex.color = color;
            }
        }
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
    chunk_index: usize,
    chunk_id: String,
    triangle_index: usize,
    vertex_indices: [u32; 3],
    region_id: Option<String>,
    region_label: Option<String>,
    region_tag: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PickEdgeKey {
    chunk_index: usize,
    endpoints: [[u32; 3]; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PickVertexKey {
    chunk_index: usize,
    position: [u32; 3],
}

#[derive(Debug, Clone)]
struct FeatureEdgeInfo {
    positions: [Vec3; 2],
    vertex_indices: [u32; 2],
    regions: Vec<String>,
    triangle_index: usize,
}

#[derive(Debug, Clone)]
struct PickCandidate {
    triangle_index: usize,
    position: Vec3,
    distance_px_sq: f32,
    endpoint_margin_px: f32,
    projected_length_px: f32,
    depth: f32,
    selection: GeometrySceneTopologySelection,
    edge_key: Option<PickEdgeKey>,
}

fn pick_edge_key(triangle: &IndexedTriangle, a: usize, b: usize) -> PickEdgeKey {
    let positions = [triangle.a, triangle.b, triangle.c];
    let first = positions[a].to_array().map(f32::to_bits);
    let second = positions[b].to_array().map(f32::to_bits);
    PickEdgeKey {
        chunk_index: triangle.chunk_index,
        endpoints: if first <= second {
            [first, second]
        } else {
            [second, first]
        },
    }
}

fn collect_feature_edges(
    triangles: &[IndexedTriangle],
) -> (
    HashMap<PickEdgeKey, FeatureEdgeInfo>,
    HashMap<PickVertexKey, Vec<PickEdgeKey>>,
) {
    struct Incidence {
        count: usize,
        region_id: Option<String>,
        regions: BTreeSet<String>,
        normal: Vec3,
        is_feature: bool,
        positions: [Vec3; 2],
        vertex_indices: [u32; 2],
        triangle_index: usize,
    }
    let mut edges: HashMap<PickEdgeKey, Incidence> = HashMap::new();
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        let normal = (triangle.b - triangle.a)
            .cross(triangle.c - triangle.a)
            .normalize_or_zero();
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let key = pick_edge_key(triangle, a, b);
            if let Some(existing) = edges.get_mut(&key) {
                existing.count += 1;
                existing.is_feature |=
                    existing.region_id != triangle.region_id || existing.normal.dot(normal) < 0.966;
                if let Some(region_id) = &triangle.region_id {
                    existing.regions.insert(region_id.clone());
                }
            } else {
                let positions = [triangle.a, triangle.b, triangle.c];
                edges.insert(
                    key,
                    Incidence {
                        count: 1,
                        region_id: triangle.region_id.clone(),
                        regions: triangle.region_id.iter().cloned().collect(),
                        normal,
                        is_feature: false,
                        positions: [positions[a], positions[b]],
                        vertex_indices: [triangle.vertex_indices[a], triangle.vertex_indices[b]],
                        triangle_index,
                    },
                );
            }
        }
    }
    let mut features = HashMap::new();
    let mut neighbors: HashMap<PickVertexKey, Vec<PickEdgeKey>> = HashMap::new();
    for (key, incidence) in edges {
        if incidence.count > 1 && !incidence.is_feature && incidence.region_id.is_some() {
            continue;
        }
        let info = FeatureEdgeInfo {
            positions: incidence.positions,
            vertex_indices: incidence.vertex_indices,
            regions: incidence.regions.into_iter().collect(),
            triangle_index: incidence.triangle_index,
        };
        for position in info.positions {
            neighbors
                .entry(PickVertexKey {
                    chunk_index: key.chunk_index,
                    position: position.to_array().map(f32::to_bits),
                })
                .or_default()
                .push(key.clone());
        }
        features.insert(key, info);
    }
    (features, neighbors)
}

fn project_pick_point(view_proj: Mat4, position: Vec3, screen_size: Vec2) -> Option<(Vec2, f32)> {
    let clip = view_proj * position.extend(1.0);
    if !clip.w.is_finite() || clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !ndc.is_finite() {
        return None;
    }
    Some((
        Vec2::new(
            (ndc.x + 1.0) * screen_size.x * 0.5,
            (1.0 - ndc.y) * screen_size.y * 0.5,
        ),
        clip.w,
    ))
}

fn triangle_barycentric_coordinates(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [f32; 3] {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = point - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() <= f32::EPSILON {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / denominator;
    let w = (d00 * d21 - d01 * d20) / denominator;
    [1.0 - v - w, v, w]
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
    // A fixed determinant cutoff depends on coordinate units and silently drops
    // small CAD faces. Use f64 and a dimensionless relative parallelism test.
    let a = a.as_dvec3();
    let edge1 = b.as_dvec3() - a;
    let edge2 = c.as_dvec3() - a;
    let direction = ray.direction.as_dvec3();
    let pvec = direction.cross(edge2);
    let det = edge1.dot(pvec);
    let scale = edge1.length() * pvec.length();
    if !det.is_finite() || !scale.is_finite() || det.abs() <= scale * (16.0 * f64::EPSILON) {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ray.origin.as_dvec3() - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(edge1);
    let v = direction.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = (edge2.dot(qvec) * inv_det) as f32;
    (t.is_finite() && t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProjectionType;

    #[test]
    fn geometry_scene_defaults_to_grid_and_presentation_can_override_it() {
        let scene = GeometryScene::new("scene", 1, Vec::new());
        assert!(scene.show_grid);
        let presentation: GeometryScenePresentation =
            serde_json::from_value(serde_json::json!({ "showGrid": false })).unwrap();
        assert_eq!(presentation.show_grid, Some(false));
        assert_eq!(GeometryScenePresentation::default().show_grid, None);
    }

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
            mode: GeometryScenePickMode::Face,
        });
        let hit = hit.expect("center ray should hit the triangle");
        assert_eq!(hit.region_id, Some("face_a".to_string()));
        assert_eq!(hit.vertex_indices, [0, 1, 2]);
        assert!((hit.barycentric_coordinates.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn edge_pick_ignores_face_diagonals_and_prefers_visible_front_edge() {
        fn square(chunk_id: &str, z: f32) -> GeometrySceneChunk {
            GeometrySceneChunk::indexed_triangles(
                chunk_id,
                [
                    [-1.0, -1.0, z],
                    [1.0, -1.0, z],
                    [1.0, 1.0, z],
                    [0.0, 1.0, z],
                    [-1.0, 1.0, z],
                ]
                .map(|position| vertex(position, [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]))
                .to_vec(),
                vec![0, 1, 2, 0, 2, 3, 0, 3, 4],
                cad_default_material(),
            )
            .with_regions(vec![GeometrySceneRegion::new(
                "face",
                None,
                None,
                vec![GeometrySceneTriangleRange::new(0, 3)],
            )])
        }
        let scene = GeometryScene::new(
            "stacked",
            1,
            vec![square("rear", -1.0), square("front", 0.0)],
        );
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
        let top = project_pick_point(
            camera.view_proj_matrix(),
            Vec3::new(0.0, 1.0, 0.0),
            Vec2::splat(800.0),
        )
        .unwrap()
        .0;
        let edge = index
            .pick(GeometryScenePickRequest {
                camera: camera.clone(),
                surface_size: [800.0, 800.0],
                position: [top.x, top.y - 4.0],
                mode: GeometryScenePickMode::Edge,
            })
            .expect("visible top edge should be selectable just outside the face");
        assert_eq!(edge.chunk_id, "front");
        let Some(GeometrySceneTopologySelection::Edge {
            chunk_id, segments, ..
        }) = edge.topology_selection
        else {
            panic!("expected a selected edge");
        };
        assert_eq!(chunk_id, "front");
        assert_eq!(segments.len(), 2);
        assert!(segments
            .iter()
            .any(|segment| *segment == [2, 3] || *segment == [3, 2]));
        assert!(segments
            .iter()
            .any(|segment| *segment == [3, 4] || *segment == [4, 3]));
        assert!(
            index
                .pick(GeometryScenePickRequest {
                    camera: camera.clone(),
                    surface_size: [800.0, 800.0],
                    position: [400.0, 400.0],
                    mode: GeometryScenePickMode::Edge,
                })
                .is_none(),
            "the coplanar triangulation diagonal is not a feature edge"
        );
        let corner = project_pick_point(
            camera.view_proj_matrix(),
            Vec3::new(1.0, 1.0, 0.0),
            Vec2::splat(800.0),
        )
        .unwrap()
        .0;
        let point = index
            .pick(GeometryScenePickRequest {
                camera,
                surface_size: [800.0, 800.0],
                position: [corner.x + 4.0, corner.y - 4.0],
                mode: GeometryScenePickMode::Point,
            })
            .expect("visible front vertex should be selectable near the silhouette");
        assert_eq!(
            point.topology_selection,
            Some(GeometrySceneTopologySelection::Point {
                chunk_id: "front".into(),
                vertex_index: 2,
            })
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
    fn wireframe_presentation_draws_only_explicit_edge_chunks() {
        let chunk = GeometrySceneChunk::indexed_triangles(
            "solver-surface",
            vec![
                vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2],
            cad_default_material(),
        );

        let render_data = chunk.render_data_with_presentation(&GeometryScenePresentation {
            display_mode: GeometrySceneDisplayMode::Wireframe,
            ..Default::default()
        });

        assert!(render_data.draw_calls.is_empty());
        let xray = chunk.render_data_with_presentation(&GeometryScenePresentation {
            display_mode: GeometrySceneDisplayMode::Edges,
            ..Default::default()
        });
        assert_eq!(xray.material.albedo.w, 0.38);
        let mut edge_data = chunk.render_data.clone();
        edge_data.pipeline_type = PipelineType::Lines;
        edge_data.indices = None;
        edge_data.vertices.truncate(2);
        edge_data.draw_calls = vec![DrawCall {
            vertex_offset: 0,
            vertex_count: 2,
            index_offset: None,
            index_count: None,
            instance_count: 1,
        }];
        let edge_chunk = GeometrySceneChunk::from_render_data("solver-mesh-edges", edge_data);
        let xray_edges = edge_chunk.render_data_with_presentation(&GeometryScenePresentation {
            display_mode: GeometrySceneDisplayMode::Edges,
            ..Default::default()
        });
        assert_eq!(xray_edges.pipeline_type, PipelineType::LinesNoDepth);
    }

    #[test]
    fn cad_wireframe_excludes_coplanar_tessellation_diagonals() {
        let surface = GeometrySceneChunk::indexed_triangles(
            "cad-surface",
            vec![
                vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2, 1, 3, 2],
            cad_default_material(),
        )
        .with_regions(vec![GeometrySceneRegion::new(
            "cad-face",
            None,
            None,
            vec![GeometrySceneTriangleRange { start: 0, count: 2 }],
        )]);
        let edges = surface.wireframe_edges([0.58, 0.72, 0.88, 0.95]).unwrap();
        assert_eq!(edges.render_data.vertices.len(), 8); // four boundary edges
        let untagged = GeometrySceneChunk {
            regions: Vec::new(),
            ..surface.clone()
        };
        assert_eq!(
            untagged
                .wireframe_edges([0.58, 0.72, 0.88, 0.95])
                .unwrap()
                .render_data
                .vertices
                .len(),
            10
        );
        let shaded = GeometryScenePresentation::default();
        assert!(!surface
            .render_data_with_presentation(&shaded)
            .draw_calls
            .is_empty());
        assert!(!edges
            .render_data_with_presentation(&shaded)
            .draw_calls
            .is_empty());
        assert!(edges
            .render_data_with_presentation(&GeometryScenePresentation {
                edge_overlay_enabled: false,
                ..Default::default()
            })
            .draw_calls
            .is_empty());
        let wireframe = GeometryScenePresentation {
            display_mode: GeometrySceneDisplayMode::Wireframe,
            ..Default::default()
        };
        assert!(surface
            .render_data_with_presentation(&wireframe)
            .draw_calls
            .is_empty());
        assert!(!edges
            .render_data_with_presentation(&wireframe)
            .draw_calls
            .is_empty());
    }

    #[test]
    fn cad_wireframe_keeps_a_boundary_between_coplanar_regions() {
        let surface = GeometrySceneChunk::indexed_triangles(
            "cad-surface",
            vec![
                vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2, 1, 3, 2],
            cad_default_material(),
        )
        .with_regions(vec![
            GeometrySceneRegion::new(
                "left",
                None,
                None,
                vec![GeometrySceneTriangleRange { start: 0, count: 1 }],
            ),
            GeometrySceneRegion::new(
                "right",
                None,
                None,
                vec![GeometrySceneTriangleRange { start: 1, count: 1 }],
            ),
        ]);
        let edges = surface.wireframe_edges([0.58, 0.72, 0.88, 0.95]).unwrap();
        assert_eq!(edges.render_data.vertices.len(), 10); // outer boundary plus shared CAD edge
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
    fn presentation_region_annotations_emit_typed_marker_and_symbol_nodes() {
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
            region_annotations: vec![
                GeometrySceneRegionAnnotation {
                    region_id: "loaded_face".to_string(),
                    color: Some([0.9, 0.1, 0.1, 1.0]),
                    kind: GeometrySceneAnnotationKind::Force,
                    label: Some("load".to_string()),
                    direction: Some([0.0, 0.0, 1.0]),
                    size: Some(18.0),
                },
                GeometrySceneRegionAnnotation {
                    region_id: "loaded_face".to_string(),
                    color: Some([0.2, 0.3, 0.8, 1.0]),
                    kind: GeometrySceneAnnotationKind::InitialCondition,
                    label: Some("initial".to_string()),
                    direction: None,
                    size: Some(18.0),
                },
            ],
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
            .find(|node| node.name == "FEA study symbols")
            .and_then(|node| node.render_data.as_ref())
            .expect("vector annotation node");
        assert_eq!(vector.pipeline_type, PipelineType::AnnotationStroke);
        assert_eq!(vector.vertices.len(), 10 * 6);
        assert!(vector.vertices[2].position[2] > vector.vertices[0].position[2]);
        assert!(vector
            .vertices
            .iter()
            .all(|vertex| vertex.position[2] >= 0.0));
    }

    #[test]
    fn face_annotations_use_one_support_and_a_distributed_load_group() {
        let chunk = GeometrySceneChunk::indexed_triangles(
            "rectangular_face",
            vec![
                vertex([-2.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([2.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([2.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([-2.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2, 0, 2, 3],
            cad_default_material(),
        )
        .with_regions(vec![GeometrySceneRegion::new(
            "face",
            None,
            None,
            vec![GeometrySceneTriangleRange::new(0, 2)],
        )]);
        let scene = GeometryScene::new("face", 1, vec![chunk]);
        let annotations = [
            GeometrySceneAnnotationKind::Traction,
            GeometrySceneAnnotationKind::FixedSupport,
        ]
        .into_iter()
        .map(|kind| GeometrySceneRegionAnnotation {
            region_id: "face".into(),
            color: None,
            kind,
            label: None,
            direction: Some([0.0, 0.0, 1.0]),
            size: None,
        })
        .collect();
        let nodes = scene.nodes_with_presentation(&GeometryScenePresentation {
            region_annotations: annotations,
            ..Default::default()
        });
        let symbols = nodes
            .iter()
            .find(|node| node.name == "FEA study symbols")
            .and_then(|node| node.render_data.as_ref())
            .unwrap();
        assert_eq!(symbols.pipeline_type, PipelineType::AnnotationStroke);
        // Three filled-head load arrows (ten segments each) and one six-segment support.
        assert_eq!(symbols.vertices.len(), (3 * 10 + 6) * 6);
        assert!(symbols.vertices[0].position != symbols.vertices[60].position);
        assert!(symbols.vertices[60].color[3] > 0.0);
        assert_eq!(symbols.vertices[180].color[3], 0.0);
        // The far endpoint reverses its tangent, so its ribbon side signs reverse too.
        assert_eq!(symbols.vertices[0].tex_coords[0], -1.0);
        assert_eq!(symbols.vertices[2].tex_coords[0], -1.0);
    }

    #[test]
    fn directional_annotation_without_a_direction_uses_a_marker() {
        let chunk = GeometrySceneChunk::indexed_triangles(
            "face",
            vec![
                vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2],
            cad_default_material(),
        )
        .with_regions(vec![GeometrySceneRegion::new(
            "face",
            None,
            None,
            vec![GeometrySceneTriangleRange::new(0, 1)],
        )]);
        let scene = GeometryScene::new("face", 1, vec![chunk]);
        let nodes = scene.nodes_with_presentation(&GeometryScenePresentation {
            region_annotations: vec![GeometrySceneRegionAnnotation {
                region_id: "face".into(),
                color: None,
                kind: GeometrySceneAnnotationKind::HeatFlux,
                label: None,
                direction: None,
                size: None,
            }],
            ..Default::default()
        });
        assert!(nodes.iter().any(|node| node.name == "FEA region markers"));
        assert!(!nodes.iter().any(|node| node.name == "FEA study symbols"));
    }

    #[test]
    fn topology_selection_colors_face_and_draws_edge_and_point() {
        let wire = serde_json::to_value(GeometrySceneTopologySelection::Edge {
            chunk_id: "picked_chunk".into(),
            vertex_indices: [0, 1],
            segments: Vec::new(),
        })
        .unwrap();
        assert_eq!(
            wire,
            serde_json::json!({
                "kind": "edge", "chunkId": "picked_chunk", "vertexIndices": [0, 1]
            })
        );
        let incoming: GeometryScenePresentation = serde_json::from_value(serde_json::json!({
            "selectedTopology": [
                { "kind": "edge", "chunkId": "picked_chunk", "vertexIndices": [0, 1] },
                { "kind": "point", "chunkId": "picked_chunk", "vertexIndex": 2 }
            ]
        }))
        .unwrap();
        assert_eq!(incoming.selected_topology.len(), 2);
        let chunk = GeometrySceneChunk::indexed_triangles(
            "picked_chunk",
            vec![
                vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2],
            cad_default_material(),
        );
        let scene = GeometryScene::new("picked", 1, vec![chunk]);
        let nodes = scene.nodes_with_presentation(&GeometryScenePresentation {
            selected_topology: vec![
                GeometrySceneTopologySelection::Face {
                    chunk_id: "picked_chunk".into(),
                    triangle_index: 0,
                },
                GeometrySceneTopologySelection::Edge {
                    chunk_id: "picked_chunk".into(),
                    vertex_indices: [0, 1],
                    segments: vec![[0, 1], [1, 2]],
                },
                GeometrySceneTopologySelection::Point {
                    chunk_id: "picked_chunk".into(),
                    vertex_index: 2,
                },
            ],
            ..Default::default()
        });
        let face = nodes[0].render_data.as_ref().unwrap();
        assert_eq!(face.indices.as_ref().unwrap()[0], 3);
        assert_eq!(face.vertices[3].color, GEOMETRY_SELECTED_REGION_COLOR);
        let topology = nodes
            .iter()
            .find(|node| node.name == "Selected geometry topology")
            .and_then(|node| node.render_data.as_ref())
            .unwrap();
        assert_eq!(topology.pipeline_type, PipelineType::AnnotationStroke);
        assert_eq!(topology.vertices.len(), 5 * 6);
        assert_eq!(topology.vertices[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(topology.vertices[0].tex_coords[1], 1.25);
        assert_eq!(topology.vertices[6].tex_coords[1], 1.0);
        assert_eq!(topology.vertices[12].position, [1.0, 0.0, 0.0]);
        assert_eq!(topology.vertices[24].position, [0.0, 1.0, 0.0]);
        assert_eq!(topology.vertices[24].tex_coords[1], 2.0);
        assert_eq!(topology.vertices[24].normal, [-1.0, -1.0, 0.0]);
    }
    #[test]
    fn small_cad_face_picking_is_invariant_to_coordinate_scale() {
        for scale in [1.0e-9, 1.0e-3, 1.0, 1.0e9] {
            let ray = PickRay {
                origin: Vec3::new(0.25, 0.25, 1.0) * scale,
                direction: -Vec3::Z,
            };
            let hit = ray_intersects_triangle(&ray, Vec3::ZERO, Vec3::X * scale, Vec3::Y * scale)
                .expect("valid face must remain selectable");
            assert!((hit / scale - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn scale_independent_picking_still_rejects_parallel_and_behind_faces() {
        for direction in [Vec3::X, Vec3::Z, Vec3::ZERO] {
            let ray = PickRay {
                origin: Vec3::new(0.25, 0.25, 1.0),
                direction,
            };
            assert!(ray_intersects_triangle(&ray, Vec3::ZERO, Vec3::X, Vec3::Y).is_none());
        }
    }
}
