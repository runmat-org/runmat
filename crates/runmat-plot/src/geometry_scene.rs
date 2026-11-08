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
mod instances;
mod paint;
mod pick;
mod pick_instances;
mod section_cap;
mod section_cut;
#[cfg(test)]
mod section_cut_tests;

pub use pick::{
    GeometryScenePickIndex, GeometryScenePickMode, GeometryScenePickRequest, GeometryScenePickResult,
};

pub use instances::GeometrySceneInstance;
use paint::{RegionPaint, HOVER_PAINT, SELECTED_PAINT};
#[cfg(test)]
use paint::GEOMETRY_HOVER_EDGE_COLOR;

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
    /// Draw the corner orientation triad (on by default).
    pub show_orientation_triad: bool,
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
            show_orientation_triad: true,
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
            .map(|(index, chunk)| {
                // A section clips world triangles, so an instanced chunk draws flattened.
                let flat;
                let (chunk, instances) = if chunk.is_instanced() && presentation.active_section().is_some() {
                    flat = chunk.flattened();
                    (&flat, None)
                } else if chunk.is_instanced() {
                    (chunk, Some(chunk.node_instances(presentation)))
                } else {
                    (chunk, None)
                };
                SceneNode {
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
                instances,
            }})
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
    /// Analysis colors (draft, thickness, curvature): each listed region's faces fill with
    /// its color while its edges keep theirs. Highlights, hover and selection paint over them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub region_colors: Vec<GeometrySceneRegionColor>,
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
    /// Regions left out of the view: their triangles and boundary segments draw nothing and
    /// picks pass through them, as hidden owner nodes do.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hidden_region_ids: Vec<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_explicit_optional_section",
        skip_serializing_if = "Option::is_none"
    )]
    pub section: Option<Option<GeometrySceneSection>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view_preset: Option<GeometrySceneViewPreset>,
    /// Perspective or orthographic (parallel) projection for the 3-D view. Unset keeps the
    /// current one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<GeometrySceneProjection>,
    /// Width of line chunks (edges, wireframes) in CSS px. Above 1 px they draw as screen-space
    /// strokes of that width; unset or 1 px or less draws GPU hairlines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_width_px: Option<f32>,
}

/// How the 3-D view projects: with perspective, or parallel as CAD drafting views do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GeometrySceneProjection {
    Perspective,
    Orthographic,
}

impl Default for GeometryScenePresentation {
    fn default() -> Self {
        Self {
            selected_region_id: None,
            selected_region_ids: Vec::new(),
            selected_topology: Vec::new(),
            hovered_region_id: None,
            region_colors: Vec::new(),
            region_highlights: Vec::new(),
            region_annotations: Vec::new(),
            display_mode: GeometrySceneDisplayMode::Shaded,
            edge_overlay_enabled: true,
            show_grid: None,
            hidden_owner_node_ids: None,
            isolated_owner_node_ids: None,
            hidden_region_ids: Vec::new(),
            section: None,
            view_preset: None,
            projection: None,
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
            || !self.region_colors.is_empty()
            || !self.region_highlights.is_empty()
            || !self.hidden_region_ids.is_empty()
            || self.active_section().is_some()
    }

    /// Whether a region's annotations (labels, markers, load arrows) are drawn: a hidden
    /// region takes its annotations with it.
    pub(crate) fn shows_region(&self, region_id: &str) -> bool {
        !self.hidden_region_ids.iter().any(|id| id == region_id)
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
    /// A second cutting plane. With two planes the view removes only what lies on the removed
    /// side of both, a corner cutaway; each plane is capped where the removed corner meets it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second_plane: Option<GeometrySceneSectionPlane>,
    /// Fill the solid's cut faces. Off leaves the cut open, showing the model's inside.
    #[serde(default = "default_section_cap")]
    pub cap: bool,
}

fn default_section_cap() -> bool {
    true
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

/// One region's analysis color: its faces fill with `color` and its edges keep theirs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySceneRegionColor {
    pub region_id: String,
    pub color: [f32; 4],
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


fn default_edge_overlay_enabled() -> bool {
    true
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
    /// An acceptance limit on the shown quantity, in `unit`: marked on the bar when it falls in
    /// the range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
    /// A line under the range, such as how far the result is from that limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
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
    /// Regions of a line chunk, with ranges counted in segments (index pairs). A CAD face's or
    /// occurrence's boundary edges carry its region id, so selection and hover color them too.
    pub segment_regions: Vec<GeometrySceneRegion>,
    pub owner_node_ids: Vec<String>,
    pub visible: bool,
    /// Topology lines generated for CAD display modes; shaded CAD stays smooth.
    pub wireframe_only: bool,
    /// Placements of an instanced chunk (see [`instances`]): its vertices are then the placed
    /// mesh's own coordinates and each placement is one region.
    pub instances: Vec<GeometrySceneInstance>,
    /// The projected diameters, in device pixels, an instanced chunk draws its placements at.
    pub lod_pixels: Option<[f32; 2]>,
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
            segment_regions: Vec::new(),
            owner_node_ids: Vec::new(),
            visible: true,
            wireframe_only: false,
            instances: Vec::new(),
            lod_pixels: None,
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
            segment_regions: Vec::new(),
            owner_node_ids: Vec::new(),
            visible: true,
            wireframe_only: false,
            instances: Vec::new(),
            lod_pixels: None,
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

    /// One chunk drawing every chunk in `chunks`, which must share a pipeline and be all indexed
    /// or all unindexed; the first chunk's material and display flags apply to all. Regions,
    /// owners and mesh ids are dropped: use it for geometry drawn as one piece, such as context.
    /// Returns `None` for no chunks, mixed chunks or more than `u32::MAX` vertices.
    pub fn merged(chunk_id: impl Into<String>, chunks: Vec<Self>) -> Option<Self> {
        let first = chunks.first()?;
        let pipeline_type = first.render_data.pipeline_type;
        let indexed = first.indices.is_some();
        let material = first.material.clone();
        let wireframe_only = first.wireframe_only;
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for chunk in &chunks {
            if chunk.render_data.pipeline_type != pipeline_type
                || chunk.indices.is_some() != indexed
            {
                return None;
            }
            let offset = u32::try_from(vertices.len()).ok()?;
            vertices.extend_from_slice(&chunk.vertices);
            if let Some(chunk_indices) = &chunk.indices {
                indices.extend(chunk_indices.iter().map(|index| index + offset));
            }
        }
        u32::try_from(vertices.len()).ok()?;
        let vertex_count = vertices.len();
        let index_count = indices.len();
        let mut merged = Self::from_render_data(
            chunk_id,
            RenderData {
                pipeline_type,
                vertices,
                indices: indexed.then_some(indices),
                gpu_vertices: None,
                bounds: None,
                material,
                draw_calls: vec![DrawCall {
                    vertex_offset: 0,
                    vertex_count,
                    index_offset: indexed.then_some(0),
                    index_count: indexed.then_some(index_count),
                    instance_count: 1,
                }],
                image: None,
            },
        );
        merged.wireframe_only = wireframe_only;
        Some(merged)
    }

    /// Restyle this chunk as translucent context geometry: surfaces take `face` (its alpha
    /// below 1 makes them blended), edge lines take `edge`. A scene that also holds opaque
    /// surfaces draws these after them without depth writes, so context never hides a result.
    pub fn into_context(mut self, face: [f32; 4], edge: [f32; 4]) -> Self {
        let color = if self.is_edge_chunk() { edge } else { face };
        let blended = color[3] < 1.0;
        for vertex in self
            .vertices
            .iter_mut()
            .chain(self.render_data.vertices.iter_mut())
        {
            vertex.color = color;
        }
        for material in [&mut self.material, &mut self.render_data.material] {
            material.albedo = Vec4::from_array(color);
            if blended {
                material.alpha_mode = AlphaMode::Blend;
            }
        }
        // The colour lives in the vertices now; a stale GPU copy would keep the old one.
        self.render_data.gpu_vertices = None;
        // Context edges are part of its look in every display mode.
        self.wireframe_only = false;
        self
    }

    pub fn with_regions(mut self, regions: Vec<GeometrySceneRegion>) -> Self {
        self.regions = regions;
        self
    }

    pub fn with_segment_regions(mut self, segment_regions: Vec<GeometrySceneRegion>) -> Self {
        self.segment_regions = segment_regions;
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

        for region in &presentation.region_colors {
            self.apply_region_color(&mut render_data, &region.region_id, RegionPaint::Face(region.color));
        }
        for highlight in &presentation.region_highlights {
            self.apply_region_color(
                &mut render_data,
                &highlight.region_id,
                RegionPaint::Fill(
                    highlight
                        .color
                        .unwrap_or_else(|| geometry_region_role_color(highlight.role.as_deref())),
                ),
            );
        }
        if let Some(region_id) = presentation.hovered_region_id.as_deref() {
            self.apply_region_color(&mut render_data, region_id, HOVER_PAINT);
        }
        for region_id in &presentation.selected_region_ids {
            let colored = self.apply_region_color(&mut render_data, region_id, SELECTED_PAINT);
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
                    self.apply_triangle_color(&mut render_data, *triangle_index, SELECTED_PAINT);
                }
            }
        }
        if let Some(region_id) = presentation.selected_region_id.as_deref() {
            let colored = self.apply_region_color(&mut render_data, region_id, SELECTED_PAINT);
            log::debug!(
                target: "runmat_plot",
                "geometry_scene.selection_color chunk_id={} region_id={} colored_triangles={}",
                self.chunk_id,
                region_id,
                colored
            );
        }
        if !presentation.hidden_region_ids.is_empty() {
            render_data = self.render_data_without_regions(render_data, &presentation.hidden_region_ids);
        }
        if let Some(section) = presentation.active_section() {
            render_data = self.render_data_with_section(render_data, section);
        }
        render_data
    }

    /// Leaves out the triangles (or, on a line chunk, the segments) of the hidden regions.
    fn render_data_without_regions(&self, render_data: RenderData, hidden: &[String]) -> RenderData {
        let lines = matches!(
            render_data.pipeline_type,
            PipelineType::Lines | PipelineType::LinesNoDepth
        );
        let regions = if lines { &self.segment_regions } else { &self.regions };
        let ranges: Vec<GeometrySceneTriangleRange> = regions
            .iter()
            .filter(|region| hidden.iter().any(|id| id == &region.region_id))
            .flat_map(|region| region.triangle_ranges.iter().copied())
            .collect();
        if ranges.is_empty() {
            return render_data;
        }
        let per = if lines { 2 } else if render_data.pipeline_type == PipelineType::Triangles { 3 } else { return render_data };
        section_cut::without_primitives(render_data, per, &ranges)
    }

    fn render_data_with_section(
        &self,
        render_data: RenderData,
        section: &GeometrySceneSection,
    ) -> RenderData {
        let Some(plane) = section_plane(&section.plane) else {
            return render_data;
        };
        let second = section.second_plane.as_ref().and_then(section_plane);
        section_cut::cut(render_data, plane, second, section.cap)
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

/// Every triangle of `render_data`, indexed or not.
fn triangles_of(render_data: &RenderData) -> Vec<[Vertex; 3]> {
    match render_data.indices.as_ref() {
        Some(indices) => indices
            .chunks_exact(3)
            .filter_map(|t| triangle_vertices_from_indices(&render_data.vertices, t))
            .collect(),
        None => render_data
            .vertices
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect(),
    }
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

#[cfg(test)]
mod tests;
