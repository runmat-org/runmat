//! Indexed triangle mesh plot primitive.

mod render;
mod geometry;
#[cfg(test)]
mod tests;
use crate::core::{AlphaMode, BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{DVec3, Vec3, Vec4};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshTriangleRange {
    pub start: u32,
    pub count: u32,
}

impl MeshTriangleRange {
    pub fn new(start: u32, count: u32) -> Self {
        Self { start, count }
    }

    pub fn end_exclusive(&self) -> Option<u32> {
        self.start.checked_add(self.count)
    }

    pub fn contains(&self, triangle_index: u32) -> bool {
        self.end_exclusive()
            .is_some_and(|end| triangle_index >= self.start && triangle_index < end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshRegion {
    pub region_id: String,
    pub label: Option<String>,
    pub tag: Option<String>,
    pub triangle_ranges: Vec<MeshTriangleRange>,
}

impl MeshRegion {
    pub fn new(
        region_id: impl Into<String>,
        label: Option<String>,
        tag: Option<String>,
        triangle_ranges: Vec<MeshTriangleRange>,
    ) -> Self {
        Self {
            region_id: region_id.into(),
            label,
            tag,
            triangle_ranges,
        }
    }

    pub fn contains_triangle(&self, triangle_index: u32) -> bool {
        self.triangle_ranges
            .iter()
            .any(|range| range.contains(triangle_index))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshFieldLocation {
    Vertex,
    Triangle,
}

impl MeshFieldLocation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Triangle => "triangle",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "vertex" | "vertices" | "node" | "nodes" => Some(Self::Vertex),
            "triangle" | "triangles" | "face" | "faces" | "element" | "elements" => {
                Some(Self::Triangle)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshEdgeMode {
    #[default]
    All,
    Feature,
    None,
}

impl MeshEdgeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Feature => "feature",
            Self::None => "none",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "all" | "triangle" | "triangles" => Some(Self::All),
            "feature" | "features" | "boundary" | "boundaries" | "region" | "regions" => {
                Some(Self::Feature)
            }
            "none" | "off" => Some(Self::None),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshScalarField {
    pub field_id: String,
    pub label: Option<String>,
    pub location: MeshFieldLocation,
    pub values: Vec<f32>,
    pub color_limits: Option<[f32; 2]>,
    pub colormap: String,
    pub alpha: f32,
}

impl MeshScalarField {
    pub fn new(field_id: impl Into<String>, location: MeshFieldLocation, values: Vec<f32>) -> Self {
        Self {
            field_id: field_id.into(),
            label: None,
            location,
            values,
            color_limits: None,
            colormap: "viridis".to_string(),
            alpha: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshVectorField {
    pub field_id: String,
    pub label: Option<String>,
    pub location: MeshFieldLocation,
    pub vectors: Vec<Vec3>,
    pub scale: f32,
    pub stride: usize,
    pub color: Vec4,
}

impl MeshVectorField {
    pub fn new(
        field_id: impl Into<String>,
        location: MeshFieldLocation,
        vectors: Vec<Vec3>,
    ) -> Self {
        Self {
            field_id: field_id.into(),
            label: None,
            location,
            vectors,
            scale: 1.0,
            stride: 1,
            color: Vec4::new(0.95, 0.86, 0.28, 1.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshDeformation {
    pub field_id: String,
    pub label: Option<String>,
    pub displacements: Vec<Vec3>,
    pub scale: f32,
}

impl MeshDeformation {
    pub fn new(field_id: impl Into<String>, displacements: Vec<Vec3>) -> Self {
        Self {
            field_id: field_id.into(),
            label: None,
            displacements,
            scale: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeshPlot {
    mesh_id: Option<String>,
    vertices: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    face_color: Vec4,
    edge_color: Vec4,
    face_alpha: f32,
    edge_alpha: f32,
    edge_width: f32,
    edge_mode: MeshEdgeMode,
    feature_edge_groups: Option<Vec<u64>>,
    vertex_colors: Option<Vec<Vec4>>,
    triangle_colors: Option<Vec<Vec4>>,
    /// Surface normals per vertex from the source (a CAD kernel's exact surface normals);
    /// `None` shades with area-weighted triangle normals.
    vertex_normals: Option<Vec<Vec3>>,
    label: Option<String>,
    regions: Vec<MeshRegion>,
    highlighted_region_id: Option<String>,
    highlight_color: Vec4,
    scalar_field: Option<MeshScalarField>,
    vector_field: Option<MeshVectorField>,
    deformation: Option<MeshDeformation>,
    visible: bool,
    bounds: Option<BoundingBox>,
    face_vertices: Option<Vec<Vertex>>,
    face_indices: Option<Vec<u32>>,
    edge_vertices: Option<Vec<Vertex>>,
    vector_vertices: Option<Vec<Vertex>>,
    dirty: bool,
}

impl MeshPlot {
    pub fn new(vertices: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> Result<Self, String> {
        validate_mesh(&vertices, &triangles)?;
        Ok(Self {
            mesh_id: None,
            vertices,
            triangles,
            face_color: Vec4::new(0.18, 0.48, 0.86, 1.0),
            edge_color: Vec4::new(0.78, 0.86, 0.96, 1.0),
            face_alpha: 1.0,
            edge_alpha: 0.65,
            edge_width: 0.5,
            edge_mode: MeshEdgeMode::All,
            feature_edge_groups: None,
            vertex_colors: None,
            triangle_colors: None,
            vertex_normals: None,
            label: None,
            regions: Vec::new(),
            highlighted_region_id: None,
            highlight_color: Vec4::new(0.98, 0.78, 0.22, 1.0),
            scalar_field: None,
            vector_field: None,
            deformation: None,
            visible: true,
            bounds: None,
            face_vertices: None,
            face_indices: None,
            edge_vertices: None,
            vector_vertices: None,
            dirty: true,
        })
    }

    pub fn mesh_id(&self) -> Option<&str> {
        self.mesh_id.as_deref()
    }

    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    pub fn face_color(&self) -> Vec4 {
        self.face_color
    }

    pub fn edge_color(&self) -> Vec4 {
        self.edge_color
    }

    pub fn face_alpha(&self) -> f32 {
        self.face_alpha
    }

    pub fn edge_alpha(&self) -> f32 {
        self.edge_alpha
    }

    pub fn edge_width(&self) -> f32 {
        self.edge_width
    }

    pub fn edge_mode(&self) -> MeshEdgeMode {
        self.edge_mode
    }

    pub fn feature_edge_groups(&self) -> Option<&[u64]> {
        self.feature_edge_groups.as_deref()
    }

    pub fn triangle_colors(&self) -> Option<&[Vec4]> {
        self.triangle_colors.as_deref()
    }

    pub fn vertex_colors(&self) -> Option<&[Vec4]> {
        self.vertex_colors.as_deref()
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn regions(&self) -> &[MeshRegion] {
        &self.regions
    }

    pub fn highlighted_region_id(&self) -> Option<&str> {
        self.highlighted_region_id.as_deref()
    }

    pub fn highlight_color(&self) -> Vec4 {
        self.highlight_color
    }

    pub fn scalar_field(&self) -> Option<&MeshScalarField> {
        self.scalar_field.as_ref()
    }

    pub fn vector_field(&self) -> Option<&MeshVectorField> {
        self.vector_field.as_ref()
    }

    pub fn deformation(&self) -> Option<&MeshDeformation> {
        self.deformation.as_ref()
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn set_mesh_id(&mut self, mesh_id: Option<String>) {
        self.mesh_id = mesh_id;
    }

    pub fn set_vertices(&mut self, vertices: Vec<Vec3>) -> Result<(), String> {
        validate_mesh(&vertices, &self.triangles)?;
        self.vertices = vertices;
        self.clear_invalid_vertex_metadata();
        self.mark_dirty();
        Ok(())
    }

    pub fn set_triangles(&mut self, triangles: Vec<[u32; 3]>) -> Result<(), String> {
        validate_mesh(&self.vertices, &triangles)?;
        self.triangles = triangles;
        self.clear_invalid_triangle_metadata();
        self.mark_dirty();
        Ok(())
    }

    pub fn set_face_color(&mut self, color: Vec4) {
        self.face_color = sanitize_color(color);
        self.mark_dirty();
    }

    pub fn set_edge_color(&mut self, color: Vec4) {
        self.edge_color = sanitize_color(color);
        self.mark_dirty();
    }

    pub fn set_face_alpha(&mut self, alpha: f32) {
        self.face_alpha = sanitize_alpha(alpha);
        self.mark_dirty();
    }

    pub fn set_edge_alpha(&mut self, alpha: f32) {
        self.edge_alpha = sanitize_alpha(alpha);
        self.mark_dirty();
    }

    pub fn set_edge_width(&mut self, edge_width: f32) {
        self.edge_width = if edge_width.is_finite() {
            edge_width.max(0.0)
        } else {
            0.0
        };
        self.mark_dirty();
    }

    pub fn set_edge_mode(&mut self, edge_mode: MeshEdgeMode) {
        self.edge_mode = edge_mode;
        self.mark_dirty();
    }

    pub fn set_feature_edge_groups(&mut self, groups: Option<Vec<u64>>) -> Result<(), String> {
        if let Some(groups) = groups.as_ref() {
            if groups.len() != self.triangles.len() {
                return Err(format!(
                    "mesh feature edge groups: group count must match triangle count ({})",
                    self.triangles.len()
                ));
            }
        }
        self.feature_edge_groups = groups;
        self.mark_dirty();
        Ok(())
    }

    pub fn set_vertex_colors(&mut self, colors: Option<Vec<Vec4>>) -> Result<(), String> {
        if let Some(colors) = colors.as_ref() {
            if colors.len() != self.vertices.len() {
                return Err(format!(
                    "mesh vertex colors: color count must match vertex count ({})",
                    self.vertices.len()
                ));
            }
        }
        self.vertex_colors = colors.map(|colors| colors.into_iter().map(sanitize_color).collect());
        self.mark_dirty();
        Ok(())
    }

    /// Exact per-vertex normals (one per vertex, unit length after normalisation). A rim of a
    /// hole or fillet then shades smoothly where averaged triangle normals would band.
    pub fn set_vertex_normals(&mut self, normals: Option<Vec<Vec3>>) -> Result<(), String> {
        if let Some(normals) = normals.as_ref() {
            if normals.len() != self.vertices.len() {
                return Err(format!(
                    "mesh vertex normals: normal count must match vertex count ({})",
                    self.vertices.len()
                ));
            }
        }
        self.vertex_normals = normals.map(|normals| {
            normals
                .into_iter()
                .map(|n| {
                    if n.is_finite() && n.length_squared() > 1e-20 {
                        n.normalize()
                    } else {
                        Vec3::ZERO
                    }
                })
                .collect()
        });
        self.mark_dirty();
        Ok(())
    }

    pub fn vertex_normals_source(&self) -> Option<&[Vec3]> {
        self.vertex_normals.as_deref()
    }

    pub fn set_triangle_colors(&mut self, colors: Option<Vec<Vec4>>) -> Result<(), String> {
        if let Some(colors) = colors.as_ref() {
            if colors.len() != self.triangles.len() {
                return Err(format!(
                    "mesh triangle colors: color count must match triangle count ({})",
                    self.triangles.len()
                ));
            }
        }
        self.triangle_colors =
            colors.map(|colors| colors.into_iter().map(sanitize_color).collect());
        self.mark_dirty();
        Ok(())
    }

    pub fn set_label(&mut self, label: Option<String>) {
        self.label = label;
    }

    pub fn set_regions(&mut self, regions: Vec<MeshRegion>) {
        self.regions = regions;
        self.mark_dirty();
    }

    pub fn set_highlighted_region_id(&mut self, region_id: Option<String>) {
        self.highlighted_region_id = region_id;
        self.mark_dirty();
    }

    pub fn set_highlight_color(&mut self, color: Vec4) {
        self.highlight_color = sanitize_color(color);
        self.mark_dirty();
    }

    pub fn set_scalar_field(&mut self, field: Option<MeshScalarField>) -> Result<(), String> {
        if let Some(field) = field.as_ref() {
            validate_scalar_field(field, self.vertices.len(), self.triangles.len())?;
        }
        self.scalar_field = field.map(sanitize_scalar_field);
        self.mark_dirty();
        Ok(())
    }

    pub fn set_vector_field(&mut self, field: Option<MeshVectorField>) -> Result<(), String> {
        if let Some(field) = field.as_ref() {
            validate_vector_field(field, self.vertices.len(), self.triangles.len())?;
        }
        self.vector_field = field.map(sanitize_vector_field);
        self.mark_dirty();
        Ok(())
    }

    pub fn set_deformation(&mut self, deformation: Option<MeshDeformation>) -> Result<(), String> {
        if let Some(deformation) = deformation.as_ref() {
            validate_deformation(deformation, self.vertices.len())?;
        }
        self.deformation = deformation.map(sanitize_deformation);
        self.mark_dirty();
        Ok(())
    }

    pub fn region_for_triangle(&self, triangle_index: u32) -> Option<&MeshRegion> {
        self.regions
            .iter()
            .find(|region| region.contains_triangle(triangle_index))
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        self.bounds = None;
        self.face_vertices = None;
        self.face_indices = None;
        self.edge_vertices = None;
        self.vector_vertices = None;
    }

    pub fn effective_face_color(&self) -> Vec4 {
        let mut color = self.face_color;
        color.w *= self.face_alpha.clamp(0.0, 1.0);
        color
    }

    pub fn effective_edge_color(&self) -> Vec4 {
        let mut color = self.edge_color;
        color.w *= self.edge_alpha.clamp(0.0, 1.0);
        color
    }
}

#[allow(unused_imports)]
pub use geometry::*;
#[allow(unused_imports)]
pub use render::*;
