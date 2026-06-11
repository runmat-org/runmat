//! Indexed triangle mesh plot primitive.

use crate::core::{AlphaMode, BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};
use std::collections::HashSet;

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
    label: Option<String>,
    regions: Vec<MeshRegion>,
    highlighted_region_id: Option<String>,
    highlight_color: Vec4,
    visible: bool,
    bounds: Option<BoundingBox>,
    face_vertices: Option<Vec<Vertex>>,
    face_indices: Option<Vec<u32>>,
    edge_vertices: Option<Vec<Vertex>>,
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
            label: None,
            regions: Vec::new(),
            highlighted_region_id: None,
            highlight_color: Vec4::new(0.98, 0.78, 0.22, 1.0),
            visible: true,
            bounds: None,
            face_vertices: None,
            face_indices: None,
            edge_vertices: None,
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

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn set_mesh_id(&mut self, mesh_id: Option<String>) {
        self.mesh_id = mesh_id;
    }

    pub fn set_vertices(&mut self, vertices: Vec<Vec3>) -> Result<(), String> {
        validate_mesh(&vertices, &self.triangles)?;
        self.vertices = vertices;
        self.mark_dirty();
        Ok(())
    }

    pub fn set_triangles(&mut self, triangles: Vec<[u32; 3]>) -> Result<(), String> {
        validate_mesh(&self.vertices, &triangles)?;
        self.triangles = triangles;
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

    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            self.bounds = Some(BoundingBox::from_points(&self.vertices));
        }
        self.bounds.unwrap()
    }

    pub fn render_data(&mut self) -> RenderData {
        let bounds = self.bounds();
        let (vertices, indices) = {
            let (vertices, indices) = self.generate_face_geometry();
            (vertices.clone(), indices.clone())
        };
        let color = self.effective_face_color();
        let vertex_count = vertices.len();
        let index_count = indices.len();
        RenderData {
            pipeline_type: PipelineType::Triangles,
            vertices,
            indices: Some(indices),
            gpu_vertices: None,
            bounds: Some(bounds),
            material: Material {
                albedo: color,
                alpha_mode: if color.w < 1.0 {
                    AlphaMode::Blend
                } else {
                    AlphaMode::Opaque
                },
                double_sided: true,
                ..Default::default()
            },
            draw_calls: vec![DrawCall {
                vertex_offset: 0,
                vertex_count,
                index_offset: Some(0),
                index_count: Some(index_count),
                instance_count: 1,
            }],
            image: None,
        }
    }

    pub fn edge_render_data(&mut self) -> Option<RenderData> {
        let bounds = self.bounds();
        if self.edge_width <= 0.0 || self.edge_alpha <= 0.0 {
            return None;
        }
        if self.dirty || self.edge_vertices.is_none() {
            let color = self.effective_edge_color();
            let mut edges = HashSet::<(u32, u32)>::new();
            for &[a, b, c] in &self.triangles {
                insert_edge(&mut edges, a, b);
                insert_edge(&mut edges, b, c);
                insert_edge(&mut edges, c, a);
            }
            let mut vertices = Vec::with_capacity(edges.len() * 2);
            for (a, b) in edges {
                vertices.push(Vertex::new(self.vertices[a as usize], color));
                vertices.push(Vertex::new(self.vertices[b as usize], color));
            }
            self.edge_vertices = Some(vertices);
        }
        let vertices = self.edge_vertices.as_ref()?.clone();
        if vertices.is_empty() {
            return None;
        }
        let vertex_count = vertices.len();
        Some(RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: Some(bounds),
            material: Material {
                albedo: self.effective_edge_color(),
                roughness: self.edge_width.max(0.1),
                alpha_mode: if self.edge_alpha < 1.0 {
                    AlphaMode::Blend
                } else {
                    AlphaMode::Opaque
                },
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
        })
    }

    pub fn estimated_memory_usage(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<Vec3>()
            + self.triangles.len() * std::mem::size_of::<[u32; 3]>()
            + self
                .face_vertices
                .as_ref()
                .map_or(0, |vertices| vertices.len() * std::mem::size_of::<Vertex>())
            + self
                .face_indices
                .as_ref()
                .map_or(0, |indices| indices.len() * std::mem::size_of::<u32>())
            + self
                .edge_vertices
                .as_ref()
                .map_or(0, |vertices| vertices.len() * std::mem::size_of::<Vertex>())
    }

    fn generate_face_geometry(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
        if self.dirty || self.face_vertices.is_none() || self.face_indices.is_none() {
            let color = self.effective_face_color();
            let normals = vertex_normals(&self.vertices, &self.triangles);
            let (vertices, indices) =
                if let Some(highlighted_region_id) = self.highlighted_region_id.as_deref() {
                    let highlight_color = self.highlight_color;
                    let highlighted_region = self
                        .regions
                        .iter()
                        .find(|region| region.region_id == highlighted_region_id);
                    let mut vertices = Vec::with_capacity(self.triangles.len() * 3);
                    let mut indices = Vec::with_capacity(self.triangles.len() * 3);
                    for (triangle_index, triangle) in self.triangles.iter().enumerate() {
                        let triangle_color = if highlighted_region
                            .is_some_and(|region| region.contains_triangle(triangle_index as u32))
                        {
                            highlight_color
                        } else {
                            color
                        };
                        for vertex_id in triangle {
                            let source_index = *vertex_id as usize;
                            let target_index = vertices.len() as u32;
                            vertices.push(Vertex {
                                position: self.vertices[source_index].to_array(),
                                color: triangle_color.to_array(),
                                normal: normals[source_index].to_array(),
                                tex_coords: [0.0, 0.0],
                            });
                            indices.push(target_index);
                        }
                    }
                    (vertices, indices)
                } else {
                    let vertices = self
                        .vertices
                        .iter()
                        .zip(normals.iter())
                        .map(|(&position, &normal)| Vertex {
                            position: position.to_array(),
                            color: color.to_array(),
                            normal: normal.to_array(),
                            tex_coords: [0.0, 0.0],
                        })
                        .collect();
                    let indices = self
                        .triangles
                        .iter()
                        .flat_map(|triangle| triangle.iter().copied())
                        .collect();
                    (vertices, indices)
                };
            self.face_vertices = Some(vertices);
            self.face_indices = Some(indices);
            self.dirty = false;
        }
        (
            self.face_vertices.as_ref().unwrap(),
            self.face_indices.as_ref().unwrap(),
        )
    }
}

fn validate_mesh(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Result<(), String> {
    if vertices.is_empty() {
        return Err("mesh: vertices must not be empty".to_string());
    }
    if triangles.is_empty() {
        return Err("mesh: triangles must not be empty".to_string());
    }
    if vertices
        .iter()
        .any(|vertex| !vertex.x.is_finite() || !vertex.y.is_finite() || !vertex.z.is_finite())
    {
        return Err("mesh: vertices must contain finite coordinates".to_string());
    }
    let vertex_count = vertices.len();
    for triangle in triangles {
        if triangle.iter().any(|index| *index as usize >= vertex_count) {
            return Err("mesh: triangle index exceeds vertex count".to_string());
        }
    }
    Ok(())
}

fn vertex_normals(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; vertices.len()];
    for &[a, b, c] in triangles {
        let ia = a as usize;
        let ib = b as usize;
        let ic = c as usize;
        let edge_ab = vertices[ib] - vertices[ia];
        let edge_ac = vertices[ic] - vertices[ia];
        let normal = edge_ab.cross(edge_ac);
        if normal.length_squared() > f32::EPSILON {
            normals[ia] += normal;
            normals[ib] += normal;
            normals[ic] += normal;
        }
    }
    normals
        .into_iter()
        .map(|normal| {
            if normal.length_squared() > f32::EPSILON {
                normal.normalize()
            } else {
                Vec3::Z
            }
        })
        .collect()
}

fn insert_edge(edges: &mut HashSet<(u32, u32)>, a: u32, b: u32) {
    edges.insert(if a <= b { (a, b) } else { (b, a) });
}

fn sanitize_color(color: Vec4) -> Vec4 {
    Vec4::new(
        sanitize_color_component(color.x),
        sanitize_color_component(color.y),
        sanitize_color_component(color.z),
        sanitize_color_component(color.w),
    )
}

fn sanitize_color_component(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn sanitize_alpha(alpha: f32) -> f32 {
    if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0
    }
}
