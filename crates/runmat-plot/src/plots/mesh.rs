//! Indexed triangle mesh plot primitive.

use crate::core::{AlphaMode, BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};
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

    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            let vertices = self.effective_vertices();
            self.bounds = Some(BoundingBox::from_points(&vertices));
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
        if self.edge_width <= 0.0
            || self.edge_alpha <= 0.0
            || matches!(self.edge_mode, MeshEdgeMode::None)
        {
            return None;
        }
        if self.dirty || self.edge_vertices.is_none() {
            let color = self.effective_edge_color();
            let vertices_source = self.effective_vertices();
            let edges = self.renderable_edges();
            let mut vertices = Vec::with_capacity(edges.len() * 2);
            for (a, b) in edges {
                vertices.push(Vertex::new(vertices_source[a as usize], color));
                vertices.push(Vertex::new(vertices_source[b as usize], color));
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

    pub fn vector_render_data(&mut self) -> Option<RenderData> {
        let bounds = self.bounds();
        if self.dirty || self.vector_vertices.is_none() {
            let vertices_source = self.effective_vertices();
            let field = self.vector_field.as_ref()?;
            let color = sanitize_color(field.color);
            let stride = field.stride.max(1);
            let mut vertices = Vec::new();
            match field.location {
                MeshFieldLocation::Vertex => {
                    for (index, vector) in field.vectors.iter().enumerate().step_by(stride) {
                        if vector.length_squared() <= f32::EPSILON {
                            continue;
                        }
                        let start = vertices_source[index];
                        vertices.push(Vertex::new(start, color));
                        vertices.push(Vertex::new(start + *vector * field.scale, color));
                    }
                }
                MeshFieldLocation::Triangle => {
                    for (index, vector) in field.vectors.iter().enumerate().step_by(stride) {
                        if vector.length_squared() <= f32::EPSILON {
                            continue;
                        }
                        let triangle = self.triangles[index];
                        let start = (vertices_source[triangle[0] as usize]
                            + vertices_source[triangle[1] as usize]
                            + vertices_source[triangle[2] as usize])
                            / 3.0;
                        vertices.push(Vertex::new(start, color));
                        vertices.push(Vertex::new(start + *vector * field.scale, color));
                    }
                }
            }
            self.vector_vertices = Some(vertices);
        }
        let vertices = self.vector_vertices.as_ref()?.clone();
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
                albedo: self
                    .vector_field
                    .as_ref()
                    .map(|field| sanitize_color(field.color))
                    .unwrap_or(Vec4::ONE),
                alpha_mode: if self
                    .vector_field
                    .as_ref()
                    .map(|field| field.color.w < 1.0)
                    .unwrap_or(false)
                {
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
            + self
                .feature_edge_groups
                .as_ref()
                .map_or(0, |groups| groups.len() * std::mem::size_of::<u64>())
            + self
                .vertex_colors
                .as_ref()
                .map_or(0, |colors| colors.len() * std::mem::size_of::<Vec4>())
            + self
                .triangle_colors
                .as_ref()
                .map_or(0, |colors| colors.len() * std::mem::size_of::<Vec4>())
            + self
                .vector_vertices
                .as_ref()
                .map_or(0, |vertices| vertices.len() * std::mem::size_of::<Vertex>())
            + self
                .scalar_field
                .as_ref()
                .map_or(0, |field| field.values.len() * std::mem::size_of::<f32>())
            + self
                .vector_field
                .as_ref()
                .map_or(0, |field| field.vectors.len() * std::mem::size_of::<Vec3>())
            + self.deformation.as_ref().map_or(0, |field| {
                field.displacements.len() * std::mem::size_of::<Vec3>()
            })
    }

    fn generate_face_geometry(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
        if self.dirty || self.face_vertices.is_none() || self.face_indices.is_none() {
            let color = self.effective_face_color();
            let vertices_source = self.effective_vertices();
            let normals = vertex_normals(&vertices_source, &self.triangles);
            let scalar_limits = self.scalar_field.as_ref().and_then(scalar_limits);
            let (vertices, indices) = if self.highlighted_region_id.is_some()
                || self
                    .scalar_field
                    .as_ref()
                    .is_some_and(|field| field.location == MeshFieldLocation::Triangle)
                || self.triangle_colors.is_some()
            {
                let highlight_color = self.highlight_color;
                let highlighted_region_id = self.highlighted_region_id.as_deref();
                let highlighted_region = self
                    .regions
                    .iter()
                    .find(|region| Some(region.region_id.as_str()) == highlighted_region_id);
                let mut vertices = Vec::with_capacity(self.triangles.len() * 3);
                let mut indices = Vec::with_capacity(self.triangles.len() * 3);
                for (triangle_index, triangle) in self.triangles.iter().enumerate() {
                    let region_highlighted = if highlighted_region
                        .is_some_and(|region| region.contains_triangle(triangle_index as u32))
                    {
                        Some(highlight_color)
                    } else {
                        None
                    };
                    for vertex_id in triangle {
                        let source_index = *vertex_id as usize;
                        let target_index = vertices.len() as u32;
                        let vertex_color = region_highlighted.unwrap_or_else(|| {
                            self.scalar_color(source_index, triangle_index, scalar_limits)
                                .or_else(|| self.triangle_color(triangle_index))
                                .or_else(|| self.vertex_color(source_index))
                                .unwrap_or(color)
                        });
                        vertices.push(Vertex {
                            position: vertices_source[source_index].to_array(),
                            color: vertex_color.to_array(),
                            normal: normals[source_index].to_array(),
                            tex_coords: [0.0, 0.0],
                        });
                        indices.push(target_index);
                    }
                }
                (vertices, indices)
            } else {
                let vertices = vertices_source
                    .iter()
                    .zip(normals.iter())
                    .enumerate()
                    .map(|(source_index, (&position, &normal))| {
                        let vertex_color = self
                            .scalar_color(source_index, 0, scalar_limits)
                            .or_else(|| self.vertex_color(source_index))
                            .unwrap_or(color);
                        Vertex {
                            position: position.to_array(),
                            color: vertex_color.to_array(),
                            normal: normal.to_array(),
                            tex_coords: [0.0, 0.0],
                        }
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

    fn effective_vertices(&self) -> Cow<'_, [Vec3]> {
        if let Some(deformation) = self.deformation.as_ref() {
            Cow::Owned(
                self.vertices
                    .iter()
                    .zip(deformation.displacements.iter())
                    .map(|(&position, &displacement)| position + displacement * deformation.scale)
                    .collect(),
            )
        } else {
            Cow::Borrowed(&self.vertices)
        }
    }

    fn scalar_color(
        &self,
        vertex_index: usize,
        triangle_index: usize,
        limits: Option<[f32; 2]>,
    ) -> Option<Vec4> {
        let field = self.scalar_field.as_ref()?;
        let value = match field.location {
            MeshFieldLocation::Vertex => *field.values.get(vertex_index)?,
            MeshFieldLocation::Triangle => *field.values.get(triangle_index)?,
        };
        if !value.is_finite() {
            return None;
        }
        let [min, max] = limits?;
        let t = if max > min {
            ((value - min) / (max - min)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        Some(colormap_color(
            &field.colormap,
            t,
            field.alpha * self.face_alpha,
        ))
    }

    fn triangle_color(&self, triangle_index: usize) -> Option<Vec4> {
        let mut color = *self.triangle_colors.as_ref()?.get(triangle_index)?;
        color.w *= self.face_alpha.clamp(0.0, 1.0);
        Some(color)
    }

    fn vertex_color(&self, vertex_index: usize) -> Option<Vec4> {
        let mut color = *self.vertex_colors.as_ref()?.get(vertex_index)?;
        color.w *= self.face_alpha.clamp(0.0, 1.0);
        Some(color)
    }

    fn renderable_edges(&self) -> Vec<(u32, u32)> {
        match self.edge_mode {
            MeshEdgeMode::All => self.all_triangle_edges(),
            MeshEdgeMode::Feature => self.feature_edges(),
            MeshEdgeMode::None => Vec::new(),
        }
    }

    fn all_triangle_edges(&self) -> Vec<(u32, u32)> {
        let mut edges = BTreeSet::<(u32, u32)>::new();
        for &[a, b, c] in &self.triangles {
            edges.insert(normalized_edge(a, b));
            edges.insert(normalized_edge(b, c));
            edges.insert(normalized_edge(c, a));
        }
        edges.into_iter().collect()
    }

    fn feature_edges(&self) -> Vec<(u32, u32)> {
        let groups = self.feature_edge_groups.as_deref();
        let mut edges = BTreeMap::<(u32, u32), FeatureEdgeAccumulator>::new();
        for (triangle_index, &[a, b, c]) in self.triangles.iter().enumerate() {
            let group = groups
                .and_then(|groups| groups.get(triangle_index).copied())
                .unwrap_or(0);
            accumulate_feature_edge(&mut edges, a, b, group);
            accumulate_feature_edge(&mut edges, b, c, group);
            accumulate_feature_edge(&mut edges, c, a, group);
        }
        edges
            .into_iter()
            .filter_map(|(edge, accumulator)| {
                (accumulator.count != 2 || accumulator.crosses_group).then_some(edge)
            })
            .collect()
    }

    fn clear_invalid_triangle_metadata(&mut self) {
        if self
            .feature_edge_groups
            .as_ref()
            .is_some_and(|groups| groups.len() != self.triangles.len())
        {
            self.feature_edge_groups = None;
        }
        if self
            .triangle_colors
            .as_ref()
            .is_some_and(|colors| colors.len() != self.triangles.len())
        {
            self.triangle_colors = None;
        }
    }

    fn clear_invalid_vertex_metadata(&mut self) {
        if self
            .vertex_colors
            .as_ref()
            .is_some_and(|colors| colors.len() != self.vertices.len())
        {
            self.vertex_colors = None;
        }
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

fn validate_scalar_field(
    field: &MeshScalarField,
    vertex_count: usize,
    triangle_count: usize,
) -> Result<(), String> {
    if field.field_id.trim().is_empty() {
        return Err("mesh scalar field: field_id must not be empty".to_string());
    }
    validate_field_len(
        "mesh scalar field",
        field.location,
        field.values.len(),
        vertex_count,
        triangle_count,
    )?;
    if field
        .color_limits
        .is_some_and(|[min, max]| !min.is_finite() || !max.is_finite() || max < min)
    {
        return Err("mesh scalar field: color_limits must be finite and ordered".to_string());
    }
    Ok(())
}

fn validate_vector_field(
    field: &MeshVectorField,
    vertex_count: usize,
    triangle_count: usize,
) -> Result<(), String> {
    if field.field_id.trim().is_empty() {
        return Err("mesh vector field: field_id must not be empty".to_string());
    }
    validate_field_len(
        "mesh vector field",
        field.location,
        field.vectors.len(),
        vertex_count,
        triangle_count,
    )?;
    if field
        .vectors
        .iter()
        .any(|vector| !vector.x.is_finite() || !vector.y.is_finite() || !vector.z.is_finite())
    {
        return Err("mesh vector field: vectors must contain finite components".to_string());
    }
    Ok(())
}

fn validate_deformation(deformation: &MeshDeformation, vertex_count: usize) -> Result<(), String> {
    if deformation.field_id.trim().is_empty() {
        return Err("mesh deformation: field_id must not be empty".to_string());
    }
    if deformation.displacements.len() != vertex_count {
        return Err("mesh deformation: displacements must match vertex count".to_string());
    }
    if deformation.displacements.iter().any(|displacement| {
        !displacement.x.is_finite() || !displacement.y.is_finite() || !displacement.z.is_finite()
    }) {
        return Err("mesh deformation: displacements must contain finite components".to_string());
    }
    Ok(())
}

fn validate_field_len(
    label: &str,
    location: MeshFieldLocation,
    actual: usize,
    vertex_count: usize,
    triangle_count: usize,
) -> Result<(), String> {
    let expected = match location {
        MeshFieldLocation::Vertex => vertex_count,
        MeshFieldLocation::Triangle => triangle_count,
    };
    if actual != expected {
        return Err(format!(
            "{label}: value count must match {} count ({expected})",
            location.as_str()
        ));
    }
    Ok(())
}

fn sanitize_scalar_field(mut field: MeshScalarField) -> MeshScalarField {
    field.alpha = sanitize_alpha(field.alpha);
    if field.colormap.trim().is_empty() {
        field.colormap = "viridis".to_string();
    }
    field
}

fn sanitize_vector_field(mut field: MeshVectorField) -> MeshVectorField {
    field.scale = if field.scale.is_finite() {
        field.scale
    } else {
        1.0
    };
    field.stride = field.stride.max(1);
    field.color = sanitize_color(field.color);
    field
}

fn sanitize_deformation(mut deformation: MeshDeformation) -> MeshDeformation {
    deformation.scale = if deformation.scale.is_finite() {
        deformation.scale
    } else {
        1.0
    };
    deformation
}

fn scalar_limits(field: &MeshScalarField) -> Option<[f32; 2]> {
    if let Some([min, max]) = field.color_limits {
        if min.is_finite() && max.is_finite() && max >= min {
            return Some([min, max]);
        }
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in field
        .values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
    {
        min = min.min(value);
        max = max.max(value);
    }
    if min.is_finite() && max.is_finite() {
        Some([min, max])
    } else {
        None
    }
}

fn colormap_color(name: &str, t: f32, alpha: f32) -> Vec4 {
    let t = t.clamp(0.0, 1.0);
    let alpha = sanitize_alpha(alpha);
    match name {
        "thermal" | "heat" => Vec4::new(t, 0.22 + 0.5 * t, 1.0 - t, alpha),
        "blue_red" | "blue-red" | "diverging" => {
            if t < 0.5 {
                let local = t * 2.0;
                Vec4::new(0.14 + 0.56 * local, 0.34 + 0.36 * local, 0.86, alpha)
            } else {
                let local = (t - 0.5) * 2.0;
                Vec4::new(
                    0.70 + 0.25 * local,
                    0.70 - 0.46 * local,
                    0.86 - 0.66 * local,
                    alpha,
                )
            }
        }
        _ => {
            let r = (0.28 + 0.65 * t).clamp(0.0, 1.0);
            let g = (0.08 + 0.85 * (1.0 - (t - 0.5).abs() * 2.0)).clamp(0.0, 1.0);
            let b = (0.55 + 0.35 * (1.0 - t)).clamp(0.0, 1.0);
            Vec4::new(r, g, b, alpha)
        }
    }
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

#[derive(Debug, Clone, Copy)]
struct FeatureEdgeAccumulator {
    first_group: u64,
    count: u8,
    crosses_group: bool,
}

fn accumulate_feature_edge(
    edges: &mut BTreeMap<(u32, u32), FeatureEdgeAccumulator>,
    a: u32,
    b: u32,
    group: u64,
) {
    let entry = edges
        .entry(normalized_edge(a, b))
        .or_insert(FeatureEdgeAccumulator {
            first_group: group,
            count: 0,
            crosses_group: false,
        });
    if entry.first_group != group {
        entry.crosses_group = true;
    }
    entry.count = entry.count.saturating_add(1);
}

fn normalized_edge(a: u32, b: u32) -> (u32, u32) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_mesh() -> MeshPlot {
        MeshPlot::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
        .expect("mesh should be valid")
    }

    fn square_mesh() -> MeshPlot {
        MeshPlot::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
        .expect("mesh should be valid")
    }

    #[test]
    fn scalar_field_colors_mesh_vertices() {
        let mut mesh = triangle_mesh();
        let mut field = MeshScalarField::new(
            "fea.structural.von_mises",
            MeshFieldLocation::Vertex,
            vec![0.0, 0.5, 1.0],
        );
        field.color_limits = Some([0.0, 1.0]);
        field.alpha = 0.75;
        mesh.set_scalar_field(Some(field))
            .expect("scalar field should be accepted");

        let render = mesh.render_data();
        assert_eq!(render.vertices.len(), 3);
        assert_ne!(render.vertices[0].color, render.vertices[2].color);
        assert!((render.vertices[0].color[3] - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn deformation_updates_render_positions_and_bounds() {
        let mut mesh = triangle_mesh();
        let mut deformation = MeshDeformation::new(
            "fea.structural.displacement",
            vec![Vec3::ZERO, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO],
        );
        deformation.scale = 0.5;
        mesh.set_deformation(Some(deformation))
            .expect("deformation should be accepted");

        let bounds = mesh.bounds();
        assert_eq!(bounds.max.z, 1.0);
        let render = mesh.render_data();
        assert!(render
            .vertices
            .iter()
            .any(|vertex| (vertex.position[2] - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn vector_field_generates_line_glyphs() {
        let mut mesh = triangle_mesh();
        let mut field = MeshVectorField::new(
            "fea.em.flux_density",
            MeshFieldLocation::Vertex,
            vec![Vec3::X, Vec3::ZERO, Vec3::Y],
        );
        field.scale = 0.25;
        mesh.set_vector_field(Some(field))
            .expect("vector field should be accepted");

        let render = mesh.vector_render_data().expect("vector glyphs");
        assert_eq!(render.pipeline_type, PipelineType::Lines);
        assert_eq!(render.vertices.len(), 4);
    }

    #[test]
    fn feature_edge_mode_suppresses_internal_edges_with_same_group() {
        let mut mesh = square_mesh();
        mesh.set_edge_mode(MeshEdgeMode::Feature);
        mesh.set_feature_edge_groups(Some(vec![7, 7]))
            .expect("feature groups should be accepted");

        let render = mesh.edge_render_data().expect("feature edges");

        assert_eq!(render.pipeline_type, PipelineType::Lines);
        assert_eq!(render.vertices.len(), 8);
    }

    #[test]
    fn feature_edge_mode_keeps_edges_between_groups() {
        let mut mesh = square_mesh();
        mesh.set_edge_mode(MeshEdgeMode::Feature);
        mesh.set_feature_edge_groups(Some(vec![7, 9]))
            .expect("feature groups should be accepted");

        let render = mesh.edge_render_data().expect("feature edges");

        assert_eq!(render.pipeline_type, PipelineType::Lines);
        assert_eq!(render.vertices.len(), 10);
    }

    #[test]
    fn triangle_colors_generate_per_face_vertices() {
        let mut mesh = square_mesh();
        mesh.set_triangle_colors(Some(vec![
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 0.0, 1.0, 1.0),
        ]))
        .expect("triangle colors should be accepted");

        let render = mesh.render_data();

        assert_eq!(render.vertices.len(), 6);
        assert_eq!(render.vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(render.vertices[3].color, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn vertex_colors_preserve_indexed_mesh_geometry() {
        let mut mesh = square_mesh();
        mesh.set_vertex_colors(Some(vec![
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
            Vec4::new(0.0, 0.0, 1.0, 1.0),
            Vec4::new(1.0, 1.0, 0.0, 1.0),
        ]))
        .expect("vertex colors should be accepted");

        let render = mesh.render_data();

        assert_eq!(render.vertices.len(), 4);
        assert_eq!(render.indices.as_ref().unwrap().len(), 6);
        assert_eq!(render.vertices[2].color, [0.0, 0.0, 1.0, 1.0]);
    }
}
