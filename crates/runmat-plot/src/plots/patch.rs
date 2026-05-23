//! MATLAB-compatible polygon patch plot implementation.

use crate::core::{AlphaMode, BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use crate::geometry::stroke3d::{tessellate_polyline, StrokeCap3D, StrokeStyle3D};
use crate::plots::line::LineStyle;
use glam::{Vec3, Vec4};

const POINTS_TO_PX: f32 = 96.0 / 72.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchFaceColorMode {
    Color,
    Flat,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchEdgeColorMode {
    Color,
    None,
}

#[derive(Debug, Clone)]
pub struct PatchPlot {
    vertices: Vec<Vec3>,
    faces: Vec<Vec<usize>>,
    face_color: Vec4,
    edge_color: Vec4,
    face_color_mode: PatchFaceColorMode,
    edge_color_mode: PatchEdgeColorMode,
    face_alpha: f32,
    edge_alpha: f32,
    line_width: f32,
    label: Option<String>,
    visible: bool,
    face_vertices: Option<Vec<Vertex>>,
    face_indices: Option<Vec<u32>>,
    edge_vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    force_3d: bool,
    dirty: bool,
}

impl PatchPlot {
    pub fn new(vertices: Vec<Vec3>, faces: Vec<Vec<usize>>) -> Result<Self, String> {
        if vertices.is_empty() {
            return Err("patch: Vertices must not be empty".to_string());
        }
        validate_finite_vertices(&vertices)?;
        let faces = normalize_faces(faces);
        if faces.is_empty() {
            return Err("patch: Faces must contain at least one polygon".to_string());
        }
        validate_faces(&vertices, &faces)?;
        Ok(Self {
            vertices,
            faces,
            face_color: Vec4::new(0.0, 0.447, 0.741, 1.0),
            edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            face_color_mode: PatchFaceColorMode::Color,
            edge_color_mode: PatchEdgeColorMode::Color,
            face_alpha: 1.0,
            edge_alpha: 1.0,
            line_width: 0.5,
            label: None,
            visible: true,
            face_vertices: None,
            face_indices: None,
            edge_vertices: None,
            bounds: None,
            force_3d: false,
            dirty: true,
        })
    }

    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    pub fn faces(&self) -> &[Vec<usize>] {
        &self.faces
    }

    pub fn face_color(&self) -> Vec4 {
        self.face_color
    }

    pub fn edge_color(&self) -> Vec4 {
        self.edge_color
    }

    pub fn face_color_mode(&self) -> PatchFaceColorMode {
        self.face_color_mode
    }

    pub fn edge_color_mode(&self) -> PatchEdgeColorMode {
        self.edge_color_mode
    }

    pub fn face_alpha(&self) -> f32 {
        self.face_alpha
    }

    pub fn edge_alpha(&self) -> f32 {
        self.edge_alpha
    }

    pub fn line_width(&self) -> f32 {
        self.line_width
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn force_3d(&self) -> bool {
        self.force_3d
    }

    pub fn set_force_3d(&mut self, force_3d: bool) {
        self.force_3d = force_3d;
        self.mark_dirty();
    }

    pub fn set_vertices(&mut self, vertices: Vec<Vec3>) -> Result<(), String> {
        if vertices.is_empty() {
            return Err("patch: Vertices must not be empty".to_string());
        }
        validate_finite_vertices(&vertices)?;
        validate_faces(&vertices, &self.faces)?;
        self.vertices = vertices;
        self.mark_dirty();
        Ok(())
    }

    pub fn set_faces(&mut self, faces: Vec<Vec<usize>>) -> Result<(), String> {
        let faces = normalize_faces(faces);
        if faces.is_empty() {
            return Err("patch: Faces must contain at least one polygon".to_string());
        }
        validate_faces(&self.vertices, &faces)?;
        self.faces = faces;
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

    pub fn set_face_color_mode(&mut self, mode: PatchFaceColorMode) {
        self.face_color_mode = mode;
        self.mark_dirty();
    }

    pub fn set_edge_color_mode(&mut self, mode: PatchEdgeColorMode) {
        self.edge_color_mode = mode;
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

    pub fn set_line_width(&mut self, line_width: f32) {
        self.line_width = line_width.max(0.0);
        self.mark_dirty();
    }

    pub fn set_label(&mut self, label: Option<String>) {
        self.label = label;
        self.mark_dirty();
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
        self.mark_dirty();
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

    fn generate_face_geometry(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
        if self.dirty || self.face_vertices.is_none() || self.face_indices.is_none() {
            let mut out_vertices = Vec::new();
            let mut out_indices = Vec::new();
            if self.face_color_mode != PatchFaceColorMode::None {
                let color = self.effective_face_color();
                for face in &self.faces {
                    if face.len() < 3 {
                        continue;
                    }
                    let base = out_vertices.len() as u32;
                    for &idx in face {
                        out_vertices.push(Vertex::new(self.vertices[idx], color));
                    }
                    for tri in 1..(face.len() - 1) {
                        out_indices.extend_from_slice(&[
                            base,
                            base + tri as u32,
                            base + tri as u32 + 1,
                        ]);
                    }
                }
            }
            self.face_vertices = Some(out_vertices);
            self.face_indices = Some(out_indices);
            self.dirty = false;
        }
        (
            self.face_vertices.as_ref().unwrap(),
            self.face_indices.as_ref().unwrap(),
        )
    }

    fn generate_edge_vertices(&mut self) -> &Vec<Vertex> {
        if self.dirty || self.edge_vertices.is_none() {
            let mut out = Vec::new();
            if self.edge_color_mode != PatchEdgeColorMode::None {
                let color = self.effective_edge_color();
                for face in &self.faces {
                    if face.len() < 2 {
                        continue;
                    }
                    for pos in 0..face.len() {
                        let a = self.vertices[face[pos]];
                        let b = self.vertices[face[(pos + 1) % face.len()]];
                        out.push(Vertex::new(a, color));
                        out.push(Vertex::new(b, color));
                    }
                }
            }
            self.edge_vertices = Some(out);
        }
        self.edge_vertices.as_ref().unwrap()
    }

    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            let points: Vec<Vec3> = self
                .vertices
                .iter()
                .copied()
                .filter(|point| point.is_finite())
                .collect();
            self.bounds = Some(if points.is_empty() {
                BoundingBox::new(Vec3::ZERO, Vec3::ZERO)
            } else {
                BoundingBox::from_points(&points)
            });
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
            indices: Some(indices.clone()),
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
        self.edge_render_data_with_viewport(None)
    }

    pub fn edge_render_data_with_viewport(
        &mut self,
        viewport_px: Option<(u32, u32)>,
    ) -> Option<RenderData> {
        let bounds = self.bounds();
        let line_width = self.line_width.max(0.0);
        if line_width == 0.0 {
            return None;
        }

        let color = self.effective_edge_color();
        let width_px = (line_width.max(0.1) * POINTS_TO_PX).max(0.1);
        if let Some(vp) = viewport_px.filter(|_| width_px > 1.0) {
            let has_3d_content =
                self.force_3d || self.vertices.iter().any(|point| point.z.abs() > 1e-6);
            let data_per_px = if has_3d_content {
                crate::core::data_units_per_px_3d(&bounds, vp)
            } else {
                crate::core::data_units_per_px(&bounds, vp)
            };
            let half_width_data = (width_px * 0.5) * data_per_px;
            let style = StrokeStyle3D::new(half_width_data, LineStyle::Solid, StrokeCap3D::Butt);
            let mut tri_vertices = Vec::new();
            for face in &self.faces {
                if face.len() < 2 {
                    continue;
                }
                let mut polyline = Vec::with_capacity(face.len() + 1);
                for &idx in face {
                    polyline.push(self.vertices[idx]);
                }
                polyline.push(self.vertices[face[0]]);
                tri_vertices.extend(tessellate_polyline(&polyline, color, style));
            }
            if !tri_vertices.is_empty() {
                let indices = (0..tri_vertices.len() as u32).collect::<Vec<u32>>();
                let index_count = indices.len();
                let vertex_count = tri_vertices.len();
                return Some(RenderData {
                    pipeline_type: PipelineType::Triangles,
                    vertices: tri_vertices,
                    indices: Some(indices),
                    gpu_vertices: None,
                    bounds: Some(bounds),
                    material: Material {
                        albedo: color,
                        roughness: width_px.max(0.5),
                        alpha_mode: if color.w < 1.0 {
                            AlphaMode::Blend
                        } else {
                            AlphaMode::Opaque
                        },
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
                });
            }
        }

        let vertices = self.generate_edge_vertices().clone();
        if vertices.is_empty() {
            return None;
        }
        Some(RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: Some(bounds),
            material: Material {
                albedo: color,
                roughness: width_px.max(0.5),
                alpha_mode: if color.w < 1.0 {
                    AlphaMode::Blend
                } else {
                    AlphaMode::Opaque
                },
                ..Default::default()
            },
            draw_calls: vec![DrawCall {
                vertex_offset: 0,
                vertex_count: self.edge_vertices.as_ref().map(|v| v.len()).unwrap_or(0),
                index_offset: None,
                index_count: None,
                instance_count: 1,
            }],
            image: None,
        })
    }

    pub fn estimated_memory_usage(&self) -> usize {
        self.face_vertices
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>())
            + self
                .face_indices
                .as_ref()
                .map_or(0, |i| i.len() * std::mem::size_of::<u32>())
            + self
                .edge_vertices
                .as_ref()
                .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>())
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

fn validate_finite_vertices(vertices: &[Vec3]) -> Result<(), String> {
    if vertices
        .iter()
        .any(|v| !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite())
    {
        return Err(
            "patch: Vertices must contain finite Vec3 coordinates before bounds/render_data"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_faces(vertices: &[Vec3], faces: &[Vec<usize>]) -> Result<(), String> {
    for face in faces {
        for &idx in face {
            if idx >= vertices.len() {
                return Err("patch: Faces index exceeds Vertices row count".to_string());
            }
        }
    }
    Ok(())
}

fn normalize_faces(faces: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    faces
        .into_iter()
        .filter_map(|mut face| {
            face.dedup();
            if face.len() > 1 && face.first() == face.last() {
                face.pop();
            }
            if face.len() >= 3 {
                Some(face)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_triangulates_quad_and_closes_edges() {
        let mut patch = PatchPlot::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .unwrap();
        let face = patch.render_data();
        assert_eq!(face.indices.as_ref().unwrap().len(), 6);
        let edge = patch.edge_render_data().unwrap();
        assert_eq!(edge.vertices.len(), 8);
    }

    #[test]
    fn patch_set_face_color_invalidates_cached_geometry() {
        let mut patch = PatchPlot::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        let initial = patch.render_data();
        assert_eq!(initial.vertices[0].color, [0.0, 0.447, 0.741, 1.0]);

        patch.set_face_color(Vec4::new(1.0, 0.0, 0.0, 1.0));
        let updated = patch.render_data();
        assert_eq!(updated.vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn patch_new_rejects_non_finite_vertices_before_render_data() {
        let err = PatchPlot::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(f32::NAN, 0.0, 0.0),
                Vec3::new(0.0, 1.0, f32::INFINITY),
            ],
            vec![vec![0, 1, 2]],
        )
        .expect_err("PatchPlot::new should reject non-finite Vec3 coordinates");
        assert!(err.contains("finite Vec3 coordinates"));
    }

    #[test]
    fn patch_style_setters_sanitize_non_finite_values() {
        let mut patch = PatchPlot::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )
        .unwrap();

        patch.set_face_color(Vec4::new(f32::NAN, 0.25, f32::INFINITY, 1.0));
        patch.set_edge_color(Vec4::new(0.5, f32::NEG_INFINITY, 0.75, f32::NAN));
        patch.set_face_alpha(f32::NAN);
        patch.set_edge_alpha(f32::INFINITY);

        assert_eq!(patch.face_color(), Vec4::new(0.0, 0.25, 0.0, 1.0));
        assert_eq!(patch.edge_color(), Vec4::new(0.5, 0.0, 0.75, 0.0));
        assert_eq!(patch.face_alpha(), 1.0);
        assert_eq!(patch.edge_alpha(), 1.0);

        let render = patch.render_data();
        assert!(render.vertices[0]
            .color
            .iter()
            .all(|component| component.is_finite()));
        assert!(render.material.albedo.is_finite());
    }

    #[test]
    fn patch_accepts_explicitly_closed_face() {
        let patch = PatchPlot::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2, 0]],
        )
        .unwrap();
        assert_eq!(patch.faces(), &[vec![0, 1, 2]]);
    }
}
