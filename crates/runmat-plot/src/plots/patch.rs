//! MATLAB-compatible polygon patch plot implementation.

use crate::core::{AlphaMode, BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};

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
    pub vertices: Vec<Vec3>,
    pub faces: Vec<Vec<usize>>,
    pub face_color: Vec4,
    pub edge_color: Vec4,
    pub face_color_mode: PatchFaceColorMode,
    pub edge_color_mode: PatchEdgeColorMode,
    pub face_alpha: f32,
    pub edge_alpha: f32,
    pub line_width: f32,
    pub label: Option<String>,
    pub visible: bool,
    face_vertices: Option<Vec<Vertex>>,
    face_indices: Option<Vec<u32>>,
    edge_vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

impl PatchPlot {
    pub fn new(vertices: Vec<Vec3>, faces: Vec<Vec<usize>>) -> Result<Self, String> {
        if vertices.is_empty() {
            return Err("patch: Vertices must not be empty".to_string());
        }
        let faces = normalize_faces(faces);
        if faces.is_empty() {
            return Err("patch: Faces must contain at least one polygon".to_string());
        }
        for face in &faces {
            for &idx in face {
                if idx >= vertices.len() {
                    return Err("patch: Faces index exceeds Vertices row count".to_string());
                }
            }
        }
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
            dirty: true,
        })
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
        let bounds = self.bounds();
        let line_width = self.line_width.max(0.0);
        let vertices = self.generate_edge_vertices().clone();
        if vertices.is_empty() || line_width == 0.0 {
            return None;
        }
        let color = self.effective_edge_color();
        Some(RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: Some(bounds),
            material: Material {
                albedo: color,
                roughness: line_width.max(0.5),
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
        assert_eq!(patch.faces, vec![vec![0, 1, 2]]);
    }
}
