//! A mesh plot's bounds, render data (faces, edges, vectors) and memory accounting.

use super::*;

impl MeshPlot {
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
                        if !vector.is_finite() || *vector == Vec3::ZERO {
                            continue;
                        }
                        let start = vertices_source[index];
                        let end = start + *vector * field.scale;
                        if !start.is_finite() || !end.is_finite() {
                            continue;
                        }
                        vertices.push(Vertex::new(start, color));
                        vertices.push(Vertex::new(end, color));
                    }
                }
                MeshFieldLocation::Triangle => {
                    for (index, vector) in field.vectors.iter().enumerate().step_by(stride) {
                        if !vector.is_finite() || *vector == Vec3::ZERO {
                            continue;
                        }
                        let triangle = self.triangles[index];
                        let start = (vertices_source[triangle[0] as usize]
                            + vertices_source[triangle[1] as usize]
                            + vertices_source[triangle[2] as usize])
                            / 3.0;
                        let end = start + *vector * field.scale;
                        if !start.is_finite() || !end.is_finite() {
                            continue;
                        }
                        vertices.push(Vertex::new(start, color));
                        vertices.push(Vertex::new(end, color));
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

    pub(crate) fn retained_snapshot_bytes(&self) -> Option<usize> {
        let mut total = 0usize;
        for value in [
            self.mesh_id.as_ref(),
            self.label.as_ref(),
            self.highlighted_region_id.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            crate::plots::clone_budget::add(&mut total, value.capacity())?;
        }
        for bytes in [
            crate::plots::clone_budget::vec::<Vec3>(self.vertices.capacity())?,
            crate::plots::clone_budget::vec::<[u32; 3]>(self.triangles.capacity())?,
            crate::plots::clone_budget::vec::<u64>(
                self.feature_edge_groups.as_ref().map_or(0, Vec::capacity),
            )?,
            crate::plots::clone_budget::vec::<Vec4>(
                self.vertex_colors.as_ref().map_or(0, Vec::capacity),
            )?,
            crate::plots::clone_budget::vec::<Vec4>(
                self.triangle_colors.as_ref().map_or(0, Vec::capacity),
            )?,
            crate::plots::clone_budget::vec::<Vertex>(
                self.face_vertices.as_ref().map_or(0, Vec::capacity),
            )?,
            crate::plots::clone_budget::vec::<u32>(
                self.face_indices.as_ref().map_or(0, Vec::capacity),
            )?,
            crate::plots::clone_budget::vec::<Vertex>(
                self.edge_vertices.as_ref().map_or(0, Vec::capacity),
            )?,
            crate::plots::clone_budget::vec::<Vertex>(
                self.vector_vertices.as_ref().map_or(0, Vec::capacity),
            )?,
            crate::plots::clone_budget::vec::<MeshRegion>(self.regions.capacity())?,
        ] {
            crate::plots::clone_budget::add(&mut total, bytes)?;
        }
        for region in &self.regions {
            for value in [
                Some(&region.region_id),
                region.label.as_ref(),
                region.tag.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                crate::plots::clone_budget::add(&mut total, value.capacity())?;
            }
            crate::plots::clone_budget::add(
                &mut total,
                crate::plots::clone_budget::vec::<MeshTriangleRange>(
                    region.triangle_ranges.capacity(),
                )?,
            )?;
        }
        if let Some(field) = &self.scalar_field {
            for value in [
                Some(&field.field_id),
                field.label.as_ref(),
                Some(&field.colormap),
            ]
            .into_iter()
            .flatten()
            {
                crate::plots::clone_budget::add(&mut total, value.capacity())?;
            }
            crate::plots::clone_budget::add(
                &mut total,
                crate::plots::clone_budget::vec::<f32>(field.values.capacity())?,
            )?;
        }
        if let Some(field) = &self.vector_field {
            crate::plots::clone_budget::add(&mut total, field.field_id.capacity())?;
            crate::plots::clone_budget::add(
                &mut total,
                crate::plots::clone_budget::optional_string(field.label.as_ref()),
            )?;
            crate::plots::clone_budget::add(
                &mut total,
                crate::plots::clone_budget::vec::<Vec3>(field.vectors.capacity())?,
            )?;
        }
        if let Some(field) = &self.deformation {
            crate::plots::clone_budget::add(&mut total, field.field_id.capacity())?;
            crate::plots::clone_budget::add(
                &mut total,
                crate::plots::clone_budget::optional_string(field.label.as_ref()),
            )?;
            crate::plots::clone_budget::add(
                &mut total,
                crate::plots::clone_budget::vec::<Vec3>(field.displacements.capacity())?,
            )?;
        }
        Some(total)
    }
}
