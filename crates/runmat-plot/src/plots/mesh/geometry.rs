//! A mesh plot's face geometry, colors and edge lists, as its render data builds them.

use super::*;

impl MeshPlot {
    pub(super) fn generate_face_geometry(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
        if self.dirty || self.face_vertices.is_none() || self.face_indices.is_none() {
            let color = self.effective_face_color();
            let vertices_source = self.effective_vertices();
            // Source normals hold for the undeformed shape; a deformed one recomputes them.
            let normals = match (&self.vertex_normals, &self.deformation) {
                (Some(source), None) => {
                    let computed = vertex_normals(&vertices_source, &self.triangles);
                    source
                        .iter()
                        .zip(computed)
                        .map(|(&n, fallback)| if n == Vec3::ZERO { fallback } else { n })
                        .collect()
                }
                _ => vertex_normals(&vertices_source, &self.triangles),
            };
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

    pub(super) fn effective_vertices(&self) -> Cow<'_, [Vec3]> {
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

    pub(super) fn scalar_color(
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

    pub(super) fn triangle_color(&self, triangle_index: usize) -> Option<Vec4> {
        let mut color = *self.triangle_colors.as_ref()?.get(triangle_index)?;
        color.w *= self.face_alpha.clamp(0.0, 1.0);
        Some(color)
    }

    pub(super) fn vertex_color(&self, vertex_index: usize) -> Option<Vec4> {
        let mut color = *self.vertex_colors.as_ref()?.get(vertex_index)?;
        color.w *= self.face_alpha.clamp(0.0, 1.0);
        Some(color)
    }

    pub(super) fn renderable_edges(&self) -> Vec<(u32, u32)> {
        match self.edge_mode {
            MeshEdgeMode::All => self.all_triangle_edges(),
            MeshEdgeMode::Feature => self.feature_edges(),
            MeshEdgeMode::None => Vec::new(),
        }
    }

    pub(super) fn all_triangle_edges(&self) -> Vec<(u32, u32)> {
        let mut edges = BTreeSet::<(u32, u32)>::new();
        for &[a, b, c] in &self.triangles {
            edges.insert(normalized_edge(a, b));
            edges.insert(normalized_edge(b, c));
            edges.insert(normalized_edge(c, a));
        }
        edges.into_iter().collect()
    }

    pub(super) fn feature_edges(&self) -> Vec<(u32, u32)> {
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

    pub(super) fn clear_invalid_triangle_metadata(&mut self) {
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

    pub(super) fn clear_invalid_vertex_metadata(&mut self) {
        if self
            .vertex_colors
            .as_ref()
            .is_some_and(|colors| colors.len() != self.vertices.len())
        {
            self.vertex_colors = None;
        }
    }
}

pub(super) fn validate_mesh(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Result<(), String> {
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

pub(super) fn validate_scalar_field(
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

pub(super) fn validate_vector_field(
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

pub(super) fn validate_deformation(
    deformation: &MeshDeformation,
    vertex_count: usize,
) -> Result<(), String> {
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

pub(super) fn validate_field_len(
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

pub(super) fn sanitize_scalar_field(mut field: MeshScalarField) -> MeshScalarField {
    field.alpha = sanitize_alpha(field.alpha);
    if field.colormap.trim().is_empty() {
        field.colormap = "viridis".to_string();
    }
    field
}

pub(super) fn sanitize_vector_field(mut field: MeshVectorField) -> MeshVectorField {
    field.scale = if field.scale.is_finite() {
        field.scale
    } else {
        1.0
    };
    field.stride = field.stride.max(1);
    field.color = sanitize_color(field.color);
    field
}

pub(super) fn sanitize_deformation(mut deformation: MeshDeformation) -> MeshDeformation {
    deformation.scale = if deformation.scale.is_finite() {
        deformation.scale
    } else {
        1.0
    };
    deformation
}

pub(super) fn scalar_limits(field: &MeshScalarField) -> Option<[f32; 2]> {
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

pub(super) fn colormap_color(name: &str, t: f32, alpha: f32) -> Vec4 {
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

pub(super) fn vertex_normals(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Vec<Vec3> {
    // Area weighting must not depend on the mesh's coordinate unit. Squared
    // triangle areas in meters can be far below f32::EPSILON on valid CAD.
    // Accumulate in f64 to retain tiny faces and avoid overflow on large ones.
    let mut normals = vec![DVec3::ZERO; vertices.len()];
    for &[a, b, c] in triangles {
        let [ia, ib, ic] = [a as usize, b as usize, c as usize];
        let origin = vertices[ia].as_dvec3();
        let normal = (vertices[ib].as_dvec3() - origin).cross(vertices[ic].as_dvec3() - origin);
        if normal.is_finite() {
            normals[ia] += normal;
            normals[ib] += normal;
            normals[ic] += normal;
        }
    }
    normals
        .into_iter()
        .map(|normal| {
            normal
                .try_normalize()
                .map_or(Vec3::Z, |normal| normal.as_vec3())
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FeatureEdgeAccumulator {
    pub(super) first_group: u64,
    pub(super) count: u8,
    pub(super) crosses_group: bool,
}

pub(super) fn accumulate_feature_edge(
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

pub(super) fn normalized_edge(a: u32, b: u32) -> (u32, u32) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

pub(super) fn sanitize_color(color: Vec4) -> Vec4 {
    Vec4::new(
        sanitize_color_component(color.x),
        sanitize_color_component(color.y),
        sanitize_color_component(color.z),
        sanitize_color_component(color.w),
    )
}

pub(super) fn sanitize_color_component(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

pub(super) fn sanitize_alpha(alpha: f32) -> f32 {
    if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0
    }
}
