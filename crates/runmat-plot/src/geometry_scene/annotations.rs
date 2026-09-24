use super::*;

impl GeometryScene {
    pub(super) fn annotation_nodes(
        &self,
        presentation: &GeometryScenePresentation,
    ) -> Vec<SceneNode> {
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
                let Some(frame) = chunk.region_annotation_frame(&annotation.region_id) else {
                    continue;
                };
                let color = annotation
                    .color
                    .unwrap_or_else(|| annotation_color(annotation.kind));
                let direction = annotation
                    .direction
                    .and_then(normalized_annotation_direction);
                match annotation.kind {
                    GeometrySceneAnnotationKind::Force | GeometrySceneAnnotationKind::BodyForce => {
                        let direction = direction.unwrap_or(frame.normal);
                        append_arrow(
                            &mut line_vertices,
                            surface_arrow_start(
                                frame.anchor,
                                frame.normal,
                                direction,
                                arrow_length,
                            ),
                            direction,
                            arrow_length,
                            color,
                        );
                    }
                    GeometrySceneAnnotationKind::Pressure
                    | GeometrySceneAnnotationKind::Traction
                    | GeometrySceneAnnotationKind::HeatFlux
                    | GeometrySceneAnnotationKind::CurrentDensity
                    | GeometrySceneAnnotationKind::NormalVelocity
                    | GeometrySceneAnnotationKind::FlowVelocity
                    | GeometrySceneAnnotationKind::MassFlow
                    | GeometrySceneAnnotationKind::WallMotion => {
                        let direction = direction.unwrap_or(-frame.normal);
                        for anchor in chunk.annotation_samples(&annotation.region_id, 5) {
                            append_arrow(
                                &mut line_vertices,
                                surface_arrow_start(anchor, frame.normal, direction, arrow_length),
                                direction,
                                arrow_length,
                                color,
                            );
                        }
                    }
                    GeometrySceneAnnotationKind::Moment => append_moment(
                        &mut line_vertices,
                        frame,
                        direction.unwrap_or(frame.normal),
                        arrow_length,
                        color,
                    ),
                    GeometrySceneAnnotationKind::FixedSupport => {
                        for anchor in chunk.annotation_samples(&annotation.region_id, 4) {
                            append_fixed_support(
                                &mut line_vertices,
                                anchor,
                                frame,
                                arrow_length * 0.52,
                                color,
                            );
                        }
                    }
                    GeometrySceneAnnotationKind::SymmetrySupport => append_symmetry_support(
                        &mut line_vertices,
                        frame,
                        arrow_length * 0.8,
                        color,
                    ),
                    GeometrySceneAnnotationKind::PrescribedMotion => {
                        let direction = direction.unwrap_or(frame.normal);
                        append_arrow(
                            &mut line_vertices,
                            frame.anchor + frame.normal * (arrow_length * 0.12)
                                - direction * (arrow_length * 0.4),
                            direction,
                            arrow_length * 0.8,
                            color,
                        );
                    }
                    _ => {
                        let mut marker = vertex(
                            frame.anchor.to_array(),
                            color,
                            [0.0, 0.0, annotation.size.unwrap_or(15.0)],
                        );
                        marker.tex_coords = [1.0, 1.0];
                        point_vertices.push(marker);
                    }
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
                "FEA study symbols",
                "__fea_annotations:vectors",
                self.chunks.len() + 1,
                // Engineering symbols must remain legible on the selected CAD face. A regular
                // depth-tested line list z-fights with that face and can collapse to a few bright
                // pixels at ordinary camera distances.
                PipelineType::LinesNoDepth,
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

impl GeometrySceneChunk {
    #[cfg(feature = "egui-overlay")]
    pub(super) fn region_anchor(&self, region_id: &str) -> Option<Vec3> {
        self.region_annotation_frame(region_id)
            .map(|frame| frame.anchor)
    }

    fn region_annotation_frame(&self, region_id: &str) -> Option<AnnotationFrame> {
        if self.render_data.pipeline_type != PipelineType::Triangles {
            return None;
        }
        let region = self
            .regions
            .iter()
            .find(|item| item.region_id == region_id)?;
        let mut weighted_centroid = Vec3::ZERO;
        let mut total_area = 0.0_f32;
        let mut weighted_normal = Vec3::ZERO;
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
                    weighted_normal += (b - a).cross(c - a).normalize_or_zero() * area;
                    total_area += area;
                } else {
                    fallback_centroid += centroid;
                    fallback_count += 1;
                }
            }
        }
        let anchor = if total_area > 0.0 {
            weighted_centroid / total_area
        } else if fallback_count > 0 {
            fallback_centroid / fallback_count as f32
        } else {
            return None;
        };
        let normal = if weighted_normal.length_squared() > 1.0e-8 {
            weighted_normal.normalize()
        } else {
            Vec3::Z
        };
        let tangent = perpendicular_unit(normal);
        Some(AnnotationFrame {
            anchor,
            normal,
            tangent,
            bitangent: normal.cross(tangent).normalize_or_zero(),
        })
    }

    fn annotation_samples(&self, region_id: &str, maximum: usize) -> Vec<Vec3> {
        let Some(region) = self.regions.iter().find(|item| item.region_id == region_id) else {
            return Vec::new();
        };
        let mut samples = Vec::new();
        for range in &region.triangle_ranges {
            let start = range.start as usize;
            let end = start.saturating_add(range.count as usize);
            for triangle_index in start..end {
                let Some((a, b, c)) = self.triangle_vertices(triangle_index) else {
                    continue;
                };
                let centroid = (a + b + c) / 3.0;
                if samples
                    .iter()
                    .all(|sample: &Vec3| sample.distance(centroid) > 1.0e-5)
                {
                    samples.push(centroid);
                }
            }
        }
        if samples.len() <= maximum {
            return samples;
        }
        (0..maximum)
            .filter_map(|index| {
                let sample_index = index * (samples.len() - 1) / maximum.saturating_sub(1).max(1);
                samples.get(sample_index).copied()
            })
            .collect()
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
}

#[derive(Debug, Clone, Copy)]
struct AnnotationFrame {
    anchor: Vec3,
    normal: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
}

fn annotation_arrow_length(bounds: BoundingBox) -> f32 {
    let diagonal = bounds.size().length();
    if diagonal.is_finite() && diagonal > 1.0e-6 {
        diagonal * 0.095
    } else {
        1.0
    }
}

fn surface_arrow_start(anchor: Vec3, normal: Vec3, direction: Vec3, length: f32) -> Vec3 {
    // Keep the complete glyph just outside the selected face. An outward vector starts at the
    // face and points away from it; an inward vector starts outside and terminates at the face.
    // Previously every arrow terminated at the face, which made outward loads look as though
    // they were buried in (and pointing into) the solid.
    let surface = anchor + normal * (length * 0.08);
    if direction.dot(normal) >= 0.0 {
        surface
    } else {
        surface - direction * length
    }
}

fn annotation_color(kind: GeometrySceneAnnotationKind) -> [f32; 4] {
    match kind {
        GeometrySceneAnnotationKind::FixedSupport
        | GeometrySceneAnnotationKind::SymmetrySupport
        | GeometrySceneAnnotationKind::PrescribedMotion => [0.24, 0.55, 0.96, 1.0],
        GeometrySceneAnnotationKind::Force
        | GeometrySceneAnnotationKind::Moment
        | GeometrySceneAnnotationKind::Pressure
        | GeometrySceneAnnotationKind::Traction
        | GeometrySceneAnnotationKind::BodyForce => [0.88, 0.28, 0.58, 1.0],
        GeometrySceneAnnotationKind::HeatFlux | GeometrySceneAnnotationKind::ThermalCondition => {
            [0.96, 0.51, 0.20, 1.0]
        }
        GeometrySceneAnnotationKind::CurrentDensity
        | GeometrySceneAnnotationKind::ElectricalCondition
        | GeometrySceneAnnotationKind::MagneticCondition => [0.96, 0.76, 0.22, 1.0],
        GeometrySceneAnnotationKind::NormalVelocity
        | GeometrySceneAnnotationKind::AcousticCondition => [0.67, 0.39, 0.96, 1.0],
        GeometrySceneAnnotationKind::FlowVelocity
        | GeometrySceneAnnotationKind::MassFlow
        | GeometrySceneAnnotationKind::WallMotion
        | GeometrySceneAnnotationKind::FluidCondition => [0.18, 0.72, 0.78, 1.0],
        GeometrySceneAnnotationKind::Connector | GeometrySceneAnnotationKind::Interface => {
            [0.26, 0.74, 0.48, 1.0]
        }
        GeometrySceneAnnotationKind::BoundaryCondition
        | GeometrySceneAnnotationKind::InitialCondition => [0.58, 0.62, 0.72, 1.0],
    }
}

fn normalized_annotation_direction(direction: [f32; 3]) -> Option<Vec3> {
    let direction = Vec3::from_array(direction);
    let length = direction.length();
    (length.is_finite() && length > 1.0e-8).then_some(direction / length)
}

fn append_arrow(
    vertices: &mut Vec<Vertex>,
    anchor: Vec3,
    direction: Vec3,
    length: f32,
    color: [f32; 4],
) {
    let end = anchor + direction * length;
    append_line(vertices, anchor, end, color);
    let side = perpendicular_unit(direction);
    let wing_base = end - direction * (length * 0.28);
    let wing_size = length * 0.16;
    append_line(vertices, end, wing_base + side * wing_size, color);
    append_line(vertices, end, wing_base - side * wing_size, color);
}

fn append_fixed_support(
    vertices: &mut Vec<Vertex>,
    anchor: Vec3,
    frame: AnnotationFrame,
    size: f32,
    color: [f32; 4],
) {
    let base = anchor + frame.normal * (size * 0.24);
    let half_width = size * 0.5;
    append_line(
        vertices,
        base - frame.tangent * half_width,
        base + frame.tangent * half_width,
        color,
    );
    for offset in [-0.42_f32, -0.14, 0.14, 0.42] {
        let root = base + frame.tangent * (offset * size);
        append_line(
            vertices,
            root,
            root + frame.normal * (size * 0.34) - frame.tangent * (size * 0.16),
            color,
        );
    }
    append_line(vertices, anchor, base, color);
}

fn append_symmetry_support(
    vertices: &mut Vec<Vertex>,
    frame: AnnotationFrame,
    size: f32,
    color: [f32; 4],
) {
    for offset in [-0.16_f32, 0.16] {
        let center = frame.anchor + frame.normal * (offset * size);
        append_line(
            vertices,
            center - frame.tangent * size * 0.5,
            center + frame.tangent * size * 0.5,
            color,
        );
    }
    append_line(
        vertices,
        frame.anchor - frame.bitangent * size * 0.18,
        frame.anchor + frame.bitangent * size * 0.18,
        color,
    );
}

fn append_moment(
    vertices: &mut Vec<Vertex>,
    frame: AnnotationFrame,
    axis: Vec3,
    size: f32,
    color: [f32; 4],
) {
    let tangent = perpendicular_unit(axis);
    let bitangent = axis.cross(tangent).normalize_or_zero();
    let radius = size * 0.55;
    let segments = 12;
    let mut previous = frame.anchor + tangent * radius;
    for index in 1..=segments {
        let angle = std::f32::consts::TAU * index as f32 / segments as f32 * 0.82;
        let next = frame.anchor + (tangent * angle.cos() + bitangent * angle.sin()) * radius;
        append_line(vertices, previous, next, color);
        previous = next;
    }
    let direction = (previous - frame.anchor).normalize_or_zero();
    append_line(
        vertices,
        previous,
        previous - direction * size * 0.22 + tangent * size * 0.12,
        color,
    );
    append_line(
        vertices,
        previous,
        previous - direction * size * 0.22 - tangent * size * 0.12,
        color,
    );
}

fn append_line(vertices: &mut Vec<Vertex>, start: Vec3, end: Vec3, color: [f32; 4]) {
    let direction = end - start;
    let length = direction.length();
    if length.is_finite() && length > 1.0e-8 {
        // WebGPU line primitives are fixed at one physical pixel on the platforms we support.
        // Add two tightly-spaced companion strokes so engineering annotations stay legible on
        // high-DPI displays without changing the general-purpose line pipeline.
        let side = perpendicular_unit(direction / length) * (length * 0.018);
        append_line_segment(vertices, start - side, end - side, color);
        append_line_segment(vertices, start + side, end + side, color);
    }
    append_line_segment(vertices, start, end, color);
}

fn append_line_segment(vertices: &mut Vec<Vertex>, start: Vec3, end: Vec3, color: [f32; 4]) {
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
