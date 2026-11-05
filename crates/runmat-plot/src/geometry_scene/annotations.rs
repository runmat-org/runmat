use super::*;

impl GeometryScene {
    /// The overlay reserves the same shaft/head geometry the symbol draws,
    /// including a head-on vector whose projected shaft has no length.
    #[cfg(feature = "egui-overlay")]
    pub(crate) fn annotation_vector_points(
        &self,
        region_id: &str,
        direction: [f32; 3],
    ) -> Option<[Vec3; 4]> {
        let direction = normalized_annotation_direction(direction)?;
        let length = annotation_arrow_length(self.bounds);
        self.chunks
            .iter()
            .filter(|chunk| chunk.visible)
            .find_map(|chunk| {
                let frame = chunk.region_annotation_frame(region_id)?;
                let start = surface_arrow_start(frame.anchor, frame.normal, direction, length);
                let end = start + direction * length;
                let side = perpendicular_unit(direction);
                let base = end - direction * (length * 0.28);
                let wing = side * (length * 0.16);
                Some([start, end, base + wing, base - wing])
            })
    }
    pub(super) fn topology_selection_nodes(
        &self,
        presentation: &GeometryScenePresentation,
    ) -> Vec<SceneNode> {
        if presentation.selected_topology.is_empty() {
            return Vec::new();
        }
        let mut strokes = Vec::new();
        let color = GEOMETRY_SELECTED_REGION_COLOR;
        for selection in &presentation.selected_topology {
            match selection {
                GeometrySceneTopologySelection::Edge {
                    chunk_id,
                    vertex_indices,
                    segments,
                } => {
                    let Some(chunk) = self
                        .chunks
                        .iter()
                        .find(|chunk| chunk.visible && chunk.chunk_id == *chunk_id)
                    else {
                        log::warn!(target: "runmat_plot", "geometry_scene.selection_edge_missing_chunk chunk_id={chunk_id}");
                        continue;
                    };
                    let selected_segments = if segments.is_empty() {
                        std::slice::from_ref(vertex_indices)
                    } else {
                        segments.as_slice()
                    };
                    for segment in selected_segments {
                        let (Some(a), Some(b)) = (
                            chunk.vertices.get(segment[0] as usize),
                            chunk.vertices.get(segment[1] as usize),
                        ) else {
                            log::warn!(target: "runmat_plot", "geometry_scene.selection_edge_missing_vertex chunk_id={chunk_id} vertex_indices={segment:?}");
                            continue;
                        };
                        append_selection_edge(
                            &mut strokes,
                            Vec3::from_array(a.position),
                            Vec3::from_array(b.position),
                            color,
                        );
                    }
                }
                GeometrySceneTopologySelection::Point {
                    chunk_id,
                    vertex_index,
                } => {
                    let Some(chunk) = self
                        .chunks
                        .iter()
                        .find(|chunk| chunk.visible && chunk.chunk_id == *chunk_id)
                    else {
                        log::warn!(target: "runmat_plot", "geometry_scene.selection_point_missing_chunk chunk_id={chunk_id}");
                        continue;
                    };
                    let Some(vertex) = chunk.vertices.get(*vertex_index as usize) else {
                        log::warn!(target: "runmat_plot", "geometry_scene.selection_point_missing_vertex chunk_id={chunk_id} vertex_index={vertex_index}");
                        continue;
                    };
                    append_selection_point(&mut strokes, Vec3::from_array(vertex.position), color);
                }
                GeometrySceneTopologySelection::Face { .. } => {}
            }
        }
        if strokes.is_empty() {
            return Vec::new();
        }
        vec![self.annotation_node(
            "Selected geometry topology",
            "__geometry_selection:topology",
            self.chunks.len() + 2,
            PipelineType::AnnotationStroke,
            strokes,
        )]
    }

    pub(super) fn annotation_nodes(
        &self,
        presentation: &GeometryScenePresentation,
    ) -> Vec<SceneNode> {
        if presentation.region_annotations.is_empty() {
            return Vec::new();
        }

        let mut point_vertices = Vec::new();
        let mut stroke_vertices = Vec::new();
        let arrow_length = annotation_arrow_length(self.bounds);
        for annotation in &presentation.region_annotations {
            if annotation.kind == GeometrySceneAnnotationKind::Label {
                continue;
            }
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
                        let Some(direction) = direction else {
                            append_annotation_marker(
                                &mut point_vertices,
                                frame.anchor,
                                color,
                                annotation.size,
                            );
                            continue;
                        };
                        append_arrow(
                            &mut stroke_vertices,
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
                        let Some(direction) = direction else {
                            append_annotation_marker(
                                &mut point_vertices,
                                frame.anchor,
                                color,
                                annotation.size,
                            );
                            continue;
                        };
                        let (anchors, face_span) =
                            chunk.distributed_annotation_anchors(&annotation.region_id, frame);
                        for (index, anchor) in anchors.into_iter().enumerate() {
                            append_arrow_with_visibility(
                                &mut stroke_vertices,
                                surface_arrow_start(anchor, frame.normal, direction, arrow_length),
                                direction,
                                arrow_length,
                                color,
                                if index == 0 { 0.0 } else { face_span },
                            );
                        }
                    }
                    GeometrySceneAnnotationKind::Moment => {
                        if let Some(direction) = direction {
                            append_moment(
                                &mut stroke_vertices,
                                frame,
                                direction,
                                arrow_length,
                                color,
                            );
                        } else {
                            append_annotation_marker(
                                &mut point_vertices,
                                frame.anchor,
                                color,
                                annotation.size,
                            );
                        }
                    }
                    GeometrySceneAnnotationKind::FixedSupport => {
                        append_fixed_support(
                            &mut stroke_vertices,
                            frame.anchor,
                            frame,
                            arrow_length * 0.52,
                            color,
                        );
                    }
                    GeometrySceneAnnotationKind::SymmetrySupport => append_symmetry_support(
                        &mut stroke_vertices,
                        frame,
                        arrow_length * 0.8,
                        color,
                    ),
                    GeometrySceneAnnotationKind::PrescribedMotion => {
                        let Some(direction) = direction else {
                            append_annotation_marker(
                                &mut point_vertices,
                                frame.anchor,
                                color,
                                annotation.size,
                            );
                            continue;
                        };
                        append_arrow(
                            &mut stroke_vertices,
                            frame.anchor + frame.normal * (arrow_length * 0.12)
                                - direction * (arrow_length * 0.4),
                            direction,
                            arrow_length * 0.8,
                            color,
                        );
                    }
                    _ => append_annotation_marker(
                        &mut point_vertices,
                        frame.anchor,
                        color,
                        annotation.size,
                    ),
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
        if !stroke_vertices.is_empty() {
            nodes.push(self.annotation_node(
                "FEA study symbols",
                "__fea_annotations:vectors",
                self.chunks.len() + 1,
                PipelineType::AnnotationStroke,
                stroke_vertices,
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
            instances: None,
        }
    }
}

fn append_annotation_marker(
    vertices: &mut Vec<Vertex>,
    anchor: Vec3,
    color: [f32; 4],
    size: Option<f32>,
) {
    let mut marker = vertex(anchor.to_array(), color, [0.0, 0.0, size.unwrap_or(15.0)]);
    marker.tex_coords = [1.0, 1.0];
    vertices.push(marker);
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

    fn distributed_annotation_anchors(
        &self,
        region_id: &str,
        frame: AnnotationFrame,
    ) -> (Vec<Vec3>, f32) {
        let Some(region) = self.regions.iter().find(|item| item.region_id == region_id) else {
            return (vec![frame.anchor], 0.0);
        };
        let mut triangles = Vec::new();
        let mut tangent_range = (f32::INFINITY, f32::NEG_INFINITY);
        let mut bitangent_range = (f32::INFINITY, f32::NEG_INFINITY);
        for range in &region.triangle_ranges {
            let start = range.start as usize;
            let end = start.saturating_add(range.count as usize);
            for triangle_index in start..end {
                let Some((a, b, c)) = self.triangle_vertices(triangle_index) else {
                    continue;
                };
                for vertex in [a, b, c] {
                    let relative = vertex - frame.anchor;
                    let t = relative.dot(frame.tangent);
                    let u = relative.dot(frame.bitangent);
                    tangent_range.0 = tangent_range.0.min(t);
                    tangent_range.1 = tangent_range.1.max(t);
                    bitangent_range.0 = bitangent_range.0.min(u);
                    bitangent_range.1 = bitangent_range.1.max(u);
                }
                triangles.push((a, b, c));
            }
        }
        if triangles.is_empty() {
            return (vec![frame.anchor], 0.0);
        }
        let tangent_span = tangent_range.1 - tangent_range.0;
        let bitangent_span = bitangent_range.1 - bitangent_range.0;
        let (axis, span) = if tangent_span >= bitangent_span {
            (frame.tangent, tangent_span)
        } else {
            (frame.bitangent, bitangent_span)
        };
        if !span.is_finite() || span <= 1.0e-6 {
            return (vec![frame.anchor], 0.0);
        }
        let mut anchors = Vec::with_capacity(3);
        for fraction in [0.0_f32, -0.28, 0.28] {
            let target = frame.anchor + axis * span * fraction;
            let anchor = triangles
                .iter()
                .map(|&(a, b, c)| nearest_point_on_triangle(target, a, b, c))
                .min_by(|a, b| {
                    a.distance_squared(target)
                        .total_cmp(&b.distance_squared(target))
                })
                .unwrap_or(frame.anchor);
            if anchors
                .iter()
                .all(|previous: &Vec3| previous.distance(anchor) > span * 0.08)
            {
                anchors.push(anchor);
            }
        }
        (anchors, span)
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
        GeometrySceneAnnotationKind::Label => [0.35, 0.65, 0.95, 1.0],
        GeometrySceneAnnotationKind::FixedSupport
        | GeometrySceneAnnotationKind::SymmetrySupport
        | GeometrySceneAnnotationKind::PrescribedMotion
        | GeometrySceneAnnotationKind::BoundaryCondition => [0.13, 0.38, 0.82, 1.0],
        GeometrySceneAnnotationKind::Force
        | GeometrySceneAnnotationKind::Moment
        | GeometrySceneAnnotationKind::Pressure
        | GeometrySceneAnnotationKind::Traction
        | GeometrySceneAnnotationKind::BodyForce => [0.78, 0.16, 0.45, 1.0],
        GeometrySceneAnnotationKind::HeatFlux | GeometrySceneAnnotationKind::ThermalCondition => {
            [0.83, 0.39, 0.08, 1.0]
        }
        GeometrySceneAnnotationKind::CurrentDensity
        | GeometrySceneAnnotationKind::ElectricalCondition => [0.63, 0.43, 0.06, 1.0],
        GeometrySceneAnnotationKind::MagneticCondition => [0.44, 0.34, 0.72, 1.0],
        GeometrySceneAnnotationKind::NormalVelocity
        | GeometrySceneAnnotationKind::AcousticCondition => [0.49, 0.29, 0.75, 1.0],
        GeometrySceneAnnotationKind::FlowVelocity
        | GeometrySceneAnnotationKind::MassFlow
        | GeometrySceneAnnotationKind::WallMotion
        | GeometrySceneAnnotationKind::FluidCondition => [0.05, 0.48, 0.54, 1.0],
        GeometrySceneAnnotationKind::Connector => [0.14, 0.52, 0.31, 1.0],
        GeometrySceneAnnotationKind::Interface => [0.05, 0.50, 0.59, 1.0],
        GeometrySceneAnnotationKind::InitialCondition => [0.43, 0.48, 0.57, 1.0],
    }
}

fn normalized_annotation_direction(direction: [f32; 3]) -> Option<Vec3> {
    let direction = Vec3::from_array(direction);
    let length = direction.length();
    (length.is_finite() && length > 1.0e-8).then_some(direction / length)
}

fn nearest_point_on_triangle(target: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = target - a;
    let d00 = ab.dot(ab);
    let d01 = ab.dot(ac);
    let d11 = ac.dot(ac);
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() <= 1.0e-12 {
        return (a + b + c) / 3.0;
    }
    let d20 = ap.dot(ab);
    let d21 = ap.dot(ac);
    let v = (d11 * d20 - d01 * d21) / denominator;
    let w = (d00 * d21 - d01 * d20) / denominator;
    let mut weights = [(1.0 - v - w).max(0.0), v.max(0.0), w.max(0.0)];
    let sum = weights.iter().sum::<f32>().max(1.0e-12);
    for weight in &mut weights {
        *weight /= sum;
    }
    a * weights[0] + b * weights[1] + c * weights[2]
}

fn append_arrow(
    vertices: &mut Vec<Vertex>,
    anchor: Vec3,
    direction: Vec3,
    length: f32,
    color: [f32; 4],
) {
    append_arrow_with_visibility(vertices, anchor, direction, length, color, 0.0);
}

fn append_arrow_with_visibility(
    vertices: &mut Vec<Vertex>,
    anchor: Vec3,
    direction: Vec3,
    length: f32,
    color: [f32; 4],
    optional_face_span: f32,
) {
    let end = anchor + direction * length;
    append_stroke(vertices, anchor, end, color, optional_face_span);
    let side = perpendicular_unit(direction);
    let wing_base = end - direction * (length * 0.28);
    let wing_size = length * 0.16;
    // Fill the head in the same annotation plane and unlit stroke pipeline.
    // Closely spaced rays keep the tip solid at the CAD marker's screen scale.
    for step in -3..=3 {
        append_stroke(
            vertices,
            end,
            wing_base + side * (wing_size * step as f32 / 4.0),
            color,
            optional_face_span,
        );
    }
    append_stroke(
        vertices,
        end,
        wing_base + side * wing_size,
        color,
        optional_face_span,
    );
    append_stroke(
        vertices,
        end,
        wing_base - side * wing_size,
        color,
        optional_face_span,
    );
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
    append_stroke(vertices, start, end, color, 0.0);
}

fn append_stroke(
    vertices: &mut Vec<Vertex>,
    start: Vec3,
    end: Vec3,
    color: [f32; 4],
    optional_face_span: f32,
) {
    let direction = end - start;
    let length = direction.length();
    if !length.is_finite() || length <= 1.0e-8 {
        return;
    }
    let a_minus = annotation_stroke_vertex(start, end, -1.0, color, optional_face_span);
    let a_plus = annotation_stroke_vertex(start, end, 1.0, color, optional_face_span);
    // The segment tangent reverses at the far endpoint, so its side signs do too.
    let b_minus = annotation_stroke_vertex(end, start, 1.0, color, optional_face_span);
    let b_plus = annotation_stroke_vertex(end, start, -1.0, color, optional_face_span);
    vertices.extend_from_slice(&[a_minus, a_plus, b_plus, a_minus, b_plus, b_minus]);
}

fn append_selection_edge(vertices: &mut Vec<Vertex>, start: Vec3, end: Vec3, color: [f32; 4]) {
    let first = vertices.len();
    append_stroke(vertices, start, end, [0.10, 0.11, 0.15, 1.0], 0.0);
    for vertex in &mut vertices[first..] {
        vertex.tex_coords[1] = 1.25;
    }
    let first = vertices.len();
    append_stroke(vertices, start, end, color, 0.0);
    for vertex in &mut vertices[first..] {
        vertex.tex_coords[1] = 1.0;
    }
}

fn append_selection_point(vertices: &mut Vec<Vertex>, center: Vec3, color: [f32; 4]) {
    // The stroke shader turns this quad into a screen-sized disc. Its center
    // remains at the picked mesh vertex as the camera zooms or rotates.
    for corner in [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ] {
        let mut marker = vertex(center.to_array(), color, [corner[0], corner[1], 0.0]);
        marker.tex_coords = [0.0, 2.0];
        vertices.push(marker);
    }
}

fn annotation_stroke_vertex(
    position: Vec3,
    other: Vec3,
    side: f32,
    color: [f32; 4],
    optional_face_span: f32,
) -> Vertex {
    let mut stroke_color = color;
    stroke_color[3] = optional_face_span;
    let mut result = vertex(position.to_array(), stroke_color, [0.0, 0.0, 1.0]);
    result.normal = other.to_array();
    result.tex_coords = [side, 0.0];
    result
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
