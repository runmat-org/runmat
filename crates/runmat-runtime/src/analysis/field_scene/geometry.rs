//! Build disposable scene geometry from exact persisted topology and paged fields.
use super::*;
#[cfg(test)]
pub(super) fn scene_from_topology(
    run_id: &str,
    topology: &AnalysisRenderTopology,
    field: Option<&FieldData>,
    deformation: Option<&FieldData>,
    shape_mode: &str,
    exaggeration: f64,
    show_undeformed_outline: bool,
) -> Result<(GeometryScene, f64), String> {
    scene_from_topology_with_stats(
        run_id,
        topology,
        field,
        deformation,
        shape_mode,
        exaggeration,
        show_undeformed_outline,
        None,
    )
}

pub(super) fn scene_from_topology_with_stats(
    run_id: &str,
    topology: &AnalysisRenderTopology,
    field: Option<&FieldData>,
    deformation: Option<&FieldData>,
    shape_mode: &str,
    exaggeration: f64,
    show_undeformed_outline: bool,
    stats: Option<SceneStats>,
) -> Result<(GeometryScene, f64), String> {
    let deformation_scale = match (shape_mode, deformation) {
        ("true", Some(_)) => 1.0,
        ("exaggerated", Some(field)) => {
            let automatic_scale = match stats.as_ref().and_then(|stats| stats.automatic_scale) {
                Some(scale) => scale,
                None => automatic_deformation_scale(topology, field)?,
            };
            automatic_scale * exaggeration
        }
        _ => 0.0,
    };
    let element_count = stats
        .as_ref()
        .map(|stats| stats.element_count)
        .unwrap_or(topology.volume_element_count);
    let element_stride = if let Some(field) =
        field.filter(|field| field.location == AnalysisFieldLocation::Element)
    {
        let entity_count = field.value_count / field.components;
        if element_count == 0 || entity_count % element_count != 0 {
            return Err(format!(
                "field '{}' cannot be mapped to the surface elements",
                field.label
            ));
        }
        entity_count / element_count
    } else {
        1
    };
    let mut minimum = stats
        .as_ref()
        .and_then(|stats| stats.scalar_range)
        .map_or(f64::INFINITY, |range| range.0);
    let mut maximum = stats
        .as_ref()
        .and_then(|stats| stats.scalar_range)
        .map_or(f64::NEG_INFINITY, |range| range.1);
    let mut sample_count = 0_usize;
    if let Some(field) = field {
        for mesh in &topology.meshes {
            match field.location {
                AnalysisFieldLocation::Node => {
                    for index in mesh.vertex_volume_node_indices.iter().flatten() {
                        let sample = field_magnitude(field, *index, 1)?;
                        minimum = minimum.min(sample);
                        maximum = maximum.max(sample);
                        sample_count += 1;
                    }
                }
                AnalysisFieldLocation::Element => {
                    for index in mesh.triangle_volume_element_indices.iter().flatten() {
                        let sample = field_magnitude(field, *index, element_stride)?;
                        minimum = minimum.min(sample);
                        maximum = maximum.max(sample);
                        sample_count += 1;
                    }
                }
                _ => unreachable!(),
            }
        }
        if sample_count == 0 {
            return Err(format!(
                "field '{}' has no mapping onto the persisted surface",
                field.label
            ));
        }
    }
    let mut chunks = Vec::new();
    let mut vertex_count = 0usize;
    let mut triangle_count = 0usize;
    for mesh in &topology.meshes {
        for (batch_index, triangles) in mesh.triangles.chunks(MAX_TRIANGLES_PER_TILE).enumerate() {
            let mut vertices = Vec::with_capacity(triangles.len() * 3);
            for (local_index, triangle) in triangles.iter().enumerate() {
                let triangle_index = batch_index * MAX_TRIANGLES_PER_TILE + local_index;
                let positions = triangle.map(|index| mesh.vertices.get(index as usize));
                let [Some(a), Some(b), Some(c)] = positions else {
                    return Err(format!(
                        "render topology '{}' contains an invalid triangle",
                        mesh.mesh_id
                    ));
                };
                let positions = [a, b, c].map(|position| cast_position(*position));
                let deformed = [0, 1, 2].map(|corner| {
                    let node = mesh
                        .vertex_volume_node_indices
                        .get(triangle[corner] as usize)
                        .copied()
                        .flatten();
                    deform_position(positions[corner], node, deformation, deformation_scale)
                });
                let [first, second, third] = deformed;
                let deformed = [first?, second?, third?];
                let normal = (Vec3::from_array(deformed[1]) - Vec3::from_array(deformed[0]))
                    .cross(Vec3::from_array(deformed[2]) - Vec3::from_array(deformed[0]))
                    .normalize_or_zero()
                    .to_array();
                for corner in 0..3 {
                    let sample = if let Some(field) = field {
                        let entity = match field.location {
                            AnalysisFieldLocation::Node => mesh
                                .vertex_volume_node_indices
                                .get(triangle[corner] as usize)
                                .copied()
                                .flatten(),
                            AnalysisFieldLocation::Element => mesh
                                .triangle_volume_element_indices
                                .get(triangle_index)
                                .copied()
                                .flatten(),
                            _ => None,
                        };
                        entity
                            .map(|index| field_magnitude(field, index, element_stride))
                            .transpose()?
                    } else {
                        None
                    };
                    let color = sample
                        .map(|value| scalar_color(value, minimum, maximum))
                        .unwrap_or([0.66, 0.73, 0.83, 1.0]);
                    vertices.push(Vertex {
                        position: deformed[corner],
                        color,
                        normal,
                        tex_coords: [0.0, 0.0],
                    });
                }
                triangle_count += 1;
            }
            vertex_count += vertices.len();
            let indices = (0..vertices.len() as u32).collect();
            let material = Material {
                albedo: Vec4::ONE,
                double_sided: true,
                ..Material::default()
            };
            let mut chunk = GeometrySceneChunk::indexed_triangles(
                format!("{}:surface:{batch_index}", mesh.mesh_id),
                vertices,
                indices,
                material,
            );
            chunk.mesh_id = Some(mesh.mesh_id.clone());
            chunks.push(chunk);
        }
    }
    if chunks.is_empty() {
        return Err("FEA run has no renderable surface mesh".into());
    }
    if show_undeformed_outline && deformation_scale != 0.0 {
        append_undeformed_outline(&mut chunks, topology);
    }
    let mut scene = GeometryScene::new(
        format!("fea:{run_id}:{}", field.map_or("mesh", |f| &f.label)),
        1,
        chunks,
    );
    append_solver_mesh_edges(&mut scene);
    scene.overlay = Some(GeometrySceneOverlay {
        source_name: None,
        status: GeometrySceneCompleteness::Complete,
        quality_label: None,
        format: None,
        source_label: None,
        allow_create_fea_study: false,
        byte_count: None,
        mesh_count: topology.meshes.len(),
        vertex_count,
        triangle_count,
        progress_percent: None,
        scalar_legend: field.map(|field| GeometrySceneScalarLegend {
            label: if element_stride > 1 {
                format!("{} mean magnitude", field.label)
            } else if field.components > 1 {
                format!("{} magnitude", field.label)
            } else {
                field.label.clone()
            },
            unit: field.unit.clone(),
            minimum,
            maximum,
            limit: None,
            note: None,
        }),
        region_count: 0,
        mapped_region_count: 0,
        assembly_nodes: Vec::new(),
        regions: Vec::new(),
        warnings: Vec::new(),
    });
    Ok((scene, deformation_scale))
}

pub(super) fn cast_position(position: [f64; 3]) -> [f32; 3] {
    position.map(|value| value as f32)
}

pub(super) fn append_undeformed_outline(
    chunks: &mut Vec<GeometrySceneChunk>,
    topology: &AnalysisRenderTopology,
) {
    for mesh in &topology.meshes {
        let mut seen = HashSet::new();
        let mut vertices = Vec::new();
        for triangle in &mesh.triangles {
            for (a, b) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                for index in [a, b] {
                    let position = cast_position(mesh.vertices[index as usize]);
                    vertices.push(Vertex {
                        position,
                        color: [0.92, 0.92, 1.0, 0.35],
                        normal: [0.0, 0.0, 0.0],
                        tex_coords: [0.0, 0.0],
                    });
                }
            }
        }
        if vertices.is_empty() {
            continue;
        }
        let vertex_count = vertices.len();
        let material = Material {
            alpha_mode: AlphaMode::Blend,
            ..Material::default()
        };
        let mut chunk = GeometrySceneChunk::from_render_data(
            format!("{}:undeformed-outline", mesh.mesh_id),
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
        chunk.mesh_id = Some(mesh.mesh_id.clone());
        chunk.label = Some("Undeformed outline".into());
        chunks.push(chunk);
    }
}

pub(super) fn automatic_deformation_scale(
    topology: &AnalysisRenderTopology,
    field: &FieldData,
) -> Result<f64, String> {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    let mut max_displacement = 0.0_f64;
    for mesh in &topology.meshes {
        for (vertex_index, position) in mesh.vertices.iter().enumerate() {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(position[axis]);
                maximum[axis] = maximum[axis].max(position[axis]);
            }
            if let Some(node) = mesh
                .vertex_volume_node_indices
                .get(vertex_index)
                .copied()
                .flatten()
            {
                let start = node
                    .checked_mul(field.components)
                    .ok_or("displacement index overflow")?;
                let values = field.range(start, 3)?;
                let magnitude = values.iter().map(|value| value * value).sum::<f64>().sqrt();
                if !magnitude.is_finite() {
                    return Err("displacement contains a non-finite value".into());
                }
                max_displacement = max_displacement.max(magnitude);
            }
        }
    }
    let diagonal = (0..3)
        .map(|axis| (maximum[axis] - minimum[axis]).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(super::deformation::automatic_scale(
        diagonal,
        max_displacement,
        field.normalized,
    ))
}

pub(super) fn deform_position(
    original: [f32; 3],
    node: Option<usize>,
    field: Option<&FieldData>,
    scale: f64,
) -> Result<[f32; 3], String> {
    let (Some(node), Some(field)) = (node, field) else {
        return Ok(original);
    };
    let start = node
        .checked_mul(field.components)
        .ok_or("displacement index overflow")?;
    let values = field.range(start, 3)?;
    Ok([0, 1, 2].map(|axis| original[axis] + (values[axis] * scale) as f32))
}

pub(super) fn field_magnitude(
    field: &FieldData,
    entity: usize,
    stride: usize,
) -> Result<f64, String> {
    let mut total = 0.0;
    for sample in 0..stride {
        let start = entity
            .checked_mul(stride)
            .and_then(|value| value.checked_add(sample))
            .and_then(|value| value.checked_mul(field.components))
            .ok_or("field index overflow")?;
        let values = field.range(start, field.components)?;
        total += if field.components == 1 {
            values[0]
        } else {
            values.iter().map(|value| value * value).sum::<f64>().sqrt()
        };
    }
    let magnitude = total / stride as f64;
    if !magnitude.is_finite() {
        return Err(format!("field '{}' has a non-finite value", field.label));
    }
    Ok(magnitude)
}

pub(super) fn scalar_color(value: f64, minimum: f64, maximum: f64) -> [f32; 4] {
    let t = ((value - minimum) / (maximum - minimum).max(f64::EPSILON)).clamp(0.0, 1.0) as f32;
    let stops = [
        [0.10, 0.20, 0.72],
        [0.02, 0.68, 0.90],
        [0.12, 0.82, 0.48],
        [0.98, 0.82, 0.12],
        [0.88, 0.10, 0.12],
    ];
    let index = (t * 4.0).floor().min(3.0) as usize;
    let fraction = t * 4.0 - index as f32;
    let mut color = [0.0; 4];
    for axis in 0..3 {
        color[axis] = stops[index][axis] * (1.0 - fraction) + stops[index + 1][axis] * fraction;
    }
    color[3] = 1.0;
    color
}
