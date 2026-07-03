use super::*;

#[cfg(test)]
pub(super) fn constrained_recovery_topology(topology: &SourceTopologyModel) -> bool {
    topology_span_aspect_ratio(topology) >= 6.0
        || topology.vertices.len() > 96
        || topology.faces.len() > 48
}

pub(super) fn thin_low_face_topology(topology: &SourceTopologyModel) -> bool {
    topology_span_aspect_ratio(topology) >= 6.0 && topology.faces.len() <= 48
}

fn topology_span_aspect_ratio(topology: &SourceTopologyModel) -> f64 {
    let spans = (0..3)
        .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
        .collect::<Vec<_>>();
    let max_span = spans.iter().copied().fold(0.0_f64, f64::max);
    let min_span = spans
        .iter()
        .copied()
        .filter(|span| span.is_finite() && *span > MeshingTolerance::default().absolute_m)
        .fold(f64::INFINITY, f64::min);
    if max_span.is_finite() && min_span.is_finite() {
        max_span / min_span
    } else {
        1.0
    }
}

pub(super) fn topology_min_span(topology: &SourceTopologyModel) -> Option<f64> {
    (0..3)
        .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
        .filter(|span| span.is_finite() && *span > MeshingTolerance::default().absolute_m)
        .min_by(|left, right| left.total_cmp(right))
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct RequestedRefinementSelection {
    pub(super) points: [[f64; 3]; 16],
    pub(super) count: usize,
    pub(super) sample_ids: BTreeMap<usize, usize>,
}

pub(super) fn requested_refinement_selection(
    topology: &SourceTopologyModel,
    sizing: Option<&MeshSizingField>,
) -> RequestedRefinementSelection {
    let Some(sizing) = sizing else {
        return RequestedRefinementSelection {
            points: [[0.0; 3]; 16],
            count: 0,
            sample_ids: BTreeMap::new(),
        };
    };
    let surface_triangles = topology_surface_triangles(topology);
    let tolerance = MeshingTolerance::from_bounds(topology.bounds_min_m, topology.bounds_max_m);
    let mut points = [[0.0; 3]; 16];
    let mut count = 0_usize;
    let mut sample_ids = BTreeMap::<usize, usize>::new();
    for (sample_index, sample) in sizing.samples.iter().enumerate() {
        if !sample.target_size_m.is_finite()
            || sample.target_size_m <= 0.0
            || sample.position_m.iter().any(|value| !value.is_finite())
            || soft_generated_cad_sizing_sample(sample)
        {
            continue;
        }
        let candidate_point = if structural_boundary_requested_sample(sample) {
            Some(sample.position_m)
        } else {
            feasible_requested_refinement_point(
                sample.position_m,
                sample.target_size_m,
                topology,
                &surface_triangles,
                tolerance,
            )
        };
        let Some(point) = candidate_point else {
            continue;
        };
        if points[..count]
            .iter()
            .any(|existing| distance_squared(*existing, point) <= 1.0e-24)
        {
            continue;
        }
        sample_ids.insert(sample_index, count);
        points[count] = point;
        count += 1;
        if count >= points.len() {
            break;
        }
    }
    for sample in &sizing.anisotropic_samples {
        if !sample.is_valid_metric()
            || points[..count]
                .iter()
                .any(|point| distance_squared(*point, sample.position_m) <= 1.0e-24)
        {
            continue;
        }
        let Some(point) = feasible_requested_refinement_point(
            sample.position_m,
            sample
                .target_sizes_m
                .into_iter()
                .fold(f64::INFINITY, f64::min),
            topology,
            &surface_triangles,
            tolerance,
        ) else {
            continue;
        };
        points[count] = point;
        count += 1;
        if count >= points.len() {
            break;
        }
    }
    RequestedRefinementSelection {
        points,
        count,
        sample_ids,
    }
}

fn topology_surface_triangles(topology: &SourceTopologyModel) -> Vec<Triangle3> {
    topology
        .faces
        .iter()
        .filter_map(|face| topology_face_points(topology, face.node_ids))
        .collect()
}

fn structural_boundary_requested_sample(sample: &SizingSample) -> bool {
    matches!(
        sample.reason.as_deref(),
        Some("structural.load_regions" | "structural.constraint_regions")
    )
}

fn soft_generated_cad_sizing_sample(sample: &SizingSample) -> bool {
    matches!(
        sample.reason.as_deref(),
        Some("cad.feature_edge" | "cad.interface" | "cad.proximity")
    )
}

fn feasible_requested_refinement_point(
    point: [f64; 3],
    target_size_m: f64,
    topology: &SourceTopologyModel,
    surface_triangles: &[Triangle3],
    tolerance: MeshingTolerance,
) -> Option<[f64; 3]> {
    match point_in_closed_triangle_surface(point, surface_triangles, tolerance) {
        PointInClosedSurface::Inside => Some(point),
        PointInClosedSurface::Outside => None,
        PointInClosedSurface::OnBoundary => inward_requested_refinement_point(
            point,
            target_size_m,
            topology,
            surface_triangles,
            tolerance,
        ),
    }
}

fn inward_requested_refinement_point(
    point: [f64; 3],
    target_size_m: f64,
    topology: &SourceTopologyModel,
    surface_triangles: &[Triangle3],
    tolerance: MeshingTolerance,
) -> Option<[f64; 3]> {
    let step = target_size_m
        .abs()
        .max(tolerance.absolute_m * 100.0)
        .min(topology_characteristic_span(topology).unwrap_or(1.0) * 0.05);
    let nudges = topology
        .faces
        .iter()
        .filter_map(|face| {
            let triangle = topology_face_points(topology, face.node_ids)?;
            (point_triangle_distance(point, triangle) <= tolerance.absolute_m * 10.0)
                .then_some(face.unit_normal)
        })
        .flat_map(|normal| {
            [0.01, 0.05, 0.10, 0.25]
                .into_iter()
                .flat_map(move |fraction| [(-fraction, normal), (fraction, normal)])
        });
    for (fraction, normal) in nudges {
        let candidate = [
            point[0] + normal[0] * step * fraction,
            point[1] + normal[1] * step * fraction,
            point[2] + normal[2] * step * fraction,
        ];
        if matches!(
            point_in_closed_triangle_surface(candidate, surface_triangles, tolerance),
            PointInClosedSurface::Inside
        ) {
            return Some(candidate);
        }
    }
    None
}

fn topology_characteristic_span(topology: &SourceTopologyModel) -> Option<f64> {
    let span = (0..3)
        .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(0.0_f64, f64::max);
    (span.is_finite() && span > 0.0).then_some(span)
}

fn topology_face_points(topology: &SourceTopologyModel, node_ids: [u32; 3]) -> Option<Triangle3> {
    Some([
        topology
            .vertices
            .get(node_ids[0] as usize)
            .filter(|vertex| vertex.vertex_id == node_ids[0])?
            .coordinates_m,
        topology
            .vertices
            .get(node_ids[1] as usize)
            .filter(|vertex| vertex.vertex_id == node_ids[1])?
            .coordinates_m,
        topology
            .vertices
            .get(node_ids[2] as usize)
            .filter(|vertex| vertex.vertex_id == node_ids[2])?
            .coordinates_m,
    ])
}

pub(super) fn solid_effective_sizing(
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
) -> Option<MeshSizingField> {
    let mut effective = sizing.cloned().unwrap_or_default();
    for sample in anisotropic_equivalent_sizing_samples(&effective.anisotropic_samples) {
        if effective
            .samples
            .iter()
            .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
        {
            continue;
        }
        effective.samples.push(sample);
    }
    if options.refinement.focus.curvature {
        let base_target_size_m = effective
            .global_target_size_m
            .filter(|value| value.is_finite() && *value > 0.0)
            .or(match options.target_size {
                MeshTargetSize::LengthM(length_m) if length_m.is_finite() && length_m > 0.0 => {
                    Some(clamp_mesh_target_size(length_m, options))
                }
                MeshTargetSize::LengthM(_) | MeshTargetSize::Auto => None,
            });
        for sample in cad_curvature_sizing_samples(cad_evaluation, base_target_size_m) {
            if effective
                .samples
                .iter()
                .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
            {
                continue;
            }
            effective.samples.push(sample);
        }
    }
    if options.refinement.focus.small_features {
        for sample in cad_feature_edge_sizing_samples(topology) {
            if effective
                .samples
                .iter()
                .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
            {
                continue;
            }
            effective.samples.push(sample);
        }
        for sample in cad_proximity_sizing_samples(topology) {
            if effective
                .samples
                .iter()
                .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
            {
                continue;
            }
            effective.samples.push(sample);
        }
    }
    for sample in cad_interface_sizing_samples(topology, options.refinement.focus.interfaces) {
        if effective
            .samples
            .iter()
            .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
        {
            continue;
        }
        effective.samples.push(sample);
    }
    if sizing.is_some() || !effective.samples.is_empty() {
        Some(effective)
    } else {
        None
    }
}

fn anisotropic_equivalent_sizing_samples(samples: &[AnisotropicSizingSample]) -> Vec<SizingSample> {
    samples
        .iter()
        .filter(|sample| sample.is_valid_metric())
        .filter_map(|sample| {
            let target_size_m = sample
                .target_sizes_m
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            (target_size_m.is_finite() && target_size_m > 0.0).then_some(SizingSample {
                position_m: sample.position_m,
                target_size_m,
                reason: sample
                    .reason
                    .clone()
                    .or_else(|| Some("anisotropic.metric".to_string())),
            })
        })
        .collect()
}

fn cad_curvature_sizing_samples(
    cad_evaluation: &CadEvaluationModel,
    base_target_size_m: Option<f64>,
) -> Vec<SizingSample> {
    cad_evaluation
        .face_frames
        .iter()
        .filter_map(|frame| {
            let curvature = frame.max_curvature_estimate_1_per_m?;
            if !curvature.is_finite() || curvature <= 0.0 {
                return None;
            }
            let radius_m = 1.0 / curvature;
            if !radius_m.is_finite() || radius_m <= 0.0 {
                return None;
            }
            let mut target_size_m = radius_m * 0.25;
            if let Some(base_target_size_m) =
                base_target_size_m.filter(|value| value.is_finite() && *value > 0.0)
            {
                target_size_m = target_size_m.min(base_target_size_m);
            }
            (target_size_m.is_finite() && target_size_m > 0.0).then_some(SizingSample {
                position_m: frame.origin_m,
                target_size_m,
                reason: Some("cad.curvature".to_string()),
            })
        })
        .collect()
}

fn cad_feature_edge_sizing_samples(topology: &SourceTopologyModel) -> Vec<SizingSample> {
    let max_span = (0..3)
        .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
        .fold(0.0_f64, f64::max);
    if !max_span.is_finite() || max_span <= 0.0 {
        return Vec::new();
    }
    let threshold_m = max_span * 0.35;
    topology
        .edges
        .iter()
        .filter_map(|edge| {
            if !edge.length_m.is_finite() || edge.length_m <= 0.0 || edge.length_m > threshold_m {
                return None;
            }
            let left = topology
                .vertices
                .get(edge.node_ids[0] as usize)
                .filter(|vertex| vertex.vertex_id == edge.node_ids[0])?;
            let right = topology
                .vertices
                .get(edge.node_ids[1] as usize)
                .filter(|vertex| vertex.vertex_id == edge.node_ids[1])?;
            Some(SizingSample {
                position_m: [
                    (left.coordinates_m[0] + right.coordinates_m[0]) * 0.5,
                    (left.coordinates_m[1] + right.coordinates_m[1]) * 0.5,
                    (left.coordinates_m[2] + right.coordinates_m[2]) * 0.5,
                ],
                target_size_m: edge.length_m * 0.5,
                reason: Some("cad.feature_edge".to_string()),
            })
        })
        .collect()
}

fn cad_interface_sizing_samples(
    topology: &SourceTopologyModel,
    focus: RefinementFocusLevel,
) -> Vec<SizingSample> {
    let target_fraction = match focus {
        RefinementFocusLevel::Off => return Vec::new(),
        RefinementFocusLevel::Normal => 0.5,
        RefinementFocusLevel::Fine => 0.25,
    };
    topology
        .edges
        .iter()
        .filter_map(|edge| {
            if edge.region_ids.len() < 2 || !edge.length_m.is_finite() || edge.length_m <= 0.0 {
                return None;
            }
            let left = topology
                .vertices
                .get(edge.node_ids[0] as usize)
                .filter(|vertex| vertex.vertex_id == edge.node_ids[0])?;
            let right = topology
                .vertices
                .get(edge.node_ids[1] as usize)
                .filter(|vertex| vertex.vertex_id == edge.node_ids[1])?;
            Some(SizingSample {
                position_m: [
                    (left.coordinates_m[0] + right.coordinates_m[0]) * 0.5,
                    (left.coordinates_m[1] + right.coordinates_m[1]) * 0.5,
                    (left.coordinates_m[2] + right.coordinates_m[2]) * 0.5,
                ],
                target_size_m: edge.length_m * target_fraction,
                reason: Some("cad.interface".to_string()),
            })
        })
        .collect()
}

fn cad_proximity_sizing_samples(topology: &SourceTopologyModel) -> Vec<SizingSample> {
    let max_span = (0..3)
        .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
        .fold(0.0_f64, f64::max);
    if !max_span.is_finite() || max_span <= 0.0 {
        return Vec::new();
    }
    let threshold_m = max_span * 0.35;
    let mut samples = Vec::<SizingSample>::new();
    for left_index in 0..topology.faces.len() {
        let left = &topology.faces[left_index];
        let Some(left_centroid) = topology_face_centroid(topology, left.node_ids) else {
            continue;
        };
        for right in topology.faces.iter().skip(left_index + 1) {
            if left
                .node_ids
                .iter()
                .any(|node_id| right.node_ids.contains(node_id))
            {
                continue;
            }
            if dot(left.unit_normal, right.unit_normal) > -0.75 {
                continue;
            }
            let Some(right_centroid) = topology_face_centroid(topology, right.node_ids) else {
                continue;
            };
            let gap_m = distance_squared(left_centroid, right_centroid).sqrt();
            if !gap_m.is_finite() || gap_m <= 0.0 || gap_m > threshold_m {
                continue;
            }
            samples.push(SizingSample {
                position_m: [
                    (left_centroid[0] + right_centroid[0]) * 0.5,
                    (left_centroid[1] + right_centroid[1]) * 0.5,
                    (left_centroid[2] + right_centroid[2]) * 0.5,
                ],
                target_size_m: gap_m * 0.5,
                reason: Some("cad.proximity".to_string()),
            });
        }
    }
    samples
}

fn topology_face_centroid(topology: &SourceTopologyModel, node_ids: [u32; 3]) -> Option<[f64; 3]> {
    let a = topology
        .vertices
        .get(node_ids[0] as usize)
        .filter(|vertex| vertex.vertex_id == node_ids[0])?
        .coordinates_m;
    let b = topology
        .vertices
        .get(node_ids[1] as usize)
        .filter(|vertex| vertex.vertex_id == node_ids[1])?
        .coordinates_m;
    let c = topology
        .vertices
        .get(node_ids[2] as usize)
        .filter(|vertex| vertex.vertex_id == node_ids[2])?
        .coordinates_m;
    Some([
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ])
}

pub(super) fn target_size_for_mesh(
    topology: &SourceTopologyModel,
    options: &VolumeMeshingOptions,
) -> f64 {
    let target_size_m = match options.target_size {
        MeshTargetSize::LengthM(length_m) if length_m.is_finite() && length_m > 0.0 => length_m,
        MeshTargetSize::LengthM(_) | MeshTargetSize::Auto => {
            let span = (0..3)
                .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
                .fold(0.0_f64, f64::max);
            (span / 20.0).max(1.0e-6)
        }
    };
    clamp_mesh_target_size(target_size_m, options)
}

pub(super) fn solid_sizing_target_size(
    base_target_size_m: f64,
    sizing: &MeshSizingField,
    options: &VolumeMeshingOptions,
    topology: Option<&SourceTopologyModel>,
) -> f64 {
    let mut target_size_m = base_target_size_m;
    let mut global_target_size_m = base_target_size_m;
    if let Some(candidate) = sizing.global_target_size_m {
        if candidate.is_finite() && candidate > 0.0 {
            target_size_m = candidate;
            global_target_size_m = candidate;
        }
    }
    for sample in &sizing.samples {
        if sample_drives_global_target_size(sample, topology)
            && sample.target_size_m.is_finite()
            && sample.target_size_m > 0.0
        {
            let sample_target_size_m = solid_sample_target_size(
                sample.target_size_m,
                Some(global_target_size_m),
                sizing.growth_rate.or(options.growth_rate),
                sizing.min_size_m.or(options.min_size_m),
                sizing.max_size_m.or(options.max_size_m),
            );
            target_size_m = target_size_m.min(sample_target_size_m);
        }
    }
    for sample in anisotropic_equivalent_sizing_samples(&sizing.anisotropic_samples) {
        let sample_target_size_m = solid_sample_target_size(
            sample.target_size_m,
            Some(global_target_size_m),
            sizing.growth_rate.or(options.growth_rate),
            sizing.min_size_m.or(options.min_size_m),
            sizing.max_size_m.or(options.max_size_m),
        );
        target_size_m = target_size_m.min(sample_target_size_m);
    }
    if let Some(min_size_m) = sizing.min_size_m.or(options.min_size_m) {
        if min_size_m.is_finite() && min_size_m > 0.0 {
            target_size_m = target_size_m.max(min_size_m);
        }
    }
    if let Some(max_size_m) = sizing.max_size_m.or(options.max_size_m) {
        if max_size_m.is_finite() && max_size_m > 0.0 {
            target_size_m = target_size_m.min(max_size_m);
        }
    }
    target_size_m.max(1.0e-9)
}

fn sample_drives_global_target_size(
    sample: &SizingSample,
    topology: Option<&SourceTopologyModel>,
) -> bool {
    if matches!(
        sample.reason.as_deref(),
        Some("structural.load_regions" | "structural.constraint_regions")
    ) && topology.is_some_and(|topology| point_on_topology_boundary(sample.position_m, topology))
    {
        return false;
    }
    true
}

fn point_on_topology_boundary(point_m: [f64; 3], topology: &SourceTopologyModel) -> bool {
    let tolerance = MeshingTolerance::from_bounds(topology.bounds_min_m, topology.bounds_max_m);
    (0..3).any(|axis| {
        (point_m[axis] - topology.bounds_min_m[axis]).abs() <= tolerance.absolute_m
            || (point_m[axis] - topology.bounds_max_m[axis]).abs() <= tolerance.absolute_m
    })
}

fn clamp_mesh_target_size(mut value: f64, options: &VolumeMeshingOptions) -> f64 {
    if let Some(min_size_m) = options.min_size_m {
        value = value.max(min_size_m);
    }
    if let Some(max_size_m) = options.max_size_m {
        value = value.min(max_size_m);
    }
    value
}

pub(super) fn solid_mesh_sizing(
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
    preparation: &SolidMeshPreparation,
) -> MeshSizingField {
    let mut mesh_sizing = sizing.cloned().unwrap_or_default();
    if mesh_sizing.global_target_size_m.is_none() {
        mesh_sizing.global_target_size_m = match options.target_size {
            MeshTargetSize::LengthM(length) => Some(clamp_mesh_target_size(length, options)),
            MeshTargetSize::Auto => None,
        };
    }
    if mesh_sizing.min_size_m.is_none() {
        mesh_sizing.min_size_m = options.min_size_m;
    }
    if mesh_sizing.max_size_m.is_none() {
        mesh_sizing.max_size_m = options.max_size_m;
    }
    if mesh_sizing.growth_rate.is_none() {
        mesh_sizing.growth_rate = options.growth_rate;
    }
    if let Some(sizing) = sizing {
        let mut seen_positions = Vec::<[f64; 3]>::new();
        let requested_sample_ids = requested_sizing_sample_ids(&preparation.topology, sizing);
        mesh_sizing.applied_samples.clear();
        mesh_sizing.rejected_samples.clear();
        for (sample_index, sample) in sizing.samples.iter().enumerate() {
            let valid_position = sample.position_m.iter().all(|value| value.is_finite());
            let valid_size = sample.target_size_m.is_finite() && sample.target_size_m > 0.0;
            if !valid_position || !valid_size {
                mesh_sizing.rejected_samples.push(SizingSampleRejection {
                    position_m: sample.position_m,
                    target_size_m: sample.target_size_m,
                    status: "skipped_invalid".to_string(),
                    reason: sample.reason.clone(),
                    detail: Some("sample position and target size must be finite".to_string()),
                });
                continue;
            }
            let target_size_m = solid_sample_target_size(
                sample.target_size_m,
                mesh_sizing.global_target_size_m,
                mesh_sizing.growth_rate,
                mesh_sizing.min_size_m,
                mesh_sizing.max_size_m,
            );
            if seen_positions
                .iter()
                .any(|position| distance_squared(*position, sample.position_m) <= 1.0e-24)
            {
                mesh_sizing.rejected_samples.push(SizingSampleRejection {
                    position_m: sample.position_m,
                    target_size_m: sample.target_size_m,
                    status: "skipped_duplicate".to_string(),
                    reason: sample.reason.clone(),
                    detail: Some("sample position was already represented".to_string()),
                });
                continue;
            }
            seen_positions.push(sample.position_m);
            let requested_id = requested_sample_ids.get(&sample_index).copied();
            let inserted_breakpoint_count = usize::from(
                requested_id.is_some()
                    && tetrahedron_mesh_has_node_near(
                        &preparation.solver_tetrahedron_mesh,
                        sample.position_m,
                    ),
            );
            if inserted_breakpoint_count > 0 {
                mesh_sizing.applied_samples.push(SizingSampleApplication {
                    position_m: sample.position_m,
                    target_size_m,
                    inserted_breakpoint_count,
                    reason: sample.reason.clone(),
                    detail: Some("solid_requested_tetrahedron_point_present".to_string()),
                });
            } else if requested_id.is_some() {
                mesh_sizing.rejected_samples.push(SizingSampleRejection {
                    position_m: sample.position_m,
                    target_size_m,
                    status: "not_inserted_by_tetrahedron_generation".to_string(),
                    reason: sample.reason.clone(),
                    detail: Some(
                        "native Tetrahedron generation does not insert requested points yet"
                            .to_string(),
                    ),
                });
            } else {
                mesh_sizing.rejected_samples.push(SizingSampleRejection {
                    position_m: sample.position_m,
                    target_size_m,
                    status: "skipped_budget".to_string(),
                    reason: sample.reason.clone(),
                    detail: Some("requested Tetrahedron seed budget was exhausted".to_string()),
                });
            }
        }
    }
    mesh_sizing
}

pub(super) fn requested_sizing_sample_ids(
    topology: &SourceTopologyModel,
    sizing: &MeshSizingField,
) -> BTreeMap<usize, usize> {
    requested_refinement_selection(topology, Some(sizing)).sample_ids
}

fn solid_sample_target_size(
    mut target_size_m: f64,
    global_target_size_m: Option<f64>,
    growth_rate: Option<f64>,
    min_size_m: Option<f64>,
    max_size_m: Option<f64>,
) -> f64 {
    if let (Some(global_target_size_m), Some(growth_rate)) = (
        global_target_size_m.filter(|value| value.is_finite() && *value > 0.0),
        growth_rate.filter(|value| value.is_finite() && *value >= 1.0),
    ) {
        target_size_m = target_size_m.max(global_target_size_m / growth_rate);
    }
    if let Some(min_size_m) = min_size_m.filter(|value| value.is_finite() && *value > 0.0) {
        target_size_m = target_size_m.max(min_size_m);
    }
    if let Some(max_size_m) = max_size_m.filter(|value| value.is_finite() && *value > 0.0) {
        target_size_m = target_size_m.min(max_size_m);
    }
    target_size_m
}
