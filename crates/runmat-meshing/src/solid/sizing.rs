use runmat_geometry_core::GeometryAsset;
use runmat_meshing_cad::SourceTopologyModel;
use runmat_meshing_core::{
    AnisotropicSizingSample, MeshSizingField, MeshTargetSize, SizingSampleApplication,
    SizingSampleRejection, VolumeMeshingOptions,
};
use runmat_meshing_curve::CurveDiscretizationOptions;

pub(super) fn target_curve_size_m(options: &VolumeMeshingOptions, geometry: &GeometryAsset) -> f64 {
    match options.target_size {
        MeshTargetSize::LengthM(length) if length.is_finite() && length > 0.0 => length,
        MeshTargetSize::Auto => geometry_span_m(geometry).unwrap_or(1.0),
        _ => 0.05,
    }
    .max(options.min_size_m.unwrap_or(f64::EPSILON))
    .min(options.max_size_m.unwrap_or(f64::INFINITY))
}

pub(super) fn sizing_with_curve_application_evidence(
    sizing: &MeshSizingField,
    topology: &SourceTopologyModel,
    options: CurveDiscretizationOptions,
) -> MeshSizingField {
    let mut enriched = sizing.clone();
    append_curve_sample_application_evidence(&mut enriched, topology, options);
    enriched
}

fn append_curve_sample_application_evidence(
    sizing: &mut MeshSizingField,
    topology: &SourceTopologyModel,
    options: CurveDiscretizationOptions,
) {
    for sample in sizing.samples.clone() {
        let Some(target_size_m) = sizing.clamped_target_size_m(sample.target_size_m) else {
            sizing.rejected_samples.push(sizing_rejection(
                sample.position_m,
                sample.target_size_m,
                sample.reason,
                "skipped_invalid",
                "sizing sample target size was not finite and positive after bounds were applied",
            ));
            continue;
        };
        append_sample_curve_evidence(
            sizing,
            topology,
            options,
            sample.position_m,
            target_size_m,
            sample.reason,
        );
    }

    for sample in sizing.anisotropic_samples.clone() {
        let Some(target_size_m) = anisotropic_sample_target_size(&sample, sizing) else {
            sizing.rejected_samples.push(sizing_rejection(
                sample.position_m,
                sample
                    .target_sizes_m
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
                sample.reason,
                "skipped_invalid",
                "anisotropic sizing sample did not define a valid metric",
            ));
            continue;
        };
        append_sample_curve_evidence(
            sizing,
            topology,
            options,
            sample.position_m,
            target_size_m,
            sample.reason,
        );
    }
}

fn append_sample_curve_evidence(
    sizing: &mut MeshSizingField,
    topology: &SourceTopologyModel,
    options: CurveDiscretizationOptions,
    position_m: [f64; 3],
    target_size_m: f64,
    reason: Option<String>,
) {
    if !position_m.iter().all(|value| value.is_finite()) {
        sizing.rejected_samples.push(sizing_rejection(
            position_m,
            target_size_m,
            reason,
            "skipped_invalid",
            "sizing sample position contained a non-finite coordinate",
        ));
        return;
    }

    let mut matched_edge_count = 0_usize;
    let mut inserted_breakpoint_count = 0_usize;
    for edge in &topology.edges {
        let Some(left) = topology
            .vertices
            .get(edge.node_ids[0] as usize)
            .filter(|vertex| vertex.vertex_id == edge.node_ids[0])
            .map(|vertex| vertex.coordinates_m)
        else {
            continue;
        };
        let Some(right) = topology
            .vertices
            .get(edge.node_ids[1] as usize)
            .filter(|vertex| vertex.vertex_id == edge.node_ids[1])
            .map(|vertex| vertex.coordinates_m)
        else {
            continue;
        };
        if !point_influences_segment(position_m, target_size_m, left, right) {
            continue;
        }
        matched_edge_count += 1;
        let baseline_segment_count =
            edge_segment_count(edge.length_m, options.target_size_m, options);
        let refined_segment_count = edge_segment_count(edge.length_m, target_size_m, options);
        inserted_breakpoint_count += refined_segment_count.saturating_sub(baseline_segment_count);
    }

    if inserted_breakpoint_count > 0 {
        sizing.applied_samples.push(SizingSampleApplication {
            position_m,
            target_size_m,
            inserted_breakpoint_count,
            reason,
            detail: Some(format!(
                "inserted {inserted_breakpoint_count} protected-edge sizing breakpoints across {matched_edge_count} source edges"
            )),
        });
    } else if matched_edge_count > 0 {
        sizing.rejected_samples.push(sizing_rejection(
            position_m,
            target_size_m,
            reason,
            "skipped_noop",
            "sizing sample did not reduce any protected-edge segment count",
        ));
    } else {
        sizing.rejected_samples.push(sizing_rejection(
            position_m,
            target_size_m,
            reason,
            "skipped_out_of_influence",
            "sizing sample did not influence any protected source edge",
        ));
    }
}

fn geometry_span_m(geometry: &GeometryAsset) -> Option<f64> {
    let vertices = geometry
        .surface_meshes
        .iter()
        .flat_map(|mesh| mesh.vertices.iter().copied());
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut count = 0_usize;
    for vertex in vertices {
        count += 1;
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    (count > 0).then(|| {
        (0..3)
            .map(|axis| max[axis] - min[axis])
            .fold(0.0_f64, f64::max)
    })
}

fn anisotropic_sample_target_size(
    sample: &AnisotropicSizingSample,
    sizing: &MeshSizingField,
) -> Option<f64> {
    if !sample.is_valid_metric() {
        return None;
    }
    sizing.clamped_target_size_m(
        sample
            .target_sizes_m
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
    )
}

fn sizing_rejection(
    position_m: [f64; 3],
    target_size_m: f64,
    reason: Option<String>,
    status: &str,
    detail: &str,
) -> SizingSampleRejection {
    SizingSampleRejection {
        position_m,
        target_size_m,
        status: status.to_string(),
        reason,
        detail: Some(detail.to_string()),
    }
}

fn edge_segment_count(
    edge_length_m: f64,
    target_size_m: f64,
    options: CurveDiscretizationOptions,
) -> usize {
    ((edge_length_m / target_size_m).ceil() as usize)
        .max(options.min_segments_per_edge)
        .min(options.max_segments_per_edge)
}

fn point_influences_segment(
    point: [f64; 3],
    target_size_m: f64,
    left: [f64; 3],
    right: [f64; 3],
) -> bool {
    if !point.iter().all(|value| value.is_finite())
        || !left.iter().all(|value| value.is_finite())
        || !right.iter().all(|value| value.is_finite())
        || !target_size_m.is_finite()
        || target_size_m <= 0.0
    {
        return false;
    }
    let segment = sub(right, left);
    let segment_length_squared = dot(segment, segment);
    if segment_length_squared <= f64::EPSILON {
        return distance(point, left) <= 1.0e-12;
    }
    let relative = sub(point, left);
    let parameter = dot(relative, segment) / segment_length_squared;
    if !(-1.0e-12..=1.0 + 1.0e-12).contains(&parameter) {
        return false;
    }
    let parameter = parameter.clamp(0.0, 1.0);
    let closest = [
        left[0] + segment[0] * parameter,
        left[1] + segment[1] * parameter,
        left[2] + segment[2] * parameter,
    ];
    let tolerance = segment_length_squared.sqrt().max(1.0) * 1.0e-9;
    distance(point, closest) <= target_size_m.max(tolerance).max(1.0e-12)
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
}
