use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    candidate::strictly_inside, candidate_identity, error, quality_error, validate_options,
    CandidateWork, DelaunayRefinementCandidateKind, DelaunayVolumeRefinementCandidate,
    DelaunayVolumeRefinementCandidateError, DelaunayVolumeRefinementCandidateErrorKind,
    DelaunayVolumeRefinementCandidateOptions, DelaunayVolumeRefinementInput,
};

pub fn validate_delaunay_volume_refinement_candidate(
    input: DelaunayVolumeRefinementInput<'_>,
    candidate: &Option<DelaunayVolumeRefinementCandidate>,
    options: DelaunayVolumeRefinementCandidateOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeRefinementCandidateError> {
    validate_options(options)?;
    super::validate_delaunay_volume_quality(
        input.topology,
        input.metric_request,
        input.metric_contexts,
        input.quality,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)?;
    let Some(source_identity) = input.quality.worst_refinement_tetrahedron else {
        if candidate.is_some() {
            return Err(invalid("converged quality must not carry a candidate"));
        }
        return Ok(());
    };
    let observed = candidate
        .as_ref()
        .ok_or_else(|| invalid("violating quality requires one refinement candidate"))?;
    let source_index = input
        .quality
        .tetrahedra
        .iter()
        .position(|tetrahedron| tetrahedron.node_identities == source_identity)
        .ok_or_else(|| invalid("candidate source is absent from quality evidence"))?;
    let source_quality = &input.quality.tetrahedra[source_index];
    let topology_index = input
        .topology
        .tetrahedra
        .iter()
        .position(|tetrahedron| {
            tetrahedron
                .vertex_indices
                .map(|vertex| input.topology.nodes[vertex as usize].identity)
                == source_identity
        })
        .ok_or_else(|| invalid("candidate source is absent from topology"))?;
    let points = input.topology.tetrahedra[topology_index]
        .vertex_indices
        .map(|vertex| input.topology.nodes[vertex as usize].coordinates_m);
    let mut work = CandidateWork::new(options, cancellation);
    work.evaluate()?;
    let transformed_center =
        transformed_metric_circumcenter(points, source_quality.resolved_metric);
    let expected_kind = match transformed_center {
        Some(center) if strictly_inside(points, center)? => {
            DelaunayRefinementCandidateKind::MetricCircumcenter
        }
        _ => DelaunayRefinementCandidateKind::InteriorCentroid,
    };
    if expected_kind == DelaunayRefinementCandidateKind::InteriorCentroid {
        work.evaluate()?;
    }
    let expected_coordinates = match expected_kind {
        DelaunayRefinementCandidateKind::MetricCircumcenter => {
            transformed_center.ok_or_else(|| {
                invalid("independent metric circumcenter classification is inconsistent")
            })?
        }
        DelaunayRefinementCandidateKind::InteriorCentroid => {
            std::array::from_fn(|axis| points.iter().map(|point| point[axis] * 0.25).sum::<f64>())
        }
    };
    if observed.kind != expected_kind
        || observed.source_node_identities != source_identity
        || observed.region_id != source_quality.region_id
        || observed.incident_metric_entity_ids != source_quality.incident_metric_entity_ids
        || observed.resolved_metric != source_quality.resolved_metric
        || observed.source_violation_ratio != source_quality.refinement_violation_ratio
        || !coordinates_match(observed.node.coordinates_m, expected_coordinates)
        || !strictly_inside(points, observed.node.coordinates_m)?
        || observed.node.identity
            != candidate_identity(source_identity, observed.kind, observed.resolved_metric)
        || input.topology.nodes.iter().any(|node| {
            node.identity == observed.node.identity
                || node.coordinates_m == observed.node.coordinates_m
        })
    {
        return Err(invalid(
            "candidate does not match independent interior construction",
        ));
    }
    Ok(())
}

fn transformed_metric_circumcenter(
    points: [[f64; 3]; 4],
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> Option<[f64; 3]> {
    let l00 = metric.xx.sqrt();
    let l10 = metric.xy / l00;
    let l20 = metric.xz / l00;
    let l11 = (metric.yy - l10 * l10).sqrt();
    let l21 = (metric.yz - l20 * l10) / l11;
    let l22 = (metric.zz - l20 * l20 - l21 * l21).sqrt();
    if ![l00, l10, l20, l11, l21, l22]
        .into_iter()
        .all(f64::is_finite)
    {
        return None;
    }
    let transformed = points.map(|point| {
        [
            l00 * point[0] + l10 * point[1] + l20 * point[2],
            l11 * point[1] + l21 * point[2],
            l22 * point[2],
        ]
    });
    let transformed_center = euclidean_circumcenter(transformed)?;
    let z = transformed_center[2] / l22;
    let y = (transformed_center[1] - l21 * z) / l11;
    let x = (transformed_center[0] - l10 * y - l20 * z) / l00;
    [x, y, z]
        .into_iter()
        .all(f64::is_finite)
        .then_some([x, y, z])
}

fn euclidean_circumcenter(points: [[f64; 3]; 4]) -> Option<[f64; 3]> {
    let mut matrix = [[0.0; 3]; 3];
    let mut right = [0.0; 3];
    let norm = |point: [f64; 3]| point.into_iter().map(|value| value * value).sum::<f64>();
    for row in 0..3 {
        for axis in 0..3 {
            matrix[row][axis] = points[row + 1][axis] - points[0][axis];
        }
        right[row] = 0.5 * (norm(points[row + 1]) - norm(points[0]));
    }
    gaussian_solve(matrix, right)
}

fn gaussian_solve(mut matrix: [[f64; 3]; 3], mut right: [f64; 3]) -> Option<[f64; 3]> {
    for pivot in 0..3 {
        let pivot_row = (pivot..3).max_by(|left, right_row| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right_row][pivot].abs())
        })?;
        if matrix[pivot_row][pivot] == 0.0 || !matrix[pivot_row][pivot].is_finite() {
            return None;
        }
        matrix.swap(pivot, pivot_row);
        right.swap(pivot, pivot_row);
        for row in (pivot + 1)..3 {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for column in pivot..3 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            right[row] -= factor * right[pivot];
        }
    }
    let mut result = [0.0; 3];
    for row in (0..3).rev() {
        let remainder = ((row + 1)..3)
            .map(|column| matrix[row][column] * result[column])
            .sum::<f64>();
        result[row] = (right[row] - remainder) / matrix[row][row];
    }
    result.into_iter().all(f64::is_finite).then_some(result)
}

fn coordinates_match(left: [f64; 3], right: [f64; 3]) -> bool {
    left.into_iter().zip(right).all(|(left, right)| {
        left.is_finite()
            && right.is_finite()
            && (left - right).abs()
                <= 512.0 * f64::EPSILON * left.abs().max(right.abs()).max(f64::MIN_POSITIVE)
    })
}

fn invalid(reason: &'static str) -> DelaunayVolumeRefinementCandidateError {
    error(
        DelaunayVolumeRefinementCandidateErrorKind::InvalidCandidate,
        reason,
    )
}
