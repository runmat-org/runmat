use runmat_meshing_core::quality::predicate::{orient3d, PredicateSign};
use runmat_meshing_size::metric::MetricTensor3;

use super::{
    candidate_identity, error, CandidateWork, DelaunayRefinementCandidateKind,
    DelaunayTetrahedronQuality, DelaunayVolumeRefinementCandidate,
    DelaunayVolumeRefinementCandidateError, DelaunayVolumeRefinementCandidateErrorKind,
    DelaunayVolumeTopology,
};

pub(super) fn construct_candidate(
    topology: &DelaunayVolumeTopology,
    quality: &DelaunayTetrahedronQuality,
    work: &mut CandidateWork<'_>,
) -> Result<DelaunayVolumeRefinementCandidate, DelaunayVolumeRefinementCandidateError> {
    work.evaluate()?;
    let tetrahedron_index = topology
        .tetrahedra
        .iter()
        .position(|tetrahedron| {
            tetrahedron
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].identity)
                == quality.node_identities
        })
        .ok_or_else(|| {
            error(
                DelaunayVolumeRefinementCandidateErrorKind::InvalidTopology,
                "quality source tetrahedron is absent from topology",
            )
        })?;
    let points = topology.tetrahedra[tetrahedron_index]
        .vertex_indices
        .map(|vertex| topology.nodes[vertex as usize].coordinates_m);
    let circumcenter = generalized_metric_circumcenter(points, quality.resolved_metric);
    let (kind, coordinates_m) = match circumcenter {
        Some(center) if strictly_inside(points, center)? => {
            (DelaunayRefinementCandidateKind::MetricCircumcenter, center)
        }
        _ => {
            work.evaluate()?;
            let centroid = std::array::from_fn(|axis| {
                points.iter().map(|point| point[axis] * 0.25).sum::<f64>()
            });
            if !strictly_inside(points, centroid)? {
                return Err(error(
                    DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure,
                    "interior centroid is not strictly inside its source tetrahedron",
                ));
            }
            (DelaunayRefinementCandidateKind::InteriorCentroid, centroid)
        }
    };
    let identity = candidate_identity(quality.node_identities, kind, quality.resolved_metric);
    if identity == runmat_meshing_core::StableDigest::ZERO
        || topology
            .nodes
            .iter()
            .any(|node| node.identity == identity || node.coordinates_m == coordinates_m)
    {
        return Err(error(
            DelaunayVolumeRefinementCandidateErrorKind::InvalidCandidate,
            "refinement candidate collides with an existing node",
        ));
    }
    Ok(DelaunayVolumeRefinementCandidate {
        node: super::DelaunayVolumeNode {
            identity,
            coordinates_m,
        },
        kind,
        source_node_identities: quality.node_identities,
        region_id: quality.region_id.clone(),
        incident_metric_entity_ids: quality.incident_metric_entity_ids.clone(),
        resolved_metric: quality.resolved_metric,
        source_violation_ratio: quality.refinement_violation_ratio,
    })
}

fn generalized_metric_circumcenter(
    points: [[f64; 3]; 4],
    metric: MetricTensor3,
) -> Option<[f64; 3]> {
    let product = |point: [f64; 3]| {
        [
            metric.xx * point[0] + metric.xy * point[1] + metric.xz * point[2],
            metric.xy * point[0] + metric.yy * point[1] + metric.yz * point[2],
            metric.xz * point[0] + metric.yz * point[1] + metric.zz * point[2],
        ]
    };
    let quadratic = |point: [f64; 3]| {
        let multiplied = product(point);
        point[0] * multiplied[0] + point[1] * multiplied[1] + point[2] * multiplied[2]
    };
    let mut matrix = [[0.0; 3]; 3];
    let mut right = [0.0; 3];
    for row in 0..3 {
        let delta = std::array::from_fn(|axis| points[row + 1][axis] - points[0][axis]);
        matrix[row] = product(delta);
        right[row] = 0.5 * (quadratic(points[row + 1]) - quadratic(points[0]));
    }
    solve(matrix, right)
}

fn solve(matrix: [[f64; 3]; 3], right: [f64; 3]) -> Option<[f64; 3]> {
    let denominator = determinant(matrix);
    if !denominator.is_finite() || denominator == 0.0 {
        return None;
    }
    let mut result = [0.0; 3];
    for (column, value) in result.iter_mut().enumerate() {
        let mut replaced = matrix;
        for row in 0..3 {
            replaced[row][column] = right[row];
        }
        *value = determinant(replaced) / denominator;
    }
    result.into_iter().all(f64::is_finite).then_some(result)
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

pub(super) fn strictly_inside(
    points: [[f64; 3]; 4],
    candidate: [f64; 3],
) -> Result<bool, DelaunayVolumeRefinementCandidateError> {
    for replace in 0..4 {
        let mut simplex = points;
        simplex[replace] = candidate;
        let sign = orient3d(simplex).map_err(|failure| {
            error(
                DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure,
                format!("candidate orientation failed: {failure:?}"),
            )
        })?;
        if sign != PredicateSign::Positive {
            return Ok(false);
        }
    }
    Ok(true)
}
