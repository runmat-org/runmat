use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_size::metric::ResolvedMetricEvaluation;

use super::{
    error, DelaunayTetrahedronQuality, DelaunayVolumeQualityError, DelaunayVolumeQualityErrorKind,
    DelaunayVolumeQualityOptions, DelaunayVolumeTopology,
};

pub(super) fn evaluate_tetrahedron(
    topology: &DelaunayVolumeTopology,
    tetrahedron_index: usize,
    region_id: PersistentEntityId,
    incident_metric_entity_ids: Vec<PersistentEntityId>,
    metric: ResolvedMetricEvaluation,
    options: DelaunayVolumeQualityOptions,
) -> Result<DelaunayTetrahedronQuality, DelaunayVolumeQualityError> {
    let tetrahedron = &topology.tetrahedra[tetrahedron_index];
    let points = tetrahedron
        .vertex_indices
        .map(|vertex| topology.nodes[vertex as usize].coordinates_m);
    let identities = tetrahedron
        .vertex_indices
        .map(|vertex| topology.nodes[vertex as usize].identity);
    let transformed = transform_points(points, metric.metric).ok_or_else(|| {
        error(
            DelaunayVolumeQualityErrorKind::NumericalFailure,
            Some(tetrahedron_index),
            "could not factor the resolved SPD metric",
        )
    })?;
    let (minimum_edge, maximum_edge) = edge_extrema(transformed).ok_or_else(|| {
        error(
            DelaunayVolumeQualityErrorKind::NumericalFailure,
            Some(tetrahedron_index),
            "metric edge lengths must be finite and nonzero",
        )
    })?;
    let radius = circumradius(transformed).ok_or_else(|| {
        error(
            DelaunayVolumeQualityErrorKind::NumericalFailure,
            Some(tetrahedron_index),
            "metric circumsphere could not be evaluated",
        )
    })?;
    let radius_edge_ratio = radius / minimum_edge;
    let violation_ratio = (maximum_edge / options.maximum_metric_edge_length)
        .max(radius_edge_ratio / options.maximum_radius_edge_ratio);
    if !radius_edge_ratio.is_finite() || !violation_ratio.is_finite() {
        return Err(error(
            DelaunayVolumeQualityErrorKind::NumericalFailure,
            Some(tetrahedron_index),
            "metric quality ratios must be finite",
        ));
    }

    Ok(DelaunayTetrahedronQuality {
        node_identities: identities,
        region_id,
        incident_metric_entity_ids,
        resolved_metric: metric.metric,
        active_metric_sources: metric.active_sources,
        applied_metric_contribution_count: metric.applied_contribution_count,
        clipped_metric_contribution_count: metric.clipped_contribution_count,
        rejected_metric_contribution_count: metric.rejected_contribution_count,
        minimum_metric_edge_length: minimum_edge,
        maximum_metric_edge_length: maximum_edge,
        metric_circumradius: radius,
        metric_radius_edge_ratio: radius_edge_ratio,
        refinement_violation_ratio: violation_ratio,
    })
}

fn transform_points(
    points: [[f64; 3]; 4],
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> Option<[[f64; 3]; 4]> {
    let l00 = metric.xx.sqrt();
    let l10 = metric.xy / l00;
    let l20 = metric.xz / l00;
    let l11 = (metric.yy - l10 * l10).sqrt();
    let l21 = (metric.yz - l20 * l10) / l11;
    let l22 = (metric.zz - l20 * l20 - l21 * l21).sqrt();
    [l00, l10, l20, l11, l21, l22]
        .into_iter()
        .all(f64::is_finite)
        .then(|| {
            points.map(|point| {
                [
                    l00 * point[0] + l10 * point[1] + l20 * point[2],
                    l11 * point[1] + l21 * point[2],
                    l22 * point[2],
                ]
            })
        })
}

fn edge_extrema(points: [[f64; 3]; 4]) -> Option<(f64, f64)> {
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for left in 0..4 {
        for right in (left + 1)..4 {
            let squared = (0..3)
                .map(|axis| (points[right][axis] - points[left][axis]).powi(2))
                .sum::<f64>();
            let length = squared.sqrt();
            if !length.is_finite() || length <= 0.0 {
                return None;
            }
            minimum = minimum.min(length);
            maximum = maximum.max(length);
        }
    }
    Some((minimum, maximum))
}

fn circumradius(points: [[f64; 3]; 4]) -> Option<f64> {
    let mut matrix = [[0.0; 3]; 3];
    let mut right = [0.0; 3];
    let base_norm = squared_norm(points[0]);
    for row in 0..3 {
        for axis in 0..3 {
            matrix[row][axis] = points[row + 1][axis] - points[0][axis];
        }
        right[row] = 0.5 * (squared_norm(points[row + 1]) - base_norm);
    }
    let center = solve(matrix, right)?;
    let radius = squared_distance(center, points[0]).sqrt();
    (radius.is_finite() && radius > 0.0).then_some(radius)
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

fn squared_norm(point: [f64; 3]) -> f64 {
    point.into_iter().map(|value| value * value).sum()
}

fn squared_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    (0..3).map(|axis| (left[axis] - right[axis]).powi(2)).sum()
}
