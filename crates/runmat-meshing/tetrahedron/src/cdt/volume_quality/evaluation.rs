use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::predicate::tetrahedron_scaled_jacobian;
use runmat_meshing_size::metric::ResolvedMetricEvaluation;

use super::{
    error, DelaunayTetrahedronQuality, DelaunayVolumeQualityError, DelaunayVolumeQualityErrorKind,
    DelaunayVolumeQualityOptions, DelaunayVolumeTopology,
};

pub(in crate::cdt) struct DelaunayTetrahedronGeometryQuality {
    pub(in crate::cdt) minimum_metric_edge_length: f64,
    pub(in crate::cdt) maximum_metric_edge_length: f64,
    pub(in crate::cdt) metric_circumradius: f64,
    pub(in crate::cdt) metric_radius_edge_ratio: f64,
    pub(in crate::cdt) metric_scaled_jacobian: f64,
}

pub(super) struct TetrahedronEvaluationContext {
    pub(super) region_id: PersistentEntityId,
    pub(super) incident_metric_entity_ids: Vec<PersistentEntityId>,
    pub(super) metric: ResolvedMetricEvaluation,
    pub(super) boundary_node_count: u8,
    pub(super) protected_facet_count: u8,
}

pub(super) fn evaluate_tetrahedron(
    topology: &DelaunayVolumeTopology,
    tetrahedron_index: usize,
    context: TetrahedronEvaluationContext,
    options: DelaunayVolumeQualityOptions,
) -> Result<DelaunayTetrahedronQuality, DelaunayVolumeQualityError> {
    let TetrahedronEvaluationContext {
        region_id,
        incident_metric_entity_ids,
        metric,
        boundary_node_count,
        protected_facet_count,
    } = context;
    let tetrahedron = &topology.tetrahedra[tetrahedron_index];
    let points = tetrahedron
        .vertex_indices
        .map(|vertex| topology.nodes[vertex as usize].coordinates_m);
    let identities = tetrahedron
        .vertex_indices
        .map(|vertex| topology.nodes[vertex as usize].identity);
    let geometry = evaluate_tetrahedron_geometry(points, metric.metric).ok_or_else(|| {
        error(
            DelaunayVolumeQualityErrorKind::NumericalFailure,
            Some(tetrahedron_index),
            "metric tetrahedron geometry must be finite, nondegenerate, and positively oriented",
        )
    })?;
    let size_violation_ratio = saturated_ratio(
        geometry.maximum_metric_edge_length,
        options.maximum_metric_edge_length,
    );
    let radius_edge_violation_ratio = saturated_ratio(
        geometry.metric_radius_edge_ratio,
        options.maximum_radius_edge_ratio,
    );
    let jacobian_violation_ratio = saturated_ratio(
        options.minimum_metric_scaled_jacobian,
        geometry.metric_scaled_jacobian,
    );
    // Radius-edge refinement is a volumetric Delaunay criterion whose repair
    // point is the circumcenter. A cell touching the immutable published PLC
    // can have that point outside the admitted solid; repairing it would first
    // require a surface-feature split owned by the surface stage. Such cells
    // remain subject to metric size and scaled-Jacobian shape quality. Only a
    // wholly interior cell receives the volume circumcenter criterion here.
    let interior_radius_violation = if radius_edge_is_refinement_owned(boundary_node_count) {
        radius_edge_violation_ratio
    } else {
        0.0
    };
    let refinement_violation_ratio = size_violation_ratio.max(interior_radius_violation);
    let optimization_violation_ratio = jacobian_violation_ratio;

    Ok(DelaunayTetrahedronQuality {
        node_identities: identities,
        region_id,
        incident_metric_entity_ids,
        resolved_metric: metric.metric,
        active_metric_sources: metric.active_sources,
        applied_metric_contribution_count: metric.applied_contribution_count,
        clipped_metric_contribution_count: metric.clipped_contribution_count,
        rejected_metric_contribution_count: metric.rejected_contribution_count,
        minimum_metric_edge_length: geometry.minimum_metric_edge_length,
        maximum_metric_edge_length: geometry.maximum_metric_edge_length,
        metric_circumradius: geometry.metric_circumradius,
        metric_radius_edge_ratio: geometry.metric_radius_edge_ratio,
        metric_scaled_jacobian: geometry.metric_scaled_jacobian,
        boundary_node_count,
        protected_facet_count,
        refinement_violation_ratio,
        optimization_violation_ratio,
    })
}

pub(super) const fn radius_edge_is_refinement_owned(boundary_node_count: u8) -> bool {
    boundary_node_count == 0
}

pub(in crate::cdt) fn evaluate_tetrahedron_geometry(
    mut points: [[f64; 3]; 4],
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> Option<DelaunayTetrahedronGeometryQuality> {
    // Published topology is identity-ordered, while metric geometry must be
    // invariant under source-revision identity changes. A fixed operand order
    // also makes the translated metric transform and circumsphere arithmetic
    // reproducible for the same canonical simplex.
    points.sort_by(|left, right| {
        left.iter()
            .zip(right.iter())
            .find_map(|(left, right)| (left != right).then(|| left.total_cmp(right)))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let transformed = transform_points(points, metric)?;
    let (minimum_metric_edge_length, maximum_metric_edge_length) = edge_extrema(transformed)?;
    let metric_scaled_jacobian = tetrahedron_scaled_jacobian(transformed);
    // Exact orientation has already admitted canonical topology. Once the
    // shape is too ill-conditioned for a reproducible floating circumsphere,
    // retain deterministic worst-case evidence while scaled-Jacobian repair
    // owns the actual rejection.
    let metric_circumradius = if metric_scaled_jacobian < super::CIRCUMRADIUS_CONDITION_FLOOR {
        f64::MAX
    } else {
        circumradius(transformed).unwrap_or(f64::MAX)
    };
    let metric_radius_edge_ratio = saturated_ratio(metric_circumradius, minimum_metric_edge_length);
    (metric_scaled_jacobian.is_finite() && metric_scaled_jacobian >= 0.0).then_some(
        DelaunayTetrahedronGeometryQuality {
            minimum_metric_edge_length,
            maximum_metric_edge_length,
            metric_circumradius,
            metric_radius_edge_ratio,
            metric_scaled_jacobian,
        },
    )
}

fn saturated_ratio(numerator: f64, denominator: f64) -> f64 {
    let ratio = numerator / denominator;
    if ratio.is_finite() {
        ratio
    } else {
        f64::MAX
    }
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
                let relative = [
                    point[0] - points[0][0],
                    point[1] - points[0][1],
                    point[2] - points[0][2],
                ];
                [
                    l00 * relative[0] + l10 * relative[1] + l20 * relative[2],
                    l11 * relative[1] + l21 * relative[2],
                    l22 * relative[2],
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
            let mut length = squared.sqrt();
            if !length.is_finite() {
                return None;
            }
            if length == 0.0 && points[right] != points[left] {
                length = f64::MIN_POSITIVE;
            }
            if length == 0.0 {
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
    for row in 0..3 {
        let edge = std::array::from_fn(|axis| points[row + 1][axis] - points[0][axis]);
        matrix[row] = edge;
        right[row] = 0.5 * squared_norm(edge);
    }
    let center_offset = solve(matrix, right)?;
    let radius = squared_norm(center_offset).sqrt();
    (radius.is_finite() && radius > 0.0).then_some(radius)
}

fn solve(mut matrix: [[f64; 3]; 3], mut right: [f64; 3]) -> Option<[f64; 3]> {
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

fn squared_norm(point: [f64; 3]) -> f64 {
    point.into_iter().map(|value| value * value).sum()
}

#[cfg(test)]
mod tests {
    use runmat_meshing_size::metric::MetricTensor3;

    use super::{circumradius, evaluate_tetrahedron_geometry, transform_points};

    #[test]
    fn circumradius_is_translation_invariant_for_small_cells_at_large_coordinates() {
        let origin = [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.25],
        ];
        let translated = origin.map(|point| point.map(|value| value + 1.0e12));
        let expected = 3.0_f64.sqrt() / 8.0;

        assert!((circumradius(origin).unwrap() - expected).abs() < 1.0e-15);
        assert!((circumradius(translated).unwrap() - expected).abs() < 1.0e-15);
    }

    #[test]
    fn metric_transform_is_translation_invariant_for_small_cells_at_large_coordinates() {
        let origin = [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.25],
        ];
        let translated = origin.map(|point| point.map(|value| value + 1.0e12));
        let metric = MetricTensor3 {
            xx: 4.0,
            yy: 9.0,
            zz: 16.0,
            xy: 0.5,
            xz: 0.25,
            yz: 0.75,
        };

        assert_eq!(
            transform_points(origin, metric),
            transform_points(translated, metric)
        );
    }

    #[test]
    fn rejected_nearly_flat_cell_has_deterministic_worst_case_radius() {
        let geometry = evaluate_tetrahedron_geometry(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, 1.0e-12],
            ],
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
        )
        .unwrap();

        assert!(geometry.metric_scaled_jacobian < super::super::CIRCUMRADIUS_CONDITION_FLOOR);
        assert_eq!(geometry.metric_circumradius, f64::MAX);
        assert_eq!(geometry.metric_radius_edge_ratio, f64::MAX);
    }
}
