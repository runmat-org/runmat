use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_size::metric::ResolvedMetricEvaluation;
use runmat_simulation_usage::{OperationCount, UsageCategory, UsageContext};

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
    /// Edges on protected facets (see `protected_edge_mask`), excluded from the size criterion.
    pub(super) protected_edges: u8,
}

pub(super) fn evaluate_tetrahedron(
    topology: &DelaunayVolumeTopology,
    tetrahedron_index: usize,
    context: TetrahedronEvaluationContext,
    options: DelaunayVolumeQualityOptions,
) -> Result<DelaunayTetrahedronQuality, DelaunayVolumeQualityError> {
    let tetrahedron = &topology.tetrahedra[tetrahedron_index];
    evaluate_tetrahedron_nodes(
        tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize]),
        tetrahedron_index,
        context,
        options,
    )
}

pub(super) fn evaluate_tetrahedron_nodes(
    nodes: [crate::cdt::DelaunayVolumeNode; 4],
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
        protected_edges,
    } = context;
    let points = nodes.map(|node| node.coordinates_m.canonical_value());
    let identities = nodes.map(|node| node.identity);
    let geometry = evaluate_tetrahedron_geometry(points, metric.metric)
        .map_err(|cause| {
            error(
                DelaunayVolumeQualityErrorKind::Accounting(cause),
                Some(tetrahedron_index),
                "metric tetrahedron numerical accounting failed",
            )
        })?
        .ok_or_else(|| {
            error(
            DelaunayVolumeQualityErrorKind::NumericalFailure,
            Some(tetrahedron_index),
            "metric tetrahedron geometry must be finite, nondegenerate, and positively oriented",
        )
        })?;
    let (refinement_violation_ratio, optimization_violation_ratio) = violation_ratios(
        volume_owned_maximum_metric_edge_length(points, metric.metric, protected_edges),
        geometry.metric_radius_edge_ratio,
        geometry.metric_scaled_jacobian,
        boundary_node_count,
        options,
    );

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

/// Size, radius-edge and shape violations of one tetrahedron. `refinement_metric_edge_length` is
/// the longest metric edge the volume stage can split: edges of protected facets belong to the
/// published surface mesh and are excluded, as the surface stage owns their length.
pub(in crate::cdt) fn violation_ratios(
    refinement_metric_edge_length: f64,
    metric_radius_edge_ratio: f64,
    metric_scaled_jacobian: f64,
    boundary_node_count: u8,
    options: DelaunayVolumeQualityOptions,
) -> (f64, f64) {
    let size_violation_ratio = saturated_ratio(
        refinement_metric_edge_length,
        options.maximum_metric_edge_length,
    );
    let radius_edge_violation_ratio =
        saturated_ratio(metric_radius_edge_ratio, options.maximum_radius_edge_ratio);
    let jacobian_violation_ratio = saturated_ratio(
        options.minimum_metric_scaled_jacobian,
        metric_scaled_jacobian,
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

    (refinement_violation_ratio, optimization_violation_ratio)
}

/// Longest metric edge length over the edges not on a protected facet (zero when every edge is
/// protected). Computed directly as sqrt(d^T M d) per edge.
pub(in crate::cdt) fn volume_owned_maximum_metric_edge_length(
    points: [[f64; 3]; 4],
    metric: runmat_meshing_size::metric::MetricTensor3,
    protected_edges: u8,
) -> f64 {
    super::TETRAHEDRON_EDGES
        .iter()
        .enumerate()
        .filter(|(bit, _)| protected_edges & (1 << bit) == 0)
        .map(|(_, (left, right))| {
            let [x, y, z]: [f64; 3] =
                std::array::from_fn(|axis| points[*right][axis] - points[*left][axis]);
            (metric.xx * x * x
                + metric.yy * y * y
                + metric.zz * z * z
                + 2.0 * (metric.xy * x * y + metric.xz * x * z + metric.yz * y * z))
                .max(0.0)
                .sqrt()
        })
        .fold(0.0, f64::max)
}

pub(super) const fn radius_edge_is_refinement_owned(boundary_node_count: u8) -> bool {
    boundary_node_count == 0
}

pub(in crate::cdt) fn evaluate_tetrahedron_geometry(
    points: [[f64; 3]; 4],
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> Result<Option<DelaunayTetrahedronGeometryQuality>, runmat_simulation_usage::LedgerError> {
    let Some(prepared) = super::jacobian::PreparedJacobian::new(metric)? else {
        return Ok(None);
    };
    let Some(geometry) = prepared.geometry(points)? else {
        return Ok(None);
    };
    let transformed = geometry.points;
    let minimum_metric_edge_length = geometry.minimum_edge;
    let maximum_metric_edge_length = geometry.maximum_edge;
    let metric_scaled_jacobian = geometry.jacobian;
    // Exact orientation has already admitted canonical topology. Once the
    // shape is too ill-conditioned for a reproducible floating circumsphere,
    // retain deterministic worst-case evidence while scaled-Jacobian repair
    // owns the actual rejection.
    // The floor branch evaluates only the radius/edge division (1). The
    // conditioned branch admits the complete 3x3 circumradius solve (68)
    // and that division before any of its arithmetic executes.
    let conditioned = metric_scaled_jacobian.partial_cmp(&super::CIRCUMRADIUS_CONDITION_FLOOR)
        != Some(std::cmp::Ordering::Less);
    let maximum = if conditioned { 69 } else { 1 };
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Meshing,
                OperationCount::from_operations(maximum),
            )
        })
        .transpose()?;
    let (metric_circumradius, work) = if conditioned {
        let (radius, work) = circumradius_reference(transformed);
        (radius.unwrap_or(f64::MAX), work)
    } else {
        (f64::MAX, 0)
    };
    let metric_radius_edge_ratio = saturated_ratio(metric_circumradius, minimum_metric_edge_length);
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(work + 1))?;
    }
    Ok(
        (metric_scaled_jacobian.is_finite() && metric_scaled_jacobian >= 0.0).then_some(
            DelaunayTetrahedronGeometryQuality {
                minimum_metric_edge_length,
                maximum_metric_edge_length,
                metric_circumradius,
                metric_radius_edge_ratio,
                metric_scaled_jacobian,
            },
        ),
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

#[cfg(test)]
fn transform_points(
    points: [[f64; 3]; 4],
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> Option<[[f64; 3]; 4]> {
    super::jacobian::PreparedJacobian::new(metric)
        .unwrap()
        .map(|prepared| prepared.transform(points).unwrap())
}

/// Circumradius reference maximum: three rows each with three differences,
/// squared norm (3 products, 2 sums), and scale = 27; elimination = 25;
/// back-substitution = 10; final squared norm and square root = 6. Total 68.
fn circumradius_reference(points: [[f64; 3]; 4]) -> (Option<f64>, u64) {
    let mut matrix = [[0.0; 3]; 3];
    let mut right = [0.0; 3];
    for row in 0..3 {
        let edge = std::array::from_fn(|axis| points[row + 1][axis] - points[0][axis]);
        matrix[row] = edge;
        right[row] = 0.5 * squared_norm(edge);
    }
    let (center_offset, solve_work) = solve_reference(matrix, right);
    let Some(center_offset) = center_offset else {
        return (None, 27 + solve_work);
    };
    let radius = squared_norm(center_offset).sqrt();
    (
        (radius.is_finite() && radius > 0.0).then_some(radius),
        27 + solve_work + 6,
    )
}

#[cfg(test)]
fn circumradius(points: [[f64; 3]; 4]) -> Option<f64> {
    circumradius_reference(points).0
}

fn solve_reference(mut matrix: [[f64; 3]; 3], mut right: [f64; 3]) -> (Option<[f64; 3]>, u64) {
    let mut completed = 0;
    for pivot in 0..3 {
        let Some(pivot_row) = (pivot..3).max_by(|left, right_row| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right_row][pivot].abs())
        }) else {
            return (None, completed);
        };
        if matrix[pivot_row][pivot] == 0.0 || !matrix[pivot_row][pivot].is_finite() {
            return (None, completed);
        }
        matrix.swap(pivot, pivot_row);
        right.swap(pivot, pivot_row);
        for row in (pivot + 1)..3 {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for column in pivot..3 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            right[row] -= factor * right[pivot];
            completed += 1 + (3 - pivot) as u64 * 2 + 2;
        }
    }
    let mut result = [0.0; 3];
    for row in (0..3).rev() {
        let remainder = ((row + 1)..3)
            .map(|column| matrix[row][column] * result[column])
            .sum::<f64>();
        result[row] = (right[row] - remainder) / matrix[row][row];
        let terms = 2 - row;
        completed += terms as u64 + terms.saturating_sub(1) as u64 + 2;
    }
    (
        result.into_iter().all(f64::is_finite).then_some(result),
        completed,
    )
}

fn squared_norm(point: [f64; 3]) -> f64 {
    point.into_iter().map(|value| value * value).sum()
}

#[cfg(test)]
mod tests {
    use runmat_meshing_size::metric::MetricTensor3;
    use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};
    use std::sync::Arc;

    use super::{circumradius, evaluate_tetrahedron_geometry, transform_points};

    #[test]
    fn geometry_settles_conditioned_radius_and_floor_division() {
        let metric = MetricTensor3::isotropic_length_m(1.0).unwrap();
        // Each geometry evaluation has Cholesky 14, transform 48, edges 54,
        // and four scaled-Jacobian corners 280. The conditioned radius adds
        // 68 solve operations and one ratio division; floor adds only one.
        let budget = OperationCount::from_operations(465 + 397);
        let ledger = Arc::new(UsageLedger::new(budget, budget).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        context.run_sync(|| {
            assert!(evaluate_tetrahedron_geometry(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ],
                metric,
            )
            .unwrap()
            .is_some());
            assert!(evaluate_tetrahedron_geometry(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.25, 0.25, 1.0e-12]
                ],
                metric,
            )
            .unwrap()
            .is_some());
        });
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.completed.meshing, budget);
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
    }

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
        .unwrap()
        .unwrap();

        assert!(geometry.metric_scaled_jacobian < super::super::CIRCUMRADIUS_CONDITION_FLOOR);
        assert_eq!(geometry.metric_circumradius, f64::MAX);
        assert_eq!(geometry.metric_radius_edge_ratio, f64::MAX);
    }
}
