use std::collections::BTreeSet;

use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::{MetricTensor3, ResolvedMetricField};

use super::{
    checkpoint, error, validate_inputs, DelaunayTetrahedronQuality, DelaunayVolumeProvenance,
    DelaunayVolumeQuality, DelaunayVolumeQualityError, DelaunayVolumeQualityErrorKind,
    DelaunayVolumeQualityOptions, DelaunayVolumeTopology, MetricFieldRequest,
};

struct IndependentTetrahedronQuality {
    identities: [runmat_meshing_core::StableDigest; 4],
    region_id: runmat_geometry_core::PersistentEntityId,
    incident_entity_ids: Vec<runmat_geometry_core::PersistentEntityId>,
    metric: MetricTensor3,
    sources: Vec<runmat_meshing_size::metric::MetricSourceKind>,
    contribution_count: u32,
    clipped_contribution_count: u32,
    rejected_contribution_count: u32,
    values: [f64; 5],
}

pub fn validate_delaunay_volume_quality(
    topology: &DelaunayVolumeTopology,
    metric_request: &MetricFieldRequest,
    provenance: &DelaunayVolumeProvenance,
    quality: &DelaunayVolumeQuality,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeQualityError> {
    let metric_contexts =
        validate_inputs(topology, metric_request, provenance, options, cancellation)?;
    if quality.tetrahedra.len() != topology.tetrahedra.len() {
        return Err(invalid(
            None,
            "quality inventory is incomplete or malformed",
        ));
    }
    let field = ResolvedMetricField::new(metric_request).map_err(super::metric_error)?;
    let mut maximum_edge = 0.0_f64;
    let mut maximum_ratio = 0.0_f64;
    let mut worst = None::<(f64, [runmat_meshing_core::StableDigest; 4])>;

    for (index, ((tetrahedron, context), observed)) in topology
        .tetrahedra
        .iter()
        .zip(&metric_contexts)
        .zip(&quality.tetrahedra)
        .enumerate()
    {
        checkpoint(index as u64, options, cancellation)?;
        let region_id = tetrahedron.region_id.clone().ok_or_else(|| {
            invalid(
                Some(index),
                "quality cannot describe an unassigned tetrahedron",
            )
        })?;
        let incident_entities = context
            .incident_entity_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let resolved = field
            .resolve(&incident_entities)
            .map_err(super::metric_error)?;
        let points = tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].coordinates_m);
        let identities = tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].identity);
        let (minimum_edge, local_maximum_edge) = metric_edge_extrema(points, resolved.metric)
            .ok_or_else(|| invalid(Some(index), "independent metric edge evaluation failed"))?;
        let radius = metric_circumradius(points, resolved.metric).ok_or_else(|| {
            invalid(
                Some(index),
                "independent metric circumsphere evaluation failed",
            )
        })?;
        let radius_edge_ratio = radius / minimum_edge;
        let violation_ratio = (local_maximum_edge / options.maximum_metric_edge_length)
            .max(radius_edge_ratio / options.maximum_radius_edge_ratio);
        let independent = IndependentTetrahedronQuality {
            identities,
            region_id,
            incident_entity_ids: context.incident_entity_ids.clone(),
            metric: resolved.metric,
            sources: resolved.active_sources,
            contribution_count: resolved.applied_contribution_count,
            clipped_contribution_count: resolved.clipped_contribution_count,
            rejected_contribution_count: resolved.rejected_contribution_count,
            values: [
                minimum_edge,
                local_maximum_edge,
                radius,
                radius_edge_ratio,
                violation_ratio,
            ],
        };
        validate_tetrahedron(observed, &independent, index)?;
        maximum_edge = maximum_edge.max(local_maximum_edge);
        maximum_ratio = maximum_ratio.max(radius_edge_ratio);
        if violation_ratio > 1.0 {
            let candidate = (violation_ratio, identities);
            if worst.as_ref().is_none_or(|current| {
                candidate.0.total_cmp(&current.0).is_gt()
                    || candidate.0.total_cmp(&current.0).is_eq() && candidate.1 < current.1
            }) {
                worst = Some(candidate);
            }
        }
    }
    if quality.worst_refinement_tetrahedron != worst.map(|(_, identity)| identity)
        || !approximately_equal(quality.maximum_metric_edge_length, maximum_edge)
        || !approximately_equal(quality.maximum_radius_edge_ratio, maximum_ratio)
    {
        return Err(invalid(None, "aggregate quality evidence is inconsistent"));
    }
    Ok(())
}

fn validate_tetrahedron(
    observed: &DelaunayTetrahedronQuality,
    independent: &IndependentTetrahedronQuality,
    index: usize,
) -> Result<(), DelaunayVolumeQualityError> {
    if observed.node_identities != independent.identities
        || observed.region_id != independent.region_id
        || observed.incident_metric_entity_ids != independent.incident_entity_ids
        || observed.resolved_metric != independent.metric
        || observed.active_metric_sources != independent.sources
        || observed.applied_metric_contribution_count != independent.contribution_count
        || observed.clipped_metric_contribution_count != independent.clipped_contribution_count
        || observed.rejected_metric_contribution_count != independent.rejected_contribution_count
        || !approximately_equal(observed.minimum_metric_edge_length, independent.values[0])
        || !approximately_equal(observed.maximum_metric_edge_length, independent.values[1])
        || !approximately_equal(observed.metric_circumradius, independent.values[2])
        || !approximately_equal(observed.metric_radius_edge_ratio, independent.values[3])
        || !approximately_equal(observed.refinement_violation_ratio, independent.values[4])
    {
        return Err(invalid(
            Some(index),
            "tetrahedron quality does not match independent metric evaluation",
        ));
    }
    Ok(())
}

fn metric_edge_extrema(points: [[f64; 3]; 4], metric: MetricTensor3) -> Option<(f64, f64)> {
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for left in 0..4 {
        for right in (left + 1)..4 {
            let delta = [
                points[right][0] - points[left][0],
                points[right][1] - points[left][1],
                points[right][2] - points[left][2],
            ];
            let product = [
                metric.xx * delta[0] + metric.xy * delta[1] + metric.xz * delta[2],
                metric.xy * delta[0] + metric.yy * delta[1] + metric.yz * delta[2],
                metric.xz * delta[0] + metric.yz * delta[1] + metric.zz * delta[2],
            ];
            let length =
                (delta[0] * product[0] + delta[1] * product[1] + delta[2] * product[2]).sqrt();
            if !length.is_finite() || length <= 0.0 {
                return None;
            }
            minimum = minimum.min(length);
            maximum = maximum.max(length);
        }
    }
    Some((minimum, maximum))
}

fn metric_circumradius(points: [[f64; 3]; 4], metric: MetricTensor3) -> Option<f64> {
    let metric_product = |point: [f64; 3]| {
        [
            metric.xx * point[0] + metric.xy * point[1] + metric.xz * point[2],
            metric.xy * point[0] + metric.yy * point[1] + metric.yz * point[2],
            metric.xz * point[0] + metric.yz * point[1] + metric.zz * point[2],
        ]
    };
    let quadratic = |point: [f64; 3]| {
        let product = metric_product(point);
        point[0] * product[0] + point[1] * product[1] + point[2] * product[2]
    };
    let mut matrix = [[0.0; 3]; 3];
    let mut right = [0.0; 3];
    for row in 0..3 {
        let delta = [
            points[row + 1][0] - points[0][0],
            points[row + 1][1] - points[0][1],
            points[row + 1][2] - points[0][2],
        ];
        matrix[row] = metric_product(delta);
        right[row] = 0.5 * (quadratic(points[row + 1]) - quadratic(points[0]));
    }
    let center = gaussian_solve(matrix, right)?;
    let delta = [
        center[0] - points[0][0],
        center[1] - points[0][1],
        center[2] - points[0][2],
    ];
    let product = metric_product(delta);
    let radius = (delta[0] * product[0] + delta[1] * product[1] + delta[2] * product[2]).sqrt();
    (radius.is_finite() && radius > 0.0).then_some(radius)
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

fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs()
            <= 256.0 * f64::EPSILON * left.abs().max(right.abs()).max(f64::MIN_POSITIVE)
}

fn invalid(index: Option<usize>, reason: &'static str) -> DelaunayVolumeQualityError {
    error(
        DelaunayVolumeQualityErrorKind::InvalidQuality,
        index,
        reason,
    )
}
