mod comparison;
use comparison::{approximately_equal, approximately_equal_ill_conditioned};

use std::collections::BTreeSet;

use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricTensor3;

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
    boundary_node_count: u8,
    protected_facet_count: u8,
    /// Longest metric edge not on a protected facet: the size criterion's measure.
    refinement_metric_edge_length: f64,
    values: [f64; 7],
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
    let resolved_metrics = super::metrics::resolve_graded_metrics(
        topology,
        &metric_contexts,
        metric_request,
        options,
        cancellation,
    )?;
    let node_size = super::node_size::NodeSizeField::new(metric_request, provenance)?;
    let mut maximum_edge = 0.0_f64;
    let mut maximum_ratio = 0.0_f64;
    let mut minimum_scaled_jacobian = f64::INFINITY;
    let mut worst = None::<(f64, [runmat_meshing_core::StableDigest; 4])>;
    let mut observed_worst = None::<(f64, [[u64; 3]; 4], [runmat_meshing_core::StableDigest; 4])>;
    let mut worst_optimization = None::<(f64, [runmat_meshing_core::StableDigest; 4])>;
    let mut observed_worst_optimization =
        None::<(f64, [[u64; 3]; 4], [runmat_meshing_core::StableDigest; 4])>;
    let mut reported_refinement_value = None;
    let mut reported_optimization_jacobian = None;
    let boundary_nodes = provenance
        .facets
        .iter()
        .flat_map(|facet| facet.node_identities)
        .collect::<BTreeSet<_>>();
    let protected_facets = super::protected_facet_identities(provenance);

    for (index, (((tetrahedron, context), observed), resolved)) in topology
        .tetrahedra
        .iter()
        .zip(metric_contexts)
        .zip(&quality.tetrahedra)
        .zip(resolved_metrics)
        .enumerate()
    {
        checkpoint(index as u64, options, cancellation)?;
        let context = context?;
        let region_id = tetrahedron.region_id.clone().ok_or_else(|| {
            invalid(
                Some(index),
                "quality cannot describe an unassigned tetrahedron",
            )
        })?;
        let points = tetrahedron.vertex_indices.map(|vertex| {
            topology.nodes[vertex as usize]
                .coordinates_m
                .canonical_value()
        });
        let identities = tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].identity);
        let (minimum_edge, local_maximum_edge) = metric_edge_extrema(points, resolved.metric)
            .ok_or_else(|| invalid(Some(index), "independent metric edge evaluation failed"))?;
        let scaled_jacobian = metric_scaled_jacobian(points, resolved.metric).ok_or_else(|| {
            invalid(
                Some(index),
                "independent metric scaled Jacobian evaluation failed",
            )
        })?;
        let radius = if scaled_jacobian < super::CIRCUMRADIUS_CONDITION_FLOOR {
            f64::MAX
        } else {
            metric_circumradius(points, resolved.metric).unwrap_or(f64::MAX)
        };
        let radius_edge_ratio = saturated_ratio(radius, minimum_edge);
        let boundary_node_count = identities
            .iter()
            .filter(|identity| boundary_nodes.contains(identity))
            .count() as u8;
        let protected_facet_count = super::protected_facet_count(identities, &protected_facets);
        let corner_metrics = node_size.corner_metrics(
            &topology.nodes,
            tetrahedron
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize]),
            &region_id,
        )?;
        let refinement_metric_edge_length =
            super::node_size::volume_owned_maximum_integrated_edge_length(
                points,
                corner_metrics,
                super::protected_edge_mask(identities, &protected_facets),
            );
        let size_violation_ratio = saturated_ratio(
            refinement_metric_edge_length,
            options.maximum_metric_edge_length,
        );
        let radius_edge_violation_ratio =
            saturated_ratio(radius_edge_ratio, options.maximum_radius_edge_ratio);
        let jacobian_violation_ratio =
            saturated_ratio(options.minimum_metric_scaled_jacobian, scaled_jacobian);
        let violation_ratio = size_violation_ratio.max(
            if super::evaluation::radius_edge_is_refinement_owned(boundary_node_count) {
                radius_edge_violation_ratio
            } else {
                0.0
            },
        );
        let optimization_violation_ratio = jacobian_violation_ratio;
        let independent = IndependentTetrahedronQuality {
            identities,
            region_id,
            incident_entity_ids: context.incident_entity_ids,
            metric: resolved.metric,
            sources: resolved.active_sources,
            contribution_count: resolved.applied_contribution_count,
            clipped_contribution_count: resolved.clipped_contribution_count,
            rejected_contribution_count: resolved.rejected_contribution_count,
            boundary_node_count,
            protected_facet_count,
            refinement_metric_edge_length,
            values: [
                minimum_edge,
                local_maximum_edge,
                radius,
                radius_edge_ratio,
                scaled_jacobian,
                violation_ratio,
                optimization_violation_ratio,
            ],
        };
        validate_tetrahedron(observed, &independent, options, index)?;
        if quality.worst_refinement_tetrahedron == Some(identities) {
            reported_refinement_value = Some(violation_ratio);
        }
        if quality.worst_optimization_tetrahedron == Some(identities) {
            reported_optimization_jacobian = Some(scaled_jacobian);
        }
        if observed.requires_refinement() {
            let candidate = (
                observed.refinement_violation_ratio,
                super::tetrahedron_geometry_key(topology, observed.node_identities),
                observed.node_identities,
            );
            if observed_worst.as_ref().is_none_or(|current| {
                super::compare_violation_priority(
                    candidate.0,
                    candidate.1,
                    candidate.2,
                    current.0,
                    current.1,
                    current.2,
                )
                .is_gt()
            }) {
                observed_worst = Some(candidate);
            }
        }
        if observed.requires_optimization() {
            let candidate = (
                observed.optimization_violation_ratio,
                super::tetrahedron_geometry_key(topology, observed.node_identities),
                observed.node_identities,
            );
            if observed_worst_optimization.as_ref().is_none_or(|current| {
                super::compare_violation_priority(
                    candidate.0,
                    candidate.1,
                    candidate.2,
                    current.0,
                    current.1,
                    current.2,
                )
                .is_gt()
            }) {
                observed_worst_optimization = Some(candidate);
            }
        }
        maximum_edge = maximum_edge.max(local_maximum_edge);
        maximum_ratio = maximum_ratio.max(radius_edge_ratio);
        minimum_scaled_jacobian = minimum_scaled_jacobian.min(scaled_jacobian);
        if super::violates_quality_bound(violation_ratio) {
            let candidate = (violation_ratio, identities);
            if worst.as_ref().is_none_or(|current| {
                candidate.0.total_cmp(&current.0).is_gt()
                    || candidate.0.total_cmp(&current.0).is_eq() && candidate.1 < current.1
            }) {
                worst = Some(candidate);
            }
        }
        if super::violates_quality_bound(optimization_violation_ratio) {
            let candidate = (optimization_violation_ratio, identities);
            if worst_optimization.as_ref().is_none_or(|current| {
                candidate.0.total_cmp(&current.0).is_gt()
                    || candidate.0.total_cmp(&current.0).is_eq() && candidate.1 < current.1
            }) {
                worst_optimization = Some(candidate);
            }
        }
    }
    let observed_shape_admissible = !super::violates_quality_bound(saturated_ratio(
        options.minimum_metric_scaled_jacobian,
        quality.minimum_metric_scaled_jacobian,
    ));
    let independent_shape_admissible = !super::violates_quality_bound(saturated_ratio(
        options.minimum_metric_scaled_jacobian,
        minimum_scaled_jacobian,
    ));
    if observed_shape_admissible != independent_shape_admissible
        || quality.worst_refinement_tetrahedron != observed_worst.map(|(_, _, identity)| identity)
        || !reported_worst_is_independently_maximal(
            quality.worst_refinement_tetrahedron,
            worst,
            reported_refinement_value,
        )
        || quality.worst_optimization_tetrahedron
            != observed_worst_optimization.map(|(_, _, identity)| identity)
        || !comparison::reported_shape_extremum_matches(
            quality.worst_optimization_tetrahedron,
            worst_optimization,
            reported_optimization_jacobian,
            minimum_scaled_jacobian,
            options.minimum_metric_scaled_jacobian,
        )
        || !approximately_equal(quality.maximum_metric_edge_length, maximum_edge)
        || !(if independent_shape_admissible {
            approximately_equal(quality.maximum_radius_edge_ratio, maximum_ratio)
                && approximately_equal(
                    quality.minimum_metric_scaled_jacobian,
                    minimum_scaled_jacobian,
                )
        } else {
            approximately_equal_ill_conditioned(
                quality.maximum_radius_edge_ratio,
                maximum_ratio,
                minimum_scaled_jacobian,
            ) && approximately_equal_ill_conditioned(
                quality.minimum_metric_scaled_jacobian,
                minimum_scaled_jacobian,
                minimum_scaled_jacobian,
            )
        })
    {
        return Err(invalid(
            None,
            format!(
                "aggregate quality evidence is inconsistent: shape admissible observed/independent={observed_shape_admissible}/{independent_shape_admissible}; reported optimization independent Jacobian={reported_optimization_jacobian:?}; minimum independent Jacobian={minimum_scaled_jacobian:.17e}; refinement worst reported/observed/independent={:?}/{:?}/{:?}; optimization worst reported/observed/independent={:?}/{:?}/{:?}; maximum edge reported/independent={:.17e}/{:.17e}; maximum radius ratio reported/independent={:.17e}/{:.17e}; minimum scaled Jacobian reported/independent={:.17e}/{:.17e}",
                quality.worst_refinement_tetrahedron,
                observed_worst.map(|(_, _, identity)| identity),
                worst.map(|(_, identity)| identity),
                quality.worst_optimization_tetrahedron,
                observed_worst_optimization.map(|(_, _, identity)| identity),
                worst_optimization.map(|(_, identity)| identity),
                quality.maximum_metric_edge_length,
                maximum_edge,
                quality.maximum_radius_edge_ratio,
                maximum_ratio,
                quality.minimum_metric_scaled_jacobian,
                minimum_scaled_jacobian,
            ),
        ));
    }
    Ok(())
}

/// Independent arithmetic can reverse two numerically tied extrema even after
/// every per-cell value has cross-validated. The published work source must
/// match its deterministic priority order, and independent arithmetic must place
/// it in the independently maximal scheduling class. Physical violation
/// classification still uses the full value.
fn reported_worst_is_independently_maximal(
    reported: Option<[runmat_meshing_core::StableDigest; 4]>,
    independent_worst: Option<(f64, [runmat_meshing_core::StableDigest; 4])>,
    reported_value: Option<f64>,
) -> bool {
    match (reported, independent_worst) {
        (None, None) => true,
        (Some(_), Some((maximum, _))) => reported_value.is_some_and(|value| {
            super::violates_quality_bound(value)
                && super::violation_priority_key(value) == super::violation_priority_key(maximum)
        }),
        _ => false,
    }
}

fn validate_tetrahedron(
    observed: &DelaunayTetrahedronQuality,
    independent: &IndependentTetrahedronQuality,
    options: DelaunayVolumeQualityOptions,
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
        || observed.boundary_node_count != independent.boundary_node_count
        || observed.protected_facet_count != independent.protected_facet_count
    {
        return Err(invalid(
            Some(index),
            "tetrahedron quality identity or metric provenance does not match independent evaluation",
        ));
    }
    let observed_shape_admissible = !super::violates_quality_bound(saturated_ratio(
        options.minimum_metric_scaled_jacobian,
        observed.metric_scaled_jacobian,
    ));
    let independent_shape_admissible = !super::violates_quality_bound(saturated_ratio(
        options.minimum_metric_scaled_jacobian,
        independent.values[4],
    ));
    let independent_shape_well_conditioned = independent.values[4] >= 1.0e-5;
    // Independent direct-metric evaluation and the construction path's factored-coordinate
    // evaluation lose agreement as a cell collapses. Numerical conditioning is geometric and
    // independent of the caller's admission threshold, so every nearly collapsed cell receives
    // the same bounded, conditioning-aware comparison.
    let approximately_equal_derived = |left, right| {
        if independent_shape_well_conditioned {
            approximately_equal(left, right)
        } else {
            approximately_equal_ill_conditioned(left, right, independent.values[4])
        }
    };
    for (name, observed, expected) in [
        (
            "minimum metric edge length",
            observed.minimum_metric_edge_length,
            independent.values[0],
        ),
        (
            "maximum metric edge length",
            observed.maximum_metric_edge_length,
            independent.values[1],
        ),
    ] {
        if !approximately_equal_derived(observed, expected) {
            return Err(invalid(
                Some(index),
                format!(
                    "tetrahedron {name} {observed:.17e} does not match independent evaluation {expected:.17e} (metric scaled Jacobian {:.17e})",
                    independent.values[4]
                ),
            ));
        }
    }
    if observed_shape_admissible != independent_shape_admissible {
        return Err(invalid(
            Some(index),
            "tetrahedron shape-admission decision does not match independent evaluation",
        ));
    }
    let size_violation = saturated_ratio(
        independent.refinement_metric_edge_length,
        options.maximum_metric_edge_length,
    );
    let radius_edge_violation = saturated_ratio(
        observed.metric_radius_edge_ratio,
        options.maximum_radius_edge_ratio,
    );
    let jacobian_violation = saturated_ratio(
        options.minimum_metric_scaled_jacobian,
        observed.metric_scaled_jacobian,
    );
    let observed_violation = size_violation.max(
        if super::evaluation::radius_edge_is_refinement_owned(observed.boundary_node_count) {
            radius_edge_violation
        } else {
            0.0
        },
    );
    let observed_optimization_violation = jacobian_violation;
    if !approximately_equal(observed.refinement_violation_ratio, observed_violation) {
        return Err(invalid(
            Some(index),
            "tetrahedron refinement violation ratio is inconsistent with its quality values",
        ));
    }
    if !approximately_equal(
        observed.optimization_violation_ratio,
        observed_optimization_violation,
    ) {
        return Err(invalid(
            Some(index),
            "tetrahedron optimization violation ratio is inconsistent with its quality values",
        ));
    }
    for (name, observed, expected) in [
        (
            "metric circumradius",
            observed.metric_circumradius,
            independent.values[2],
        ),
        (
            "metric radius-edge ratio",
            observed.metric_radius_edge_ratio,
            independent.values[3],
        ),
        (
            "metric scaled Jacobian",
            observed.metric_scaled_jacobian,
            independent.values[4],
        ),
    ] {
        if !approximately_equal_derived(observed, expected) {
            return Err(invalid(
                Some(index),
                format!(
                    "tetrahedron {name} {observed:.17e} does not match independent evaluation {expected:.17e} (metric scaled Jacobian {:.17e})",
                    independent.values[4]
                ),
            ));
        }
    }
    // A violation ratio divides by the scaled Jacobian. Once both evaluation paths have
    // independently rejected a nearly collapsed cell, that division can amplify their bounded
    // Jacobian disagreement without bound. The observed ratio was checked against the observed
    // primitive values above; retain direct cross-path comparison for admissible cells only.
    if independent_shape_well_conditioned
        && !approximately_equal(observed.refinement_violation_ratio, independent.values[5])
    {
        return Err(invalid(
            Some(index),
            format!(
                "tetrahedron refinement violation ratio {:.17e} does not match independent evaluation {:.17e}",
                observed.refinement_violation_ratio, independent.values[5]
            ),
        ));
    }
    if independent_shape_well_conditioned
        && !approximately_equal(observed.optimization_violation_ratio, independent.values[6])
    {
        return Err(invalid(
            Some(index),
            format!(
                "tetrahedron optimization violation ratio {:.17e} does not match independent evaluation {:.17e}",
                observed.optimization_violation_ratio, independent.values[6]
            ),
        ));
    }
    Ok(())
}

fn metric_scaled_jacobian(points: [[f64; 3]; 4], metric: MetricTensor3) -> Option<f64> {
    let determinant = metric.xx * (metric.yy * metric.zz - metric.yz * metric.yz)
        - metric.xy * (metric.xy * metric.zz - metric.yz * metric.xz)
        + metric.xz * (metric.xy * metric.yz - metric.yy * metric.xz);
    if !determinant.is_finite() || determinant <= 0.0 {
        return None;
    }
    let corners = [(0, 1, 2, 3), (1, 0, 3, 2), (2, 0, 1, 3), (3, 0, 2, 1)];
    let mut minimum = f64::INFINITY;
    for (origin, first, second, third) in corners {
        let edges = [first, second, third].map(|vertex| {
            [
                points[vertex][0] - points[origin][0],
                points[vertex][1] - points[origin][1],
                points[vertex][2] - points[origin][2],
            ]
        });
        let lengths = edges.map(|edge| metric_length(edge, metric));
        if lengths
            .into_iter()
            .any(|length| !length.is_finite() || length <= 0.0)
        {
            return None;
        }
        let physical_jacobian = determinant3(edges).abs();
        let scaled = 2.0_f64.sqrt() * determinant.sqrt() * physical_jacobian
            / lengths.into_iter().product::<f64>();
        if !scaled.is_finite() || scaled < 0.0 {
            return None;
        }
        minimum = minimum.min(scaled);
    }
    Some(minimum)
}

fn metric_length(vector: [f64; 3], metric: MetricTensor3) -> f64 {
    let product = [
        metric.xx * vector[0] + metric.xy * vector[1] + metric.xz * vector[2],
        metric.xy * vector[0] + metric.yy * vector[1] + metric.yz * vector[2],
        metric.xz * vector[0] + metric.yz * vector[1] + metric.zz * vector[2],
    ];
    (vector[0] * product[0] + vector[1] * product[1] + vector[2] * product[2]).sqrt()
}

fn saturated_ratio(numerator: f64, denominator: f64) -> f64 {
    let ratio = numerator / denominator;
    if ratio.is_finite() {
        ratio
    } else {
        f64::MAX
    }
}

fn determinant3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
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
            let mut length =
                (delta[0] * product[0] + delta[1] * product[1] + delta[2] * product[2]).sqrt();
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

fn metric_circumradius(points: [[f64; 3]; 4], metric: MetricTensor3) -> Option<f64> {
    let metric_product = |point: [f64; 3]| {
        [
            metric.xx * point[0] + metric.xy * point[1] + metric.xz * point[2],
            metric.xy * point[0] + metric.yy * point[1] + metric.yz * point[2],
            metric.xz * point[0] + metric.yz * point[1] + metric.zz * point[2],
        ]
    };
    let quadratic = |vector: [f64; 3]| {
        let product = metric_product(vector);
        vector[0] * product[0] + vector[1] * product[1] + vector[2] * product[2]
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
        right[row] = 0.5 * quadratic(delta);
    }
    let center_offset = gaussian_solve(matrix, right)?;
    let radius = quadratic(center_offset).sqrt();
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

fn invalid(index: Option<usize>, reason: impl Into<String>) -> DelaunayVolumeQualityError {
    error(
        DelaunayVolumeQualityErrorKind::InvalidQuality,
        index,
        reason,
    )
}
