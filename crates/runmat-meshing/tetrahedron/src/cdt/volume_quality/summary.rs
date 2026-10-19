//! Canonical aggregate and worst-cell reduction shared by full and mutation evaluation.
use super::*;

pub(super) fn summarize(
    topology: &DelaunayVolumeTopology,
    tetrahedra: Vec<DelaunayTetrahedronQuality>,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeQuality, DelaunayVolumeQualityError> {
    let summary = reduce(topology, &tetrahedra, options, cancellation)?;
    Ok(DelaunayVolumeQuality {
        tetrahedra,
        worst_refinement_tetrahedron: summary.worst_refinement_tetrahedron,
        worst_optimization_tetrahedron: summary.worst_optimization_tetrahedron,
        maximum_metric_edge_length: summary.maximum_metric_edge_length,
        maximum_radius_edge_ratio: summary.maximum_radius_edge_ratio,
        minimum_metric_scaled_jacobian: summary.minimum_metric_scaled_jacobian,
    })
}

#[derive(PartialEq)]
pub(super) struct QualitySummary {
    worst_refinement_tetrahedron: Option<[StableDigest; 4]>,
    worst_optimization_tetrahedron: Option<[StableDigest; 4]>,
    maximum_metric_edge_length: f64,
    maximum_radius_edge_ratio: f64,
    minimum_metric_scaled_jacobian: f64,
}

impl QualitySummary {
    pub(super) fn matches(&self, quality: &DelaunayVolumeQuality) -> bool {
        *self
            == Self {
                worst_refinement_tetrahedron: quality.worst_refinement_tetrahedron,
                worst_optimization_tetrahedron: quality.worst_optimization_tetrahedron,
                maximum_metric_edge_length: quality.maximum_metric_edge_length,
                maximum_radius_edge_ratio: quality.maximum_radius_edge_ratio,
                minimum_metric_scaled_jacobian: quality.minimum_metric_scaled_jacobian,
            }
    }
}

pub(super) fn reduce(
    topology: &DelaunayVolumeTopology,
    tetrahedra: &[DelaunayTetrahedronQuality],
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<QualitySummary, DelaunayVolumeQualityError> {
    let mut worst = None::<(f64, [[u64; 3]; 4], [StableDigest; 4])>;
    let mut worst_optimization = None::<(f64, [[u64; 3]; 4], [StableDigest; 4])>;
    let mut maximum_metric_edge_length = 0.0_f64;
    let mut maximum_radius_edge_ratio = 0.0_f64;
    let mut minimum_metric_scaled_jacobian = f64::INFINITY;

    for (index, quality) in tetrahedra.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        maximum_metric_edge_length =
            maximum_metric_edge_length.max(quality.maximum_metric_edge_length);
        maximum_radius_edge_ratio = maximum_radius_edge_ratio.max(quality.metric_radius_edge_ratio);
        minimum_metric_scaled_jacobian =
            minimum_metric_scaled_jacobian.min(quality.metric_scaled_jacobian);
        if quality.requires_refinement() {
            let candidate = (
                quality.refinement_violation_ratio,
                tetrahedron_geometry_key(topology, quality.node_identities),
                quality.node_identities,
            );
            if worst.as_ref().is_none_or(|current| {
                compare_violation_priority(
                    candidate.0,
                    candidate.1,
                    candidate.2,
                    current.0,
                    current.1,
                    current.2,
                )
                .is_gt()
            }) {
                worst = Some(candidate);
            }
        }
        if quality.requires_optimization() {
            let candidate = (
                quality.optimization_violation_ratio,
                tetrahedron_geometry_key(topology, quality.node_identities),
                quality.node_identities,
            );
            if worst_optimization.as_ref().is_none_or(|current| {
                compare_violation_priority(
                    candidate.0,
                    candidate.1,
                    candidate.2,
                    current.0,
                    current.1,
                    current.2,
                )
                .is_gt()
            }) {
                worst_optimization = Some(candidate);
            }
        }
    }

    Ok(QualitySummary {
        worst_refinement_tetrahedron: worst.map(|(_, _, identity)| identity),
        worst_optimization_tetrahedron: worst_optimization.map(|(_, _, identity)| identity),
        maximum_metric_edge_length,
        maximum_radius_edge_ratio,
        minimum_metric_scaled_jacobian,
    })
}
