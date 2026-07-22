use std::collections::BTreeMap;

use super::{
    LocalTetrahedron, LocalTetrahedronFlipCandidate, LocalTetrahedronFlipError,
    LocalTetrahedronFlipImprovementReport, LocalTetrahedronFlipQualityReport,
    LocalTetrahedronFlipQualityThresholds,
};
use runmat_meshing_core::quality::predicate::{
    tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_signed_volume, Point3,
};

const MIN_IMPROVEMENT: f64 = 1.0e-12;

pub fn evaluate_local_tetrahedron_flip_improvement(
    removed_tetrahedra: &[LocalTetrahedron],
    candidate: &LocalTetrahedronFlipCandidate,
    node_coordinates: &BTreeMap<u32, Point3>,
    thresholds: LocalTetrahedronFlipQualityThresholds,
) -> Result<LocalTetrahedronFlipImprovementReport, LocalTetrahedronFlipError> {
    let mut removed_ids = removed_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.tetrahedron_id)
        .collect::<Vec<_>>();
    removed_ids.sort();
    if removed_ids != candidate.removed_tetrahedron_ids {
        return Err(LocalTetrahedronFlipError::QualityDoesNotImprove);
    }
    let current = evaluate_local_tetrahedron_flip_quality(
        &LocalTetrahedronFlipCandidate {
            kind: candidate.kind,
            removed_tetrahedron_ids: removed_ids,
            created_tetrahedra: removed_tetrahedra
                .iter()
                .map(|tetrahedron| tetrahedron.node_ids)
                .collect(),
            shared_face: candidate.shared_face,
            shared_edge: candidate.shared_edge,
        },
        node_coordinates,
        LocalTetrahedronFlipQualityThresholds {
            min_volume_m3: 0.0,
            min_scaled_jacobian: 0.0,
        },
    )?;
    let candidate_quality =
        evaluate_local_tetrahedron_flip_quality(candidate, node_coordinates, thresholds)?;
    let improves_quality =
        candidate_quality.min_scaled_jacobian > current.min_scaled_jacobian + MIN_IMPROVEMENT;
    let preserves_quality = (candidate_quality.min_scaled_jacobian - current.min_scaled_jacobian)
        .abs()
        <= MIN_IMPROVEMENT;
    let reduces_tetrahedron_count =
        candidate_quality.created_tetrahedron_count < removed_tetrahedra.len();
    if !(improves_quality || preserves_quality && reduces_tetrahedron_count) {
        return Err(LocalTetrahedronFlipError::QualityDoesNotImprove);
    }

    Ok(LocalTetrahedronFlipImprovementReport {
        removed_tetrahedron_count: removed_tetrahedra.len(),
        created_tetrahedron_count: candidate_quality.created_tetrahedron_count,
        current_min_scaled_jacobian: current.min_scaled_jacobian,
        candidate_min_scaled_jacobian: candidate_quality.min_scaled_jacobian,
        current_total_volume_m3: current.total_volume_m3,
        candidate_total_volume_m3: candidate_quality.total_volume_m3,
    })
}

pub fn evaluate_local_tetrahedron_flip_quality(
    candidate: &LocalTetrahedronFlipCandidate,
    node_coordinates: &BTreeMap<u32, Point3>,
    thresholds: LocalTetrahedronFlipQualityThresholds,
) -> Result<LocalTetrahedronFlipQualityReport, LocalTetrahedronFlipError> {
    if !thresholds.min_volume_m3.is_finite()
        || thresholds.min_volume_m3 < 0.0
        || !thresholds.min_scaled_jacobian.is_finite()
        || thresholds.min_scaled_jacobian < 0.0
    {
        return Err(LocalTetrahedronFlipError::InvalidQualityThresholds);
    }

    let mut total_volume_m3 = 0.0_f64;
    let mut min_volume_m3 = f64::INFINITY;
    let mut min_scaled_jacobian = f64::INFINITY;
    let mut max_aspect_ratio = 0.0_f64;
    for node_ids in &candidate.created_tetrahedra {
        let points = [
            *node_coordinates
                .get(&node_ids[0])
                .ok_or(LocalTetrahedronFlipError::MissingNode {
                    node_id: node_ids[0],
                })?,
            *node_coordinates
                .get(&node_ids[1])
                .ok_or(LocalTetrahedronFlipError::MissingNode {
                    node_id: node_ids[1],
                })?,
            *node_coordinates
                .get(&node_ids[2])
                .ok_or(LocalTetrahedronFlipError::MissingNode {
                    node_id: node_ids[2],
                })?,
            *node_coordinates
                .get(&node_ids[3])
                .ok_or(LocalTetrahedronFlipError::MissingNode {
                    node_id: node_ids[3],
                })?,
        ];
        let volume_m3 = tetrahedron_signed_volume(points).abs();
        if !volume_m3.is_finite() || volume_m3 <= 0.0 {
            return Err(LocalTetrahedronFlipError::NonPositiveVolume {
                node_ids: *node_ids,
            });
        }
        if volume_m3 < thresholds.min_volume_m3 {
            return Err(LocalTetrahedronFlipError::VolumeBelowThreshold {
                node_ids: *node_ids,
                volume_m3: stable_float(volume_m3),
            });
        }
        let scaled_jacobian = tetrahedron_scaled_jacobian(points);
        if !scaled_jacobian.is_finite() || scaled_jacobian < thresholds.min_scaled_jacobian {
            return Err(LocalTetrahedronFlipError::ScaledJacobianBelowThreshold {
                node_ids: *node_ids,
                scaled_jacobian: stable_float(scaled_jacobian),
            });
        }
        let aspect_ratio = tetrahedron_edge_aspect_ratio(points);
        total_volume_m3 += volume_m3;
        min_volume_m3 = min_volume_m3.min(volume_m3);
        min_scaled_jacobian = min_scaled_jacobian.min(scaled_jacobian);
        max_aspect_ratio = max_aspect_ratio.max(aspect_ratio);
    }

    Ok(LocalTetrahedronFlipQualityReport {
        created_tetrahedron_count: candidate.created_tetrahedra.len(),
        total_volume_m3,
        min_volume_m3,
        min_scaled_jacobian,
        max_aspect_ratio,
    })
}

fn stable_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.12e}")
    } else {
        value.to_string()
    }
}
