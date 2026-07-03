use std::collections::BTreeMap;

use super::{
    LocalTetrahedronFlipCandidate, LocalTetrahedronFlipError, LocalTetrahedronFlipQualityReport,
    LocalTetrahedronFlipQualityThresholds,
};
use runmat_meshing_core::predicate::{
    tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_signed_volume, Point3,
};

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
