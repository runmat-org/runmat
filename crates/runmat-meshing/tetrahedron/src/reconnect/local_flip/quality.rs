use std::collections::BTreeMap;

use super::{
    LocalTetrahedron, LocalTetrahedronFlipCandidate, LocalTetrahedronFlipError,
    LocalTetrahedronFlipImprovementReport, LocalTetrahedronFlipQualityReport,
    LocalTetrahedronFlipQualityThresholds,
};
use runmat_meshing_core::quality::predicate::{
    tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian,
    tetrahedron_scaled_jacobian_with_owner, tetrahedron_signed_volume, Point3,
};
use runmat_simulation_usage::{OperationCount, UsageCategory, UsageContext};

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
    let usage = UsageContext::capture();
    let mut comparison_batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Meshing, OperationCount::from_operations(2))
        })
        .transpose()
        .map_err(LocalTetrahedronFlipError::Accounting)?;
    let improves_quality =
        candidate_quality.min_scaled_jacobian > current.min_scaled_jacobian + MIN_IMPROVEMENT;
    let preserves_quality = (candidate_quality.min_scaled_jacobian - current.min_scaled_jacobian)
        .abs()
        <= MIN_IMPROVEMENT;
    if let Some(batch) = &mut comparison_batch {
        batch
            .complete(OperationCount::from_operations(2))
            .map_err(LocalTetrahedronFlipError::Accounting)?;
    }
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

    // Per visited tetrahedron: signed volume orient3d/division 24, four
    // scaled-Jacobian corners up to 280, and six edge lengths/ratio 55.
    // The total-volume sum adds n-1 terms; the first term initializes it.
    // Rejected prefixes settle against the candidate's actual n-bound.
    let tetrahedron_count = u64::try_from(candidate.created_tetrahedra.len())
        .ok()
        .ok_or(LocalTetrahedronFlipError::Accounting(
            runmat_simulation_usage::LedgerError::CountOverflow,
        ))?;
    let maximum = tetrahedron_count
        .checked_mul(359)
        .and_then(|work| work.checked_add(tetrahedron_count.saturating_sub(1)))
        .ok_or(LocalTetrahedronFlipError::Accounting(
            runmat_simulation_usage::LedgerError::CountOverflow,
        ))?;
    let usage = UsageContext::capture();
    let mut batch = if maximum == 0 {
        None
    } else {
        usage
            .as_ref()
            .map(|usage| {
                usage.ledger().reserve(
                    UsageCategory::Meshing,
                    OperationCount::from_operations(maximum),
                )
            })
            .transpose()
            .map_err(LocalTetrahedronFlipError::Accounting)?
    };
    let mut completed = 0;
    let result = (|| {
        let mut total_volume_m3 = 0.0_f64;
        let mut min_volume_m3 = f64::INFINITY;
        let mut min_scaled_jacobian = f64::INFINITY;
        let mut max_aspect_ratio = 0.0_f64;
        for (completed_tetrahedra, node_ids) in candidate.created_tetrahedra.iter().enumerate() {
            let points = [
                *node_coordinates.get(&node_ids[0]).ok_or(
                    LocalTetrahedronFlipError::MissingNode {
                        node_id: node_ids[0],
                    },
                )?,
                *node_coordinates.get(&node_ids[1]).ok_or(
                    LocalTetrahedronFlipError::MissingNode {
                        node_id: node_ids[1],
                    },
                )?,
                *node_coordinates.get(&node_ids[2]).ok_or(
                    LocalTetrahedronFlipError::MissingNode {
                        node_id: node_ids[2],
                    },
                )?,
                *node_coordinates.get(&node_ids[3]).ok_or(
                    LocalTetrahedronFlipError::MissingNode {
                        node_id: node_ids[3],
                    },
                )?,
            ];
            let volume_m3 = tetrahedron_signed_volume(points).abs();
            completed += 24;
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
            let scaled_jacobian = match batch.as_mut() {
                Some(owner) => {
                    tetrahedron_scaled_jacobian_with_owner(points, owner, &mut completed)
                }
                None => tetrahedron_scaled_jacobian(points),
            }
            .map_err(LocalTetrahedronFlipError::Accounting)?;
            if !scaled_jacobian.is_finite() || scaled_jacobian < thresholds.min_scaled_jacobian {
                return Err(LocalTetrahedronFlipError::ScaledJacobianBelowThreshold {
                    node_ids: *node_ids,
                    scaled_jacobian: stable_float(scaled_jacobian),
                });
            }
            let aspect_ratio = tetrahedron_edge_aspect_ratio(points);
            total_volume_m3 += volume_m3;
            completed += 55 + u64::from(completed_tetrahedra > 0);
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
    })();
    if let Some(batch) = &mut batch {
        batch
            .complete(OperationCount::from_operations(completed))
            .map_err(LocalTetrahedronFlipError::Accounting)?;
    }
    result
}

fn stable_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.12e}")
    } else {
        value.to_string()
    }
}
