use std::collections::BTreeMap;

use runmat_meshing_core::quality::predicate::{
    tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian,
    tetrahedron_scaled_jacobian_with_owner, tetrahedron_signed_volume, Point3,
};
use runmat_simulation_usage::{OperationCount, UsageCategory, UsageContext};

use super::{
    topology::{boundary_face_map, sorted_face, tetrahedron_faces},
    validate_constrained_cavity_boundary_preserved, validate_constrained_cavity_refill_volume,
    ConstrainedCavity, ConstrainedCavityBoundaryFace, ConstrainedCavityNode,
    ConstrainedCavityRefill, ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityRefillTetrahedronSplitError, ConstrainedCavityValidationError,
};

mod flips;
mod split;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RefillTetrahedronError {
    Rejected(&'static str),
    Accounting(runmat_simulation_usage::LedgerError),
}
pub use flips::{
    flip_refill_tetrahedra_across_shared_face, flip_refill_tetrahedra_around_shared_edge,
};
pub(super) use flips::{
    improve_refill_with_local_flips_with_diagnostics, refill_is_better,
    LocalTetrahedronReconnectionDiagnostics,
};
pub use split::split_refill_tetrahedra_across_shared_face_at_barycentric;

pub(super) fn star_refill_candidate_with_rejection_reason(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    interior_node: ConstrainedCavityNode,
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let mut tetrahedra =
        Vec::<ConstrainedCavityRefillTetrahedron>::with_capacity(cavity.boundary_faces.len());
    for face in &cavity.boundary_faces {
        let node_ids = [
            face.node_ids[0],
            face.node_ids[1],
            face.node_ids[2],
            interior_node.node_id,
        ];
        let points = [
            boundary_nodes[&face.node_ids[0]],
            boundary_nodes[&face.node_ids[1]],
            boundary_nodes[&face.node_ids[2]],
            interior_node.coordinates_m.canonical_value(),
        ];
        let tetrahedron =
            match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
                Ok(tetrahedron) => tetrahedron,
                Err(RefillTetrahedronError::Rejected(reason)) => return Ok(Err(reason)),
                Err(RefillTetrahedronError::Accounting(error)) => {
                    return Err(ConstrainedCavityValidationError::Accounting(error));
                }
            };
        tetrahedra.push(tetrahedron);
    }
    let refill = refill_from_tetrahedra(cavity, tetrahedra, options.volume_relative_tolerance)?;
    Ok(Ok(refill))
}

pub(super) fn raw_refill_tetrahedron(
    node_ids: [u32; 4],
    points: [Point3; 4],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefillTetrahedron>, runmat_simulation_usage::LedgerError> {
    match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
        Ok(tetrahedron) => Ok(Some(tetrahedron)),
        Err(RefillTetrahedronError::Rejected(_)) => Ok(None),
        Err(RefillTetrahedronError::Accounting(error)) => Err(error),
    }
}

pub(super) fn raw_refill_tetrahedron_with_rejection_reason(
    mut node_ids: [u32; 4],
    mut points: [Point3; 4],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavityRefillTetrahedron, RefillTetrahedronError> {
    // One candidate computes signed volume (orient3d 23, /6 one), then six
    // Euclidean edge lengths (six * nine) and one aspect ratio division, then
    // at most four scaled-Jacobian corners (280): maximum 24 + 55 + 280 = 359.
    // Rejection settles the completed 24- or 79-operation prefix.
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Meshing, OperationCount::from_operations(359))
        })
        .transpose()
        .map_err(RefillTetrahedronError::Accounting)?;
    let mut completed = 0;
    let result = (|| {
        let mut signed_volume_m3 = tetrahedron_signed_volume(points);
        completed += 24;
        if signed_volume_m3 < 0.0 {
            node_ids.swap(1, 2);
            points.swap(1, 2);
            signed_volume_m3 = -signed_volume_m3;
        }
        let volume_m3 = signed_volume_m3.abs();
        if volume_m3 < options.min_volume_m3 {
            return Err(RefillTetrahedronError::Rejected(
                "star_tetrahedron_min_volume",
            ));
        }
        let aspect_ratio = tetrahedron_edge_aspect_ratio(points);
        completed += 55;
        if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
            return Err(RefillTetrahedronError::Rejected(
                "star_tetrahedron_aspect_ratio",
            ));
        }
        let exact_scaled_jacobian = match batch.as_mut() {
            Some(owner) => tetrahedron_scaled_jacobian_with_owner(points, owner, &mut completed),
            None => tetrahedron_scaled_jacobian(points),
        }
        .map_err(RefillTetrahedronError::Accounting)?;
        if !exact_scaled_jacobian.is_finite() || exact_scaled_jacobian < options.min_scaled_jacobian
        {
            return Err(RefillTetrahedronError::Rejected(
                "star_tetrahedron_scaled_jacobian",
            ));
        }
        Ok(ConstrainedCavityRefillTetrahedron {
            node_ids,
            volume_m3,
            aspect_ratio,
            exact_scaled_jacobian,
        })
    })();
    if let Some(batch) = &mut batch {
        batch
            .complete(OperationCount::from_operations(completed))
            .map_err(RefillTetrahedronError::Accounting)?;
    }
    result
}

pub(super) fn refill_from_tetrahedra(
    cavity: &ConstrainedCavity,
    tetrahedra: Vec<ConstrainedCavityRefillTetrahedron>,
    volume_relative_tolerance: f64,
) -> Result<ConstrainedCavityRefill, ConstrainedCavityValidationError> {
    let boundary_faces = boundary_faces_from_refill_tetrahedra(cavity, &tetrahedra)?;
    validate_constrained_cavity_boundary_preserved(cavity, &boundary_faces)?;
    // A mathematical sum of n volumes has n-1 additions, including the
    // empty/singleton zero-work cases; Iterator::sum's initial zero is not a
    // reference addition. Admit the known bound before reducing.
    let additions = u64::try_from(tetrahedra.len().saturating_sub(1)).map_err(|_| {
        ConstrainedCavityValidationError::Accounting(
            runmat_simulation_usage::LedgerError::CountOverflow,
        )
    })?;
    let usage = UsageContext::capture();
    let mut batch = if additions == 0 {
        None
    } else {
        usage
            .as_ref()
            .map(|usage| {
                usage.ledger().reserve(
                    UsageCategory::Meshing,
                    OperationCount::from_operations(additions),
                )
            })
            .transpose()
            .map_err(ConstrainedCavityValidationError::Accounting)?
    };
    let total_volume_m3 = tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.volume_m3)
        .sum::<f64>();
    if let Some(batch) = &mut batch {
        batch
            .complete(OperationCount::from_operations(additions))
            .map_err(ConstrainedCavityValidationError::Accounting)?;
    }
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        total_volume_m3,
        volume_relative_tolerance,
    )?;
    Ok(ConstrainedCavityRefill {
        tetrahedra,
        boundary_faces,
        inserted_nodes: Vec::new(),
        total_volume_m3,
    })
}

pub(super) fn boundary_faces_from_refill_tetrahedra(
    cavity: &ConstrainedCavity,
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityValidationError> {
    let cavity_faces = boundary_face_map(&cavity.boundary_faces)?;
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in tetrahedra {
        for face in tetrahedron_faces(tetrahedron.node_ids) {
            *face_counts.entry(sorted_face(face)).or_default() += 1;
        }
    }
    let boundary_faces = face_counts
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(face, _)| {
            cavity_faces
                .get(&face)
                .map(|source| (*source).clone())
                .unwrap_or(ConstrainedCavityBoundaryFace {
                    node_ids: face,
                    outside_tetrahedron_ids: Vec::new(),
                    source_face_id: None,
                    source_edge_ids: [None, None, None],
                    region_ids: Vec::new(),
                })
        })
        .collect::<Vec<_>>();
    Ok(boundary_faces)
}

pub(super) fn record_refill_rejection(
    rejected_by_reason: &mut BTreeMap<String, usize>,
    reason: &str,
) {
    *rejected_by_reason.entry(reason.to_string()).or_default() += 1;
}

pub(super) fn refill_validation_reason(
    error: &ConstrainedCavityValidationError,
) -> Result<&'static str, runmat_simulation_usage::LedgerError> {
    Ok(match error {
        ConstrainedCavityValidationError::Accounting(cause) => return Err(*cause),
        ConstrainedCavityValidationError::InvalidRefillVolume { .. } => "volume_mismatch",
        ConstrainedCavityValidationError::BoundaryFaceCountMismatch { .. } => {
            "boundary_face_count_mismatch"
        }
        ConstrainedCavityValidationError::MissingBoundaryFace { .. } => "missing_boundary_face",
        ConstrainedCavityValidationError::UnexpectedBoundaryFace { .. } => {
            "unexpected_boundary_face"
        }
        ConstrainedCavityValidationError::BoundarySourceFaceMismatch { .. } => {
            "boundary_source_face_mismatch"
        }
        ConstrainedCavityValidationError::BoundarySourceEdgeMismatch { .. } => {
            "boundary_source_edge_mismatch"
        }
        ConstrainedCavityValidationError::BoundaryRegionMismatch { .. } => {
            "boundary_region_mismatch"
        }
        ConstrainedCavityValidationError::BoundaryOutsideTetrahedronMismatch { .. } => {
            "boundary_outside_tetrahedron_mismatch"
        }
        ConstrainedCavityValidationError::EmptyRemovedTetrahedronSet
        | ConstrainedCavityValidationError::InvalidTargetVolume { .. }
        | ConstrainedCavityValidationError::TooFewBoundaryFaces { .. }
        | ConstrainedCavityValidationError::DegenerateBoundaryFace { .. }
        | ConstrainedCavityValidationError::DuplicateBoundaryFace { .. }
        | ConstrainedCavityValidationError::NonManifoldBoundaryEdge { .. }
        | ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary { .. } => "invalid_cavity",
    })
}

#[cfg(test)]
mod orientation_tests {
    use super::*;
    use runmat_simulation_usage::UsageLedger;
    use std::sync::Arc;

    #[test]
    fn raw_candidate_settles_full_and_min_volume_prefix() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        // Each candidate reserves its full bound before it can reject early.
        let budget = OperationCount::from_operations(2 * 359);
        let expected_completed = OperationCount::from_operations(359 + 24);
        let ledger = Arc::new(UsageLedger::new(budget, budget).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        context.run_sync(|| {
            raw_refill_tetrahedron_with_rejection_reason(
                [0, 1, 2, 3],
                points,
                ConstrainedCavityRefillOptions {
                    min_scaled_jacobian: 0.0,
                    ..ConstrainedCavityRefillOptions::default()
                },
            )
            .unwrap();
            assert!(matches!(
                raw_refill_tetrahedron_with_rejection_reason(
                    [0, 1, 2, 3],
                    points,
                    ConstrainedCavityRefillOptions {
                        min_volume_m3: 2.0,
                        ..ConstrainedCavityRefillOptions::default()
                    }
                ),
                Err(RefillTetrahedronError::Rejected(
                    "star_tetrahedron_min_volume"
                ))
            ));
        });
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.completed.meshing, expected_completed);
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
    }

    #[test]
    fn raw_refill_preserves_node_coordinate_pairing_when_orienting_a_tetrahedron() {
        let options = ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.0,
            ..ConstrainedCavityRefillOptions::default()
        };
        let node_ids = [10, 20, 30, 40];
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let forward =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options).unwrap();
        let reverse = raw_refill_tetrahedron_with_rejection_reason(
            [node_ids[0], node_ids[2], node_ids[1], node_ids[3]],
            [points[0], points[2], points[1], points[3]],
            options,
        )
        .unwrap();

        assert_eq!(forward, reverse);
        assert!(forward.exact_scaled_jacobian >= 0.0);
    }
}
