use std::collections::BTreeMap;

use runmat_meshing_core::quality::predicate::{
    tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_signed_volume, Point3,
};

use super::{
    topology::{boundary_face_map, sorted_face, tetrahedron_faces},
    validate_constrained_cavity_boundary_preserved, validate_constrained_cavity_refill_volume,
    ConstrainedCavity, ConstrainedCavityBoundaryFace, ConstrainedCavityNode,
    ConstrainedCavityRefill, ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityRefillTetrahedronSplitError, ConstrainedCavityValidationError,
};

mod flips;
mod split;
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
            interior_node.coordinates_m,
        ];
        let tetrahedron =
            match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
                Ok(tetrahedron) => tetrahedron,
                Err(reason) => return Ok(Err(reason)),
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
) -> Option<ConstrainedCavityRefillTetrahedron> {
    raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options).ok()
}

pub(super) fn raw_refill_tetrahedron_with_rejection_reason(
    mut node_ids: [u32; 4],
    points: [Point3; 4],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavityRefillTetrahedron, &'static str> {
    let mut signed_volume_m3 = tetrahedron_signed_volume(points);
    if signed_volume_m3 < 0.0 {
        node_ids.swap(1, 2);
        signed_volume_m3 = -signed_volume_m3;
    }
    let volume_m3 = signed_volume_m3.abs();
    if volume_m3 < options.min_volume_m3 {
        return Err("star_tetrahedron_min_volume");
    }
    let aspect_ratio = tetrahedron_edge_aspect_ratio(points);
    if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
        return Err("star_tetrahedron_aspect_ratio");
    }
    let exact_scaled_jacobian = tetrahedron_scaled_jacobian(points);
    if !exact_scaled_jacobian.is_finite() || exact_scaled_jacobian < options.min_scaled_jacobian {
        return Err("star_tetrahedron_scaled_jacobian");
    }
    Ok(ConstrainedCavityRefillTetrahedron {
        node_ids,
        volume_m3,
        aspect_ratio,
        exact_scaled_jacobian,
    })
}

pub(super) fn refill_from_tetrahedra(
    cavity: &ConstrainedCavity,
    tetrahedra: Vec<ConstrainedCavityRefillTetrahedron>,
    volume_relative_tolerance: f64,
) -> Result<ConstrainedCavityRefill, ConstrainedCavityValidationError> {
    let boundary_faces = boundary_faces_from_refill_tetrahedra(cavity, &tetrahedra)?;
    validate_constrained_cavity_boundary_preserved(cavity, &boundary_faces)?;
    let total_volume_m3 = tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.volume_m3)
        .sum::<f64>();
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
        .filter_map(|(face, count)| {
            (count == 1).then(|| {
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

pub(super) fn refill_validation_reason(error: &ConstrainedCavityValidationError) -> &'static str {
    match error {
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
    }
}
