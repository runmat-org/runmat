#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[cfg(test)]
use super::topology::tetrahedron_faces;
use super::{
    boundary_faces_from_refill_tetrahedra, topology::sorted_face, ConstrainedCavity,
    ConstrainedCavityRefillTetrahedron, ConstrainedCavityValidationError,
};

pub(super) struct RefillBoundaryFaceDelta {
    pub(super) missing: Vec<[u32; 3]>,
    pub(super) unexpected: Vec<[u32; 3]>,
}

pub(super) fn refill_boundary_face_delta(
    cavity: &ConstrainedCavity,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
) -> Result<RefillBoundaryFaceDelta, ConstrainedCavityValidationError> {
    let expected = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let actual = boundary_faces_from_refill_tetrahedra(cavity, refill_tetrahedra)?
        .into_iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    Ok(RefillBoundaryFaceDelta {
        missing: expected.difference(&actual).copied().collect(),
        unexpected: actual.difference(&expected).copied().collect(),
    })
}

#[cfg(test)]
pub(super) fn missing_refill_boundary_faces(
    cavity: &ConstrainedCavity,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
) -> Result<Vec<[u32; 3]>, ConstrainedCavityValidationError> {
    Ok(refill_boundary_face_delta(cavity, refill_tetrahedra)?.missing)
}

#[cfg(test)]
pub(super) fn open_interior_refill_faces(
    cavity: &ConstrainedCavity,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
) -> Vec<[u32; 3]> {
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in refill_tetrahedra {
        for face in tetrahedron_faces(tetrahedron.node_ids).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    face_counts
        .into_iter()
        .filter_map(|(face, count)| (!boundary_faces.contains(&face) && count == 1).then_some(face))
        .collect()
}

#[cfg(test)]
pub(super) fn cap_side_face_mate_counts(
    cap_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    candidate_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    inserted_node_ids: &BTreeSet<u32>,
) -> Vec<usize> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in candidate_tetrahedra {
        for face in tetrahedron_faces(tetrahedron.node_ids).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    let mut mate_counts = Vec::<usize>::new();
    for cap_tetrahedron in cap_tetrahedra {
        for face in tetrahedron_faces(cap_tetrahedron.node_ids).map(sorted_face) {
            if !face
                .iter()
                .any(|node_id| inserted_node_ids.contains(node_id))
            {
                continue;
            }
            mate_counts.push(
                face_counts
                    .get(&face)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(1),
            );
        }
    }
    mate_counts
}

#[cfg(test)]
pub(super) fn candidate_orphan_interior_face_counts(
    cavity: &ConstrainedCavity,
    candidate_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
) -> (usize, usize) {
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in candidate_tetrahedra {
        for face in tetrahedron_faces(tetrahedron.node_ids).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let mut with_orphan = 0_usize;
    let mut without_orphan = 0_usize;
    for tetrahedron in candidate_tetrahedra {
        let has_orphan = tetrahedron_faces(tetrahedron.node_ids)
            .map(sorted_face)
            .into_iter()
            .any(|face| !boundary_faces.contains(&face) && face_counts[&face] == 1);
        if has_orphan {
            with_orphan += 1;
        } else {
            without_orphan += 1;
        }
    }
    (with_orphan, without_orphan)
}
