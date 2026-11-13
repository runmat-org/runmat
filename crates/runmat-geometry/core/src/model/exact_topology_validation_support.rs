use super::validation_storage::{Claims, IdIndex, ValidationWork};

use super::{
    exact_contacts::contact_pair_matches, ExactBRepTopology, GeometryContractError,
    PersistentEntityId, PersistentEntityKind,
};

pub(super) fn validate_interfaces(
    topology: &ExactBRepTopology,
    faces: &IdIndex<'_>,
    regions: &IdIndex<'_>,
) -> Result<(), GeometryContractError> {
    let mut ordered = None;
    for interface in &topology.interfaces {
        if ordered.is_some_and(|face| face >= &interface.face_id) {
            return Err(invalid(
                "shared interfaces",
                "interfaces must be canonical and unique",
            ));
        }
        ordered = Some(&interface.face_id);
        require_reference(
            "shared interface face",
            &interface.face_id,
            PersistentEntityKind::Face,
            faces,
        )?;
        for region in [&interface.side_a_region_id, &interface.side_b_region_id] {
            require_reference(
                "shared interface region",
                region,
                PersistentEntityKind::Region,
                regions,
            )?;
        }
        if interface.side_a_region_id == interface.side_b_region_id
            || interface.side_a_orientation == interface.side_b_orientation
        {
            return Err(invalid(
                "shared interface orientation",
                "one face must have distinct regions and opposite oriented uses",
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_contacts(
    topology: &ExactBRepTopology,
    faces: &IdIndex<'_>,
    work: ValidationWork<'_>,
) -> Result<(), GeometryContractError> {
    let mut claimed_faces = Claims::new(faces, work)?;
    collect_ids(
        "contacts",
        PersistentEntityKind::Contact,
        topology.contacts.iter().map(|value| &value.id),
        work,
    )?;
    for contact in &topology.contacts {
        require_ordered_refs(
            "contact side A faces",
            &contact.side_a_face_ids,
            PersistentEntityKind::Face,
            faces,
            true,
        )?;
        require_ordered_refs(
            "contact side B faces",
            &contact.side_b_face_ids,
            PersistentEntityKind::Face,
            faces,
            true,
        )?;
        if !ordered_sets_are_disjoint(&contact.side_a_face_ids, &contact.side_b_face_ids)
            || contact.side_b_face_ids < contact.side_a_face_ids
            || contact
                .side_a_face_ids
                .iter()
                .chain(&contact.side_b_face_ids)
                .any(|face| {
                    topology
                        .interfaces
                        .binary_search_by(|interface| interface.face_id.cmp(face))
                        .is_ok()
                })
        {
            return Err(invalid(
                "contact pairing",
                "contact sides must be canonical, disjoint from each other and conformal interfaces, and bind their source-face pairing contract",
            ));
        }
        if contact
            .side_a_face_ids
            .iter()
            .chain(&contact.side_b_face_ids)
            .any(|face| !claimed_faces.insert(face))
        {
            return Err(invalid(
                "contact face ownership",
                "an exact face cannot belong to more than one contact pair",
            ));
        }
        if !contact_pair_matches(&topology.root_assembly_id.assembly_path, contact) {
            return Err(invalid(
                "contact pairing contract",
                "contact identity and pairing digest must derive from the canonical source-face sides",
            ));
        }
    }
    Ok(())
}

fn ordered_sets_are_disjoint(left: &[PersistentEntityId], right: &[PersistentEntityId]) -> bool {
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => return false,
        }
    }
    true
}

pub(super) fn collect_ids<'a>(
    field: &str,
    kind: PersistentEntityKind,
    ids: impl ExactSizeIterator<Item = &'a PersistentEntityId>,
    work: ValidationWork<'a>,
) -> Result<IdIndex<'a>, GeometryContractError> {
    let values = IdIndex::collect(ids, work)?;
    let mut prior = None;
    for id in values.iter() {
        require_kind(field, id, kind)?;
        if prior.is_some_and(|value| value >= id) {
            return Err(invalid(
                field,
                "entities must be strictly canonical and unique",
            ));
        }
        prior = Some(id);
    }
    Ok(values)
}
pub(super) fn require_ordered_refs(
    field: &str,
    ids: &[PersistentEntityId],
    kind: PersistentEntityKind,
    known: &IdIndex<'_>,
    nonempty: bool,
) -> Result<(), GeometryContractError> {
    require_ordered_ref_iter(field, ids.iter(), kind, known, nonempty)
}
pub(super) fn require_ordered_ref_iter<'a>(
    field: &str,
    ids: impl Iterator<Item = &'a PersistentEntityId>,
    kind: PersistentEntityKind,
    known: &IdIndex<'_>,
    nonempty: bool,
) -> Result<(), GeometryContractError> {
    let mut prior = None;
    for id in ids {
        require_reference(field, id, kind, known)?;
        if prior.is_some_and(|value| value >= id) {
            return Err(invalid(
                field,
                "entities must be strictly canonical and unique",
            ));
        }
        prior = Some(id);
    }
    if nonempty && prior.is_none() {
        return Err(invalid(field, "reference list must not be empty"));
    }
    Ok(())
}
/// Preserve cycle traversal, checking uniqueness using bounded borrowed scratch.
pub(super) fn require_canonical_cycle_refs<'a>(
    field: &str,
    ids: &'a [PersistentEntityId],
    kind: PersistentEntityKind,
    known: &IdIndex<'_>,
    work: ValidationWork<'a>,
) -> Result<(), GeometryContractError> {
    let Some(first) = ids.first() else {
        return Err(invalid(field, "cyclic reference list must not be empty"));
    };
    let mut ordered = work.buffer(ids.len())?;
    for id in ids {
        work.checkpoint()?;
        require_reference(field, id, kind, known)?;
        if id < first {
            return Err(invalid(
                field,
                "cyclic traversal must start at its smallest persistent identity",
            ));
        }
        ordered.push(id);
    }
    ordered.sort_unstable();
    if ordered.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid(field, "cyclic references must be unique"));
    }
    Ok(())
}
pub(super) fn require_reference(
    field: &str,
    id: &PersistentEntityId,
    kind: PersistentEntityKind,
    known: &IdIndex<'_>,
) -> Result<(), GeometryContractError> {
    require_kind(field, id, kind)?;
    if !known.contains(id) {
        return Err(invalid(field, "reference names an unknown entity"));
    }
    Ok(())
}

pub(super) fn require_same_scope<'a>(
    field: &str,
    owner: &PersistentEntityId,
    children: impl IntoIterator<Item = &'a PersistentEntityId>,
) -> Result<(), GeometryContractError> {
    if children
        .into_iter()
        .any(|child| child.assembly_path != owner.assembly_path)
    {
        return Err(invalid(
            field,
            "topology incidence must remain within one assembly occurrence scope",
        ));
    }
    Ok(())
}

pub(super) fn require_kind(
    field: &str,
    id: &PersistentEntityId,
    kind: PersistentEntityKind,
) -> Result<(), GeometryContractError> {
    id.validate()?;
    if id.kind != kind {
        return Err(invalid(field, "persistent entity has the wrong kind"));
    }
    Ok(())
}

pub(super) fn claim_unique(
    field: &str,
    ids: &[PersistentEntityId],
    claimed: &mut Claims<'_, '_>,
) -> Result<(), GeometryContractError> {
    if ids.iter().any(|id| !claimed.insert(id)) {
        return Err(invalid(field, "entity has multiple topology owners"));
    }
    Ok(())
}

pub(super) fn require_count(
    field: &str,
    actual: usize,
    advertised: u64,
) -> Result<(), GeometryContractError> {
    if u64::try_from(actual).ok() != Some(advertised) {
        return Err(invalid(
            field,
            "topology count differs from document summary",
        ));
    }
    Ok(())
}

pub(super) fn validate_transform(transform: &[f64; 16]) -> Result<(), GeometryContractError> {
    let determinant = transform[0] * (transform[5] * transform[10] - transform[6] * transform[9])
        - transform[1] * (transform[4] * transform[10] - transform[6] * transform[8])
        + transform[2] * (transform[4] * transform[9] - transform[5] * transform[8]);
    if transform.iter().any(|value| !value.is_finite())
        || transform[12] != 0.0
        || transform[13] != 0.0
        || transform[14] != 0.0
        || transform[15] != 1.0
        || !determinant.is_finite()
        || determinant == 0.0
    {
        return Err(invalid(
            "instance transform",
            "must be a finite invertible row-major affine transform with canonical final row",
        ));
    }
    Ok(())
}

pub(super) fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}

#[cfg(test)]
mod cycle_tests;
