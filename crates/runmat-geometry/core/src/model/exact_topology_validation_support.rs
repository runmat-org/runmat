use std::collections::{BTreeMap, BTreeSet};

use super::{ExactBRepTopology, GeometryContractError, PersistentEntityId, PersistentEntityKind};

pub(super) fn validate_interfaces(
    topology: &ExactBRepTopology,
    faces: &BTreeSet<PersistentEntityId>,
    regions: &BTreeSet<PersistentEntityId>,
) -> Result<(), GeometryContractError> {
    let mut ordered = None;
    for interface in &topology.interfaces {
        if ordered
            .as_ref()
            .is_some_and(|face| face >= &interface.face_id)
        {
            return Err(invalid(
                "shared interfaces",
                "interfaces must be canonical and unique",
            ));
        }
        ordered = Some(interface.face_id.clone());
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
    faces: &BTreeSet<PersistentEntityId>,
) -> Result<(), GeometryContractError> {
    let interface_faces = topology
        .interfaces
        .iter()
        .map(|interface| &interface.face_id)
        .collect::<BTreeSet<_>>();
    collect_ids(
        "contacts",
        PersistentEntityKind::Contact,
        topology.contacts.iter().map(|value| &value.id),
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
            || contact.pairing_contract_digest == [0; 32]
            || contact
                .side_a_face_ids
                .iter()
                .chain(&contact.side_b_face_ids)
                .any(|face| interface_faces.contains(face))
        {
            return Err(invalid(
                "contact pairing",
                "contact sides must be disjoint from each other and conformal interfaces, and bind a nonzero pairing contract",
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
    ids: impl Iterator<Item = &'a PersistentEntityId>,
) -> Result<BTreeSet<PersistentEntityId>, GeometryContractError> {
    let mut set = BTreeSet::new();
    let mut previous = None;
    for id in ids {
        require_kind(field, id, kind)?;
        if previous.as_ref().is_some_and(|value| value >= id) || !set.insert(id.clone()) {
            return Err(invalid(
                field,
                "entities must be strictly canonical and unique",
            ));
        }
        previous = Some(id.clone());
    }
    Ok(set)
}

pub(super) fn require_ordered_refs(
    field: &str,
    ids: &[PersistentEntityId],
    kind: PersistentEntityKind,
    known: &BTreeSet<PersistentEntityId>,
    nonempty: bool,
) -> Result<(), GeometryContractError> {
    if nonempty && ids.is_empty() {
        return Err(invalid(field, "reference list must not be empty"));
    }
    let collected = collect_ids(field, kind, ids.iter())?;
    if !collected.is_subset(known) {
        return Err(invalid(field, "reference list contains an unknown entity"));
    }
    Ok(())
}

pub(super) fn require_reference(
    field: &str,
    id: &PersistentEntityId,
    kind: PersistentEntityKind,
    known: &BTreeSet<PersistentEntityId>,
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
    claimed: &mut BTreeSet<PersistentEntityId>,
) -> Result<(), GeometryContractError> {
    if ids.iter().any(|id| !claimed.insert(id.clone())) {
        return Err(invalid(field, "entity has multiple topology owners"));
    }
    Ok(())
}

pub(super) fn claim_wire(
    owners: &mut BTreeMap<PersistentEntityId, PersistentEntityId>,
    wire: &PersistentEntityId,
    face: &PersistentEntityId,
) -> Result<(), GeometryContractError> {
    if owners.insert(wire.clone(), face.clone()).is_some() {
        return Err(invalid("wire ownership", "wire has multiple face owners"));
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
