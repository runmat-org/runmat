use std::collections::{BTreeMap, BTreeSet};

use super::{MeshingContractError, PersistentEntityId, PersistentEntityKind, SolverMeshTopology};

pub(super) fn validate_classification(
    topology: &SolverMeshTopology,
    element_ids: &BTreeSet<u64>,
    face_ids: &BTreeSet<u64>,
) -> Result<(), MeshingContractError> {
    if !strictly_increasing_by(&topology.regions, |region| &region.region_id)
        || !strictly_increasing_by(&topology.conformal_interfaces, |interface| {
            &interface.source_face_id
        })
        || !strictly_increasing_by(&topology.contacts, |contact| &contact.contact_id)
    {
        return Err(MeshingContractError::invalid(
            "mesh classification",
            "regions, interfaces, and contacts must be unique and canonically ordered",
        ));
    }

    let region_ids = validate_regions(topology, element_ids)?;
    for element in &topology.volume_elements {
        if !region_ids.contains(&element.region_id) {
            return Err(MeshingContractError::invalid(
                "volume element classification",
                "element region must exist in the canonical region inventory",
            ));
        }
    }
    let mut classified_faces = BTreeSet::new();
    let faces = topology
        .boundary_faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    validate_interfaces(
        topology,
        face_ids,
        &faces,
        &region_ids,
        &mut classified_faces,
    )?;
    validate_contacts(topology, face_ids, &faces, &mut classified_faces)?;
    for face in &topology.boundary_faces {
        let requires_classification = face.role != super::BoundaryFaceRole::Exterior;
        if classified_faces.contains(&face.face_id) != requires_classification {
            return Err(MeshingContractError::invalid(
                "boundary face classification",
                "every interface/contact face must be classified exactly once and exterior faces must remain unclassified",
            ));
        }
    }
    Ok(())
}

fn validate_regions<'a>(
    topology: &'a SolverMeshTopology,
    element_ids: &BTreeSet<u64>,
) -> Result<BTreeSet<&'a PersistentEntityId>, MeshingContractError> {
    let mut classified_elements = BTreeSet::new();
    let mut region_ids = BTreeSet::new();
    for region in &topology.regions {
        validate_region_id(&region.region_id)?;
        if region.element_ids.is_empty()
            || !strictly_increasing(&region.element_ids)
            || !region.element_ids.iter().all(|id| element_ids.contains(id))
            || region
                .element_ids
                .iter()
                .any(|id| !classified_elements.insert(*id))
        {
            return Err(MeshingContractError::invalid(
                "mesh region",
                "element classification must be complete, unique, and canonical",
            ));
        }
        region_ids.insert(&region.region_id);
    }
    if &classified_elements != element_ids {
        return Err(MeshingContractError::invalid(
            "mesh regions",
            "every volume element must belong to exactly one region",
        ));
    }
    Ok(region_ids)
}

fn validate_interfaces(
    topology: &SolverMeshTopology,
    face_ids: &BTreeSet<u64>,
    faces: &BTreeMap<u64, &super::SolverBoundaryFace>,
    region_ids: &BTreeSet<&PersistentEntityId>,
    classified_faces: &mut BTreeSet<u64>,
) -> Result<(), MeshingContractError> {
    for interface in &topology.conformal_interfaces {
        if interface.source_face_id.kind != PersistentEntityKind::Face {
            return Err(MeshingContractError::invalid(
                "conformal interface source face",
                "must identify a persistent face entity",
            ));
        }
        interface.source_face_id.validate()?;
        validate_region_id(&interface.side_a_region_id)?;
        validate_region_id(&interface.side_b_region_id)?;
        if interface.side_a_region_id == interface.side_b_region_id
            || !region_ids.contains(&interface.side_a_region_id)
            || !region_ids.contains(&interface.side_b_region_id)
        {
            return Err(MeshingContractError::invalid(
                "conformal interface",
                "both distinct sides must reference known regions",
            ));
        }
        validate_face_set(
            "conformal interface faces",
            &interface.boundary_face_ids,
            face_ids,
        )?;
        for face_id in &interface.boundary_face_ids {
            let face = faces[face_id];
            if face.role != super::BoundaryFaceRole::ConformalInterface
                || !face.provenance.contains(&interface.source_face_id)
                || !classified_faces.insert(*face_id)
            {
                return Err(MeshingContractError::invalid(
                    "conformal interface faces",
                    "faces must have the interface role and exact source-face provenance exactly once",
                ));
            }
        }
    }
    Ok(())
}

fn validate_contacts(
    topology: &SolverMeshTopology,
    face_ids: &BTreeSet<u64>,
    faces: &BTreeMap<u64, &super::SolverBoundaryFace>,
    classified_faces: &mut BTreeSet<u64>,
) -> Result<(), MeshingContractError> {
    for contact in &topology.contacts {
        if contact.contact_id.kind != PersistentEntityKind::Contact {
            return Err(MeshingContractError::invalid(
                "contact id",
                "must identify a persistent contact entity",
            ));
        }
        contact.contact_id.validate()?;
        validate_face_set(
            "contact primary faces",
            &contact.primary_boundary_face_ids,
            face_ids,
        )?;
        validate_contact_faces(
            faces,
            &contact.primary_boundary_face_ids,
            super::BoundaryFaceRole::ContactPrimary,
            &contact.contact_id,
            classified_faces,
        )?;
        validate_contact_faces(
            faces,
            &contact.secondary_boundary_face_ids,
            super::BoundaryFaceRole::ContactSecondary,
            &contact.contact_id,
            classified_faces,
        )?;
        validate_face_set(
            "contact secondary faces",
            &contact.secondary_boundary_face_ids,
            face_ids,
        )?;
    }
    Ok(())
}

fn validate_contact_faces(
    faces: &BTreeMap<u64, &super::SolverBoundaryFace>,
    face_ids: &[u64],
    role: super::BoundaryFaceRole,
    contact_id: &PersistentEntityId,
    classified_faces: &mut BTreeSet<u64>,
) -> Result<(), MeshingContractError> {
    for face_id in face_ids {
        let face = faces[face_id];
        if face.role != role
            || !face.provenance.contains(contact_id)
            || !classified_faces.insert(*face_id)
        {
            return Err(MeshingContractError::invalid(
                "contact faces",
                "faces must have the declared contact-side role and persistent contact provenance exactly once",
            ));
        }
    }
    Ok(())
}

fn validate_region_id(id: &PersistentEntityId) -> Result<(), MeshingContractError> {
    if id.kind != PersistentEntityKind::Region {
        return Err(MeshingContractError::invalid(
            "region id",
            "must identify a persistent region entity",
        ));
    }
    Ok(id.validate()?)
}

fn validate_face_set(
    field: &str,
    ids: &[u64],
    valid: &BTreeSet<u64>,
) -> Result<(), MeshingContractError> {
    if ids.is_empty() || !strictly_increasing(ids) || !ids.iter().all(|id| valid.contains(id)) {
        return Err(MeshingContractError::invalid(
            field,
            "must be non-empty, canonical, and reference known boundary faces",
        ));
    }
    Ok(())
}

fn strictly_increasing<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn strictly_increasing_by<T, K: Ord>(values: &[T], key: impl Fn(&T) -> &K) -> bool {
    values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1]))
}
