use super::{
    artifact::validation_checkpoint, artifact_workspace::flags, MeshingCancellationSignal,
    MeshingContractError, PersistentEntityId, PersistentEntityKind, SolverMeshTopology,
};

pub(super) fn validate_classification(
    topology: &SolverMeshTopology,
    element_ids: &[u64],
    face_ids: &[u64],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    validation_checkpoint(cancellation)?;
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

    let region_ids = validate_regions(topology, element_ids, cancellation)?;
    for element in &topology.volume_elements {
        validation_checkpoint(cancellation)?;
        if region_ids
            .binary_search_by(|region| region.region_id.cmp(&element.region_id))
            .is_err()
        {
            return Err(MeshingContractError::invalid(
                "volume element classification",
                "element region must exist in the canonical region inventory",
            ));
        }
    }
    let mut classified_faces = flags(topology.boundary_faces.len(), cancellation)?;
    let faces = &topology.boundary_faces;
    validate_interfaces(
        topology,
        face_ids,
        faces,
        region_ids,
        &mut classified_faces,
        cancellation,
    )?;
    validate_contacts(
        topology,
        face_ids,
        faces,
        &mut classified_faces,
        cancellation,
    )?;
    for face in &topology.boundary_faces {
        validation_checkpoint(cancellation)?;
        let requires_classification = face.role != super::BoundaryFaceRole::Exterior;
        if (classified_faces[face_ids
            .binary_search(&face.face_id)
            .expect("validated face")]
            != 0)
            != requires_classification
        {
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
    element_ids: &[u64],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<&'a [super::MeshRegion], MeshingContractError> {
    let mut classified_elements = flags(element_ids.len(), cancellation)?;
    for region in &topology.regions {
        validation_checkpoint(cancellation)?;
        validate_region_id(&region.region_id)?;
        if region.element_ids.is_empty()
            || !strictly_increasing(&region.element_ids)
            || !region
                .element_ids
                .iter()
                .all(|id| element_ids.binary_search(id).is_ok())
            || region.element_ids.iter().any(|id| {
                std::mem::replace(
                    &mut classified_elements
                        [element_ids.binary_search(id).expect("validated element")],
                    1,
                ) != 0
            })
        {
            return Err(MeshingContractError::invalid(
                "mesh region",
                "element classification must be complete, unique, and canonical",
            ));
        }
    }
    if classified_elements.contains(&0) {
        return Err(MeshingContractError::invalid(
            "mesh regions",
            "every volume element must belong to exactly one region",
        ));
    }
    Ok(&topology.regions)
}

fn validate_interfaces(
    topology: &SolverMeshTopology,
    face_ids: &[u64],
    faces: &[super::SolverBoundaryFace],
    region_ids: &[super::MeshRegion],
    classified_faces: &mut [u8],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    for interface in &topology.conformal_interfaces {
        validation_checkpoint(cancellation)?;
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
            || region_ids
                .binary_search_by(|region| region.region_id.cmp(&interface.side_a_region_id))
                .is_err()
            || region_ids
                .binary_search_by(|region| region.region_id.cmp(&interface.side_b_region_id))
                .is_err()
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
            validation_checkpoint(cancellation)?;
            let index = faces
                .binary_search_by_key(face_id, |face| face.face_id)
                .expect("validated face");
            let face = &faces[index];
            if face.role != super::BoundaryFaceRole::ConformalInterface
                || !face.provenance.contains(&interface.source_face_id)
                || std::mem::replace(&mut classified_faces[index], 1) != 0
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
    face_ids: &[u64],
    faces: &[super::SolverBoundaryFace],
    classified_faces: &mut [u8],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    for contact in &topology.contacts {
        validation_checkpoint(cancellation)?;
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
            cancellation,
        )?;
        // Validate references before indexing the face inventory, including malformed inputs.
        validate_face_set(
            "contact secondary faces",
            &contact.secondary_boundary_face_ids,
            face_ids,
        )?;
        validate_contact_faces(
            faces,
            &contact.secondary_boundary_face_ids,
            super::BoundaryFaceRole::ContactSecondary,
            &contact.contact_id,
            classified_faces,
            cancellation,
        )?;
    }
    Ok(())
}

fn validate_contact_faces(
    faces: &[super::SolverBoundaryFace],
    face_ids: &[u64],
    role: super::BoundaryFaceRole,
    contact_id: &PersistentEntityId,
    classified_faces: &mut [u8],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    for face_id in face_ids {
        validation_checkpoint(cancellation)?;
        let index = faces
            .binary_search_by_key(face_id, |face| face.face_id)
            .expect("validated face");
        let face = &faces[index];
        if face.role != role
            || !face.provenance.contains(contact_id)
            || std::mem::replace(&mut classified_faces[index], 1) != 0
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

fn validate_face_set(field: &str, ids: &[u64], valid: &[u64]) -> Result<(), MeshingContractError> {
    if ids.is_empty()
        || !strictly_increasing(ids)
        || !ids.iter().all(|id| valid.binary_search(id).is_ok())
    {
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
