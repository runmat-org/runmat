use runmat_meshing_core::{
    ConformalInterface, ContactPair, FieldTopologyLocation, FieldTopologyMap, MeshRegion,
};

use super::{
    classification::ProjectedFaceClass, error, DelaunaySolverTopologyError,
    DelaunaySolverTopologyErrorKind, ProjectionInput,
};

pub(super) fn build_regions(
    input: &ProjectionInput<'_>,
) -> Result<Vec<MeshRegion>, DelaunaySolverTopologyError> {
    let storage = super::storage::Storage::for_input(input);
    let mut result = storage.vector(input.volume_mesh.topology.incidence.regions.len())?;
    for region in &input.volume_mesh.topology.incidence.regions {
        result.push(MeshRegion {
            region_id: storage.entity(&region.region_id)?,
            element_ids: storage.collect(
                region
                    .tetrahedron_indices
                    .iter()
                    .map(|index| *index as u64 + 1),
            )?,
        });
    }
    Ok(result)
}

pub(super) fn build_interfaces(
    input: &ProjectionInput<'_>,
    classes: &[ProjectedFaceClass],
) -> Result<Vec<ConformalInterface>, DelaunaySolverTopologyError> {
    let storage = super::storage::Storage::for_input(input);
    let mut result = storage.vector(input.exact_topology.interfaces.len())?;
    for interface in &input.exact_topology.interfaces {
        let boundary_face_ids = storage.collect(
            classes
                .iter()
                .enumerate()
                .filter(|(_, class)| class.source_face_id == &interface.face_id)
                .map(|(index, _)| index as u64 + 1),
        )?;
        if boundary_face_ids.is_empty() {
            return Err(invalid_mesh(
                "exact conformal interface has no solver boundary faces",
            ));
        }
        result.push(ConformalInterface {
            source_face_id: storage.entity(&interface.face_id)?,
            side_a_region_id: storage.entity(&interface.side_a_region_id)?,
            side_b_region_id: storage.entity(&interface.side_b_region_id)?,
            boundary_face_ids,
        });
    }
    Ok(result)
}

pub(super) fn build_contacts(
    input: &ProjectionInput<'_>,
    classes: &[ProjectedFaceClass],
) -> Result<Vec<ContactPair>, DelaunaySolverTopologyError> {
    let storage = super::storage::Storage::for_input(input);
    let mut result = storage.vector(input.exact_topology.contacts.len())?;
    for contact in &input.exact_topology.contacts {
        let collect = |role| {
            storage.collect(
                classes
                    .iter()
                    .enumerate()
                    .filter(|(_, class)| {
                        class.contact_id == Some(&contact.id) && class.role == role
                    })
                    .map(|(index, _)| index as u64 + 1),
            )
        };
        let primary_boundary_face_ids =
            collect(runmat_meshing_core::BoundaryFaceRole::ContactPrimary)?;
        let secondary_boundary_face_ids =
            collect(runmat_meshing_core::BoundaryFaceRole::ContactSecondary)?;
        if primary_boundary_face_ids.is_empty() || secondary_boundary_face_ids.is_empty() {
            return Err(invalid_mesh(
                "exact contact must project nonempty primary and secondary solver faces",
            ));
        }
        result.push(ContactPair {
            contact_id: storage.entity(&contact.id)?,
            primary_boundary_face_ids,
            secondary_boundary_face_ids,
        });
    }
    Ok(result)
}

pub(super) fn field_topologies(
    input: &ProjectionInput<'_>,
    nodes: usize,
    elements: usize,
    faces: usize,
    edges: usize,
) -> Result<Vec<FieldTopologyMap>, DelaunaySolverTopologyError> {
    let storage = super::storage::Storage::for_input(input);
    let mut result = storage.vector(4)?;
    for (topology_id, location, count) in [
        ("nodes", FieldTopologyLocation::Node, nodes),
        ("elements", FieldTopologyLocation::VolumeElement, elements),
        ("boundary_faces", FieldTopologyLocation::BoundaryFace, faces),
        ("boundary_edges", FieldTopologyLocation::BoundaryEdge, edges),
    ] {
        result.push(FieldTopologyMap {
            topology_id: storage.string(topology_id)?,
            location,
            ordered_entity_ids: storage.collect(1..=count as u64)?,
        });
    }
    Ok(result)
}

fn invalid_mesh(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
