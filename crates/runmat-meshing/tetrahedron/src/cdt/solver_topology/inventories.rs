use runmat_meshing_core::{
    ContactPair, FieldTopologyLocation, FieldTopologyMap, MaterialInterface, MeshRegion,
};

use super::{
    classification::ProjectedFaceClass, error, DelaunaySolverTopologyError,
    DelaunaySolverTopologyErrorKind, DelaunaySolverTopologyInput,
};

pub(super) fn build_regions(
    input: &DelaunaySolverTopologyInput<'_>,
    materials: &std::collections::BTreeMap<&runmat_geometry_core::PersistentEntityId, &str>,
) -> Result<Vec<MeshRegion>, DelaunaySolverTopologyError> {
    input
        .volume_mesh
        .topology
        .incidence
        .regions
        .iter()
        .map(|region| {
            let material = materials
                .get(&region.region_id)
                .ok_or_else(|| invalid_materials("mesh region has no material assignment"))?;
            Ok(MeshRegion {
                region_id: region.region_id.clone(),
                material_id: (*material).to_owned(),
                element_ids: region
                    .tetrahedron_indices
                    .iter()
                    .map(|index| *index as u64 + 1)
                    .collect(),
            })
        })
        .collect()
}

pub(super) fn build_interfaces(
    input: &DelaunaySolverTopologyInput<'_>,
    classes: &[ProjectedFaceClass],
) -> Result<Vec<MaterialInterface>, DelaunaySolverTopologyError> {
    input
        .exact_topology
        .interfaces
        .iter()
        .map(|interface| {
            let boundary_face_ids = classes
                .iter()
                .enumerate()
                .filter(|(_, class)| class.source_face_id == interface.face_id)
                .map(|(index, _)| index as u64 + 1)
                .collect::<Vec<_>>();
            if boundary_face_ids.is_empty() {
                return Err(invalid_mesh(
                    "exact conformal interface has no solver boundary faces",
                ));
            }
            Ok(MaterialInterface {
                source_face_id: interface.face_id.clone(),
                side_a_region_id: interface.side_a_region_id.clone(),
                side_b_region_id: interface.side_b_region_id.clone(),
                boundary_face_ids,
            })
        })
        .collect()
}

pub(super) fn build_contacts(
    input: &DelaunaySolverTopologyInput<'_>,
    classes: &[ProjectedFaceClass],
) -> Result<Vec<ContactPair>, DelaunaySolverTopologyError> {
    input
        .exact_topology
        .contacts
        .iter()
        .map(|contact| {
            let collect = |role| {
                classes
                    .iter()
                    .enumerate()
                    .filter(|(_, class)| {
                        class.contact_id.as_ref() == Some(&contact.id) && class.role == role
                    })
                    .map(|(index, _)| index as u64 + 1)
                    .collect::<Vec<_>>()
            };
            let primary_boundary_face_ids =
                collect(runmat_meshing_core::BoundaryFaceRole::ContactPrimary);
            let secondary_boundary_face_ids =
                collect(runmat_meshing_core::BoundaryFaceRole::ContactSecondary);
            if primary_boundary_face_ids.is_empty() || secondary_boundary_face_ids.is_empty() {
                return Err(invalid_mesh(
                    "exact contact must project nonempty primary and secondary solver faces",
                ));
            }
            Ok(ContactPair {
                contact_id: contact.id.clone(),
                primary_boundary_face_ids,
                secondary_boundary_face_ids,
            })
        })
        .collect()
}

pub(super) fn field_topologies(
    nodes: usize,
    elements: usize,
    faces: usize,
    edges: usize,
) -> Vec<FieldTopologyMap> {
    [
        ("nodes", FieldTopologyLocation::Node, nodes),
        ("elements", FieldTopologyLocation::VolumeElement, elements),
        ("boundary_faces", FieldTopologyLocation::BoundaryFace, faces),
        ("boundary_edges", FieldTopologyLocation::BoundaryEdge, edges),
    ]
    .into_iter()
    .map(|(topology_id, location, count)| FieldTopologyMap {
        topology_id: topology_id.to_owned(),
        location,
        ordered_entity_ids: (1..=count as u64).collect(),
    })
    .collect()
}

fn invalid_mesh(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}

fn invalid_materials(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMaterials, reason)
}
