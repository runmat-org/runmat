use std::collections::BTreeMap;

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::BoundaryFaceRole;

use super::{error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ProjectedFaceClass {
    pub source_face_id: PersistentEntityId,
    pub role: BoundaryFaceRole,
    pub contact_id: Option<PersistentEntityId>,
    /// Region from which the solver face connectivity is oriented outward.
    pub outward_region_id: PersistentEntityId,
}

pub(super) struct ClassificationIndex<'a> {
    topology: &'a ExactBRepTopology,
    interfaces: BTreeMap<&'a PersistentEntityId, &'a runmat_geometry_core::ExactSharedInterface>,
    contacts: BTreeMap<&'a PersistentEntityId, (&'a PersistentEntityId, BoundaryFaceRole)>,
}

impl<'a> ClassificationIndex<'a> {
    pub(super) fn new(topology: &'a ExactBRepTopology) -> Self {
        let interfaces = topology
            .interfaces
            .iter()
            .map(|interface| (&interface.face_id, interface))
            .collect();
        let mut contacts = BTreeMap::new();
        for contact in &topology.contacts {
            for face in &contact.side_a_face_ids {
                contacts.insert(face, (&contact.id, BoundaryFaceRole::ContactPrimary));
            }
            for face in &contact.side_b_face_ids {
                contacts.insert(face, (&contact.id, BoundaryFaceRole::ContactSecondary));
            }
        }
        Self {
            topology,
            interfaces,
            contacts,
        }
    }

    pub(super) fn classify(
        &self,
        entities: &[PersistentEntityId],
        regions: &[PersistentEntityId],
    ) -> Result<ProjectedFaceClass, DelaunaySolverTopologyError> {
        let source_faces = entities
            .iter()
            .filter(|entity| entity.kind == PersistentEntityKind::Face)
            .collect::<Vec<_>>();
        let [source_face_id] = source_faces.as_slice() else {
            return Err(invalid(
                "protected facet must identify exactly one exact source face",
            ));
        };
        if let Some(interface) = self.interfaces.get(source_face_id) {
            let mut expected = [
                interface.side_a_region_id.clone(),
                interface.side_b_region_id.clone(),
            ];
            expected.sort();
            if regions != expected {
                return Err(invalid(
                    "conformal interface facet does not bind both exact interface regions",
                ));
            }
            return Ok(ProjectedFaceClass {
                source_face_id: (*source_face_id).clone(),
                role: BoundaryFaceRole::ConformalInterface,
                contact_id: None,
                outward_region_id: interface.side_a_region_id.clone(),
            });
        }
        if let Some((contact_id, role)) = self.contacts.get(source_face_id) {
            if regions.len() != 1 || !entities.contains(contact_id) {
                return Err(invalid(
                    "contact facet must bind its exact contact and one incident region",
                ));
            }
            return Ok(ProjectedFaceClass {
                source_face_id: (*source_face_id).clone(),
                role: *role,
                contact_id: Some((*contact_id).clone()),
                outward_region_id: regions[0].clone(),
            });
        }
        if regions.len() != 1
            || entities
                .iter()
                .any(|entity| entity.kind == PersistentEntityKind::Contact)
            || !self
                .topology
                .faces
                .iter()
                .any(|face| face.id == **source_face_id)
        {
            return Err(invalid(
                "exterior facet must bind one known exact face and one incident region",
            ));
        }
        Ok(ProjectedFaceClass {
            source_face_id: (*source_face_id).clone(),
            role: BoundaryFaceRole::Exterior,
            contact_id: None,
            outward_region_id: regions[0].clone(),
        })
    }
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidGeometry, reason)
}
