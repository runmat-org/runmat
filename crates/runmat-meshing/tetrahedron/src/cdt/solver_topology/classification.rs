use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::BoundaryFaceRole;

use super::{error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ProjectedFaceClass<'a> {
    pub source_face_id: &'a PersistentEntityId,
    pub role: BoundaryFaceRole,
    pub contact_id: Option<&'a PersistentEntityId>,
    /// Region from which the solver face connectivity is oriented outward.
    pub outward_region_id: &'a PersistentEntityId,
}

type ContactSide<'a> = (
    &'a PersistentEntityId,
    &'a PersistentEntityId,
    BoundaryFaceRole,
);
pub(super) struct ClassificationIndex<'a> {
    faces: Vec<&'a PersistentEntityId>,
    interfaces: Vec<&'a runmat_geometry_core::ExactSharedInterface>,
    contacts: Vec<ContactSide<'a>>,
    _storage: Option<GeometryAllocationReservation<'a>>,
}

impl<'a> ClassificationIndex<'a> {
    pub(super) fn new(
        topology: &'a ExactBRepTopology,
        control: Option<&'a dyn GeometryEvaluationControl>,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let contact_count = topology
            .contacts
            .iter()
            .try_fold(0usize, |count, contact| {
                count
                    .checked_add(contact.side_a_face_ids.len())
                    .and_then(|count| count.checked_add(contact.side_b_face_ids.len()))
                    .ok_or_else(|| invalid("contact side inventory overflow"))
            })?;
        let bytes = topology
            .faces
            .len()
            .checked_mul(std::mem::size_of::<&PersistentEntityId>())
            .and_then(|bytes| {
                topology
                    .interfaces
                    .len()
                    .checked_mul(std::mem::size_of::<
                        &runmat_geometry_core::ExactSharedInterface,
                    >())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                contact_count
                    .checked_mul(std::mem::size_of::<ContactSide<'_>>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .ok_or_else(|| invalid("face classification storage overflow"))?;
        let storage = control
            .map(|control| GeometryAllocationReservation::new(control, bytes as u64))
            .transpose()
            .map_err(error::geometry)?;
        let mut faces = Vec::with_capacity(topology.faces.len());
        faces.extend(topology.faces.iter().map(|face| &face.id));
        faces.sort_unstable();
        let mut interfaces = Vec::with_capacity(topology.interfaces.len());
        interfaces.extend(topology.interfaces.iter());
        interfaces.sort_unstable_by_key(|interface| &interface.face_id);
        let mut contacts = Vec::with_capacity(contact_count);
        for contact in &topology.contacts {
            if let Some(control) = control {
                control.checkpoint().map_err(error::geometry)?;
            }
            contacts.extend(
                contact
                    .side_a_face_ids
                    .iter()
                    .map(|face| (face, &contact.id, BoundaryFaceRole::ContactPrimary)),
            );
            contacts.extend(
                contact
                    .side_b_face_ids
                    .iter()
                    .map(|face| (face, &contact.id, BoundaryFaceRole::ContactSecondary)),
            );
        }
        contacts.sort_unstable_by_key(|side| side.0);
        Ok(Self {
            faces,
            interfaces,
            contacts,
            _storage: storage,
        })
    }

    pub(super) fn classify(
        &self,
        entities: &'a [PersistentEntityId],
        regions: &'a [PersistentEntityId],
    ) -> Result<ProjectedFaceClass<'a>, DelaunaySolverTopologyError> {
        let mut source_faces = entities
            .iter()
            .filter(|entity| entity.kind == PersistentEntityKind::Face);
        let source_face_id = source_faces.next().ok_or_else(|| {
            invalid("protected facet must identify exactly one exact source face")
        })?;
        if source_faces.next().is_some() {
            return Err(invalid(
                "protected facet must identify exactly one exact source face",
            ));
        }
        if let Ok(index) = self
            .interfaces
            .binary_search_by(|interface| interface.face_id.cmp(source_face_id))
        {
            let interface = self.interfaces[index];
            let mut expected = [&interface.side_a_region_id, &interface.side_b_region_id];
            expected.sort();
            if !regions.iter().eq(expected) {
                return Err(invalid(format!(
                    "conformal interface facet on {source_face_id:?} does not bind both exact interface regions: facet regions {regions:?}, interface regions {expected:?}"
                )));
            }
            return Ok(ProjectedFaceClass {
                source_face_id,
                role: BoundaryFaceRole::ConformalInterface,
                contact_id: None,
                outward_region_id: &interface.side_a_region_id,
            });
        }
        if let Ok(index) = self
            .contacts
            .binary_search_by_key(&source_face_id, |side| side.0)
        {
            let (_, contact_id, role) = self.contacts[index];
            if regions.len() != 1 || !entities.contains(contact_id) {
                return Err(invalid(
                    "contact facet must bind its exact contact and one incident region",
                ));
            }
            return Ok(ProjectedFaceClass {
                source_face_id,
                role,
                contact_id: Some(contact_id),
                outward_region_id: &regions[0],
            });
        }
        if regions.len() != 1
            || entities
                .iter()
                .any(|entity| entity.kind == PersistentEntityKind::Contact)
            || self.faces.binary_search(&source_face_id).is_err()
        {
            return Err(invalid(
                "exterior facet must bind one known exact face and one incident region",
            ));
        }
        Ok(ProjectedFaceClass {
            source_face_id,
            role: BoundaryFaceRole::Exterior,
            contact_id: None,
            outward_region_id: &regions[0],
        })
    }
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidGeometry, reason)
}
