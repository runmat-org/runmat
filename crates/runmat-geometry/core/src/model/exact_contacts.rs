use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use super::{
    ExactBRepTopology, ExactContactPair, GeometryContractError, PersistentEntityId,
    PersistentEntityKind, EXACT_CONTACT_PAIRING_SCHEMA_VERSION,
};

const MAX_AUTHORED_CONTACTS: usize = 1_000_000;
const MAX_CONTACT_FACE_REFERENCES: usize = 10_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactContactDefinition {
    pub side_a_face_ids: Vec<PersistentEntityId>,
    pub side_b_face_ids: Vec<PersistentEntityId>,
}

/// Resolves explicit user-authored contact sides against exact source faces. Coincidence or
/// proximity never creates a contact implicitly.
pub fn author_exact_contacts(
    topology: &ExactBRepTopology,
    definitions: &[ExactContactDefinition],
) -> Result<Vec<ExactContactPair>, GeometryContractError> {
    validate_exact_contact_definitions(definitions)?;
    let known_faces = topology
        .faces
        .iter()
        .map(|face| face.id.clone())
        .collect::<BTreeSet<_>>();
    let interface_faces = topology
        .interfaces
        .iter()
        .map(|interface| interface.face_id.clone())
        .collect::<BTreeSet<_>>();
    let mut claimed_faces = BTreeSet::new();
    let mut contacts = Vec::with_capacity(definitions.len());

    for definition in definitions {
        let mut side_a = canonical_side(&definition.side_a_face_ids, &known_faces)?;
        let mut side_b = canonical_side(&definition.side_b_face_ids, &known_faces)?;
        if !ordered_sets_are_disjoint(&side_a, &side_b) {
            return Err(invalid("contact sides", "contact sides must be disjoint"));
        }
        if side_a
            .iter()
            .chain(&side_b)
            .any(|face| interface_faces.contains(face))
        {
            return Err(invalid(
                "contact faces",
                "conformal interface faces cannot also be authored as contact faces",
            ));
        }
        if side_b < side_a {
            std::mem::swap(&mut side_a, &mut side_b);
        }
        if side_a
            .iter()
            .chain(&side_b)
            .any(|face| !claimed_faces.insert(face.clone()))
        {
            return Err(invalid(
                "contact face ownership",
                "an exact face cannot belong to more than one contact pair",
            ));
        }
        contacts.push(contact_pair(
            &topology.root_assembly_id.assembly_path,
            side_a,
            side_b,
        ));
    }
    contacts.sort_by(|left, right| left.id.cmp(&right.id));
    if contacts.windows(2).any(|pair| pair[0].id == pair[1].id) {
        return Err(invalid(
            "contact identity",
            "contact definitions must be semantically unique",
        ));
    }
    Ok(contacts)
}

fn validate_exact_contact_definitions(
    definitions: &[ExactContactDefinition],
) -> Result<(), GeometryContractError> {
    let face_reference_count = definitions
        .iter()
        .map(|definition| {
            definition
                .side_a_face_ids
                .len()
                .saturating_add(definition.side_b_face_ids.len())
        })
        .fold(0usize, usize::saturating_add);
    if definitions.len() > MAX_AUTHORED_CONTACTS
        || face_reference_count > MAX_CONTACT_FACE_REFERENCES
    {
        return Err(invalid(
            "contact authoring bounds",
            "contact count or aggregate source-face references exceed the hard bound",
        ));
    }
    for definition in definitions {
        if definition.side_a_face_ids.is_empty() || definition.side_b_face_ids.is_empty() {
            return Err(invalid("contact side", "contact sides must not be empty"));
        }
        for face in definition
            .side_a_face_ids
            .iter()
            .chain(&definition.side_b_face_ids)
        {
            face.validate()?;
            if face.kind != PersistentEntityKind::Face {
                return Err(invalid(
                    "contact side",
                    "contact definitions may contain only exact source-face identities",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn expected_contact_pair(
    root_assembly_path: &[String],
    side_a_face_ids: &[PersistentEntityId],
    side_b_face_ids: &[PersistentEntityId],
) -> ExactContactPair {
    contact_pair(
        root_assembly_path,
        side_a_face_ids.to_vec(),
        side_b_face_ids.to_vec(),
    )
}

fn canonical_side(
    faces: &[PersistentEntityId],
    known_faces: &BTreeSet<PersistentEntityId>,
) -> Result<Vec<PersistentEntityId>, GeometryContractError> {
    let mut canonical = faces.to_vec();
    canonical.sort();
    if canonical.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid(
            "contact side",
            "a contact side cannot repeat a source face",
        ));
    }
    for face in &canonical {
        face.validate()?;
        if face.kind != PersistentEntityKind::Face || !known_faces.contains(face) {
            return Err(invalid(
                "contact side",
                "every contact side entity must resolve to an exact source face",
            ));
        }
    }
    Ok(canonical)
}

fn contact_pair(
    root_assembly_path: &[String],
    side_a_face_ids: Vec<PersistentEntityId>,
    side_b_face_ids: Vec<PersistentEntityId>,
) -> ExactContactPair {
    let pairing_contract_digest = pairing_digest(&side_a_face_ids, &side_b_face_ids);
    let mut identity = Sha256::new();
    identity.update(b"runmat.exact-contact-identity\0");
    identity.update(pairing_contract_digest);
    ExactContactPair {
        id: PersistentEntityId {
            kind: PersistentEntityKind::Contact,
            source_topology_id: format!("contact:{:x}", identity.finalize()),
            assembly_path: root_assembly_path.to_vec(),
        },
        side_a_face_ids,
        side_b_face_ids,
        pairing_schema_version: EXACT_CONTACT_PAIRING_SCHEMA_VERSION,
        pairing_contract_digest,
    }
}

fn pairing_digest(side_a: &[PersistentEntityId], side_b: &[PersistentEntityId]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"runmat.exact-contact-pairing\0");
    digest.update(EXACT_CONTACT_PAIRING_SCHEMA_VERSION.to_be_bytes());
    write_side(&mut digest, side_a);
    write_side(&mut digest, side_b);
    digest.finalize().into()
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

fn write_side(digest: &mut Sha256, faces: &[PersistentEntityId]) {
    digest.update((faces.len() as u64).to_be_bytes());
    for face in faces {
        write_bytes(digest, face.source_topology_id.as_bytes());
        digest.update((face.assembly_path.len() as u64).to_be_bytes());
        for segment in &face.assembly_path {
            write_bytes(digest, segment.as_bytes());
        }
    }
}

fn write_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}

fn invalid(field: &str, reason: &str) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
