use std::collections::BTreeMap;

use runmat_meshing_core::{
    BoundaryEdgeOrder, BoundaryFaceRole, BoundaryTriangleOrder, PersistentEntityId,
    SolverMeshArtifact, StableDigest,
};

use super::{error, DelaunayAdaptiveTransferError, DelaunayAdaptiveTransferErrorKind};

pub(super) fn require_unchanged_boundaries(
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
) -> Result<(), DelaunayAdaptiveTransferError> {
    if face_signatures(source) != face_signatures(target)
        || edge_signatures(source) != edge_signatures(target)
        || interface_signatures(source) != interface_signatures(target)
        || contact_signatures(source) != contact_signatures(target)
    {
        return Err(error(
            DelaunayAdaptiveTransferErrorKind::UnsupportedBoundaryChange,
            "interior adaptive lineage must preserve exact boundary and contact semantics",
        ));
    }
    Ok(())
}

fn node_identities(artifact: &SolverMeshArtifact) -> BTreeMap<u64, StableDigest> {
    artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.stable_identity))
        .collect()
}

fn face_identities(artifact: &SolverMeshArtifact) -> BTreeMap<u64, StableDigest> {
    artifact
        .topology
        .boundary_faces
        .iter()
        .map(|face| (face.face_id, face.stable_identity))
        .collect()
}

#[derive(PartialEq)]
struct FaceSignature {
    identity: StableDigest,
    order: BoundaryTriangleOrder,
    node_identities: Vec<StableDigest>,
    role: BoundaryFaceRole,
    provenance: Vec<PersistentEntityId>,
}

fn face_signatures(artifact: &SolverMeshArtifact) -> Vec<FaceSignature> {
    let nodes = node_identities(artifact);
    artifact
        .topology
        .boundary_faces
        .iter()
        .map(|face| FaceSignature {
            identity: face.stable_identity,
            order: face.order,
            node_identities: face.node_ids.iter().map(|node| nodes[node]).collect(),
            role: face.role,
            provenance: face.provenance.clone(),
        })
        .collect()
}

#[derive(PartialEq)]
struct EdgeSignature {
    identity: StableDigest,
    order: BoundaryEdgeOrder,
    node_identities: Vec<StableDigest>,
    provenance: Vec<PersistentEntityId>,
}

fn edge_signatures(artifact: &SolverMeshArtifact) -> Vec<EdgeSignature> {
    let nodes = node_identities(artifact);
    artifact
        .topology
        .boundary_edges
        .iter()
        .map(|edge| EdgeSignature {
            identity: edge.stable_identity,
            order: edge.order,
            node_identities: edge.node_ids.iter().map(|node| nodes[node]).collect(),
            provenance: edge.provenance.clone(),
        })
        .collect()
}

#[derive(PartialEq)]
struct InterfaceSignature {
    source_face_id: PersistentEntityId,
    side_a_region_id: PersistentEntityId,
    side_b_region_id: PersistentEntityId,
    boundary_face_identities: Vec<StableDigest>,
}

fn interface_signatures(artifact: &SolverMeshArtifact) -> Vec<InterfaceSignature> {
    let faces = face_identities(artifact);
    artifact
        .topology
        .conformal_interfaces
        .iter()
        .map(|interface| InterfaceSignature {
            source_face_id: interface.source_face_id.clone(),
            side_a_region_id: interface.side_a_region_id.clone(),
            side_b_region_id: interface.side_b_region_id.clone(),
            boundary_face_identities: interface
                .boundary_face_ids
                .iter()
                .map(|face| faces[face])
                .collect(),
        })
        .collect()
}

#[derive(PartialEq)]
struct ContactSignature {
    contact_id: PersistentEntityId,
    primary_face_identities: Vec<StableDigest>,
    secondary_face_identities: Vec<StableDigest>,
}

fn contact_signatures(artifact: &SolverMeshArtifact) -> Vec<ContactSignature> {
    let faces = face_identities(artifact);
    artifact
        .topology
        .contacts
        .iter()
        .map(|contact| ContactSignature {
            contact_id: contact.contact_id.clone(),
            primary_face_identities: contact
                .primary_boundary_face_ids
                .iter()
                .map(|face| faces[face])
                .collect(),
            secondary_face_identities: contact
                .secondary_boundary_face_ids
                .iter()
                .map(|face| faces[face])
                .collect(),
        })
        .collect()
}
