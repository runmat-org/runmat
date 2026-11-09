//! The cross-section mesh as a canonical solver mesh artifact.
//!
//! The solve, the publication and Desktop read one mesh: the triangles the potential solves
//! ran on, published as a surface (`MeshDimension::Surface`) artifact of linear boundary
//! triangles in the z = 0 plane. Its geometry identity comes from the section digest, so an
//! edited stackup, trace or mesh control is a new geometry revision.
use std::collections::BTreeMap;

use runmat_geometry_core::{
    CrossSectionDefinition, LengthUnit, PersistentEntityId, PersistentEntityKind,
};
use runmat_meshing_core::{
    solver_boundary_edge_identity, solver_boundary_face_identity, BoundaryEdgeOrder,
    BoundaryFaceRole, BoundaryTriangleOrder, CrossSectionMeshSettings, FieldTopologyLocation,
    FieldTopologyMap, GeometryRevisionRef, MeshDimension, MeshingRequest, SolverBoundaryEdge,
    SolverBoundaryFace, SolverMeshArtifact, SolverMeshNode, SolverMeshTopology, StableDigest,
    SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
};
use sha2::{Digest as _, Sha256};

use super::{geometry::Geometry, mesh::CrossSectionMesh, solve, CrossSectionError};

const SECTION_DOMAIN: &[u8] = b"runmat-cross-section/v1\0";
const NODE_DOMAIN: &[u8] = b"runmat-cross-section-node/v1\0";
const STAGE_DOMAIN: &[u8] = b"runmat-cross-section-stage/v1\0";

/// Digest of a checked section and its mesh controls: the identity of the meshed geometry.
pub fn cross_section_digest(
    definition: &CrossSectionDefinition,
    mesh: Option<&CrossSectionMeshSettings>,
) -> Result<StableDigest, CrossSectionError> {
    let bytes = serde_json::to_vec(&(definition, mesh))
        .map_err(|error| CrossSectionError::Invalid(format!("section encoding: {error}")))?;
    let mut hasher = Sha256::new();
    hasher.update(SECTION_DOMAIN);
    hasher.update(bytes);
    Ok(StableDigest::from_bytes(hasher.finalize().into()))
}

/// The geometry revision of a section: the leading digest bytes, never zero.
pub fn cross_section_geometry_revision(digest: &StableDigest) -> u32 {
    let bytes = digest.bytes();
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]).max(1)
}

/// The geometry identity every artifact of this section carries.
pub fn cross_section_geometry(digest: StableDigest) -> GeometryRevisionRef {
    GeometryRevisionRef {
        source_unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(
            LengthUnit::Meter,
            "cross-section lengths are canonical metres",
        ),
        source_digest: digest,
        geometry_revision: u64::from(cross_section_geometry_revision(&digest)),
        persistent_mapping_version: 1,
    }
}

fn entity(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, name.to_string(), Vec::new())
}

/// Build the canonical artifact of the mesh the solves ran on. `request` is the study's
/// resolved meshing request, with its size limits from the admitted run budget.
pub(super) fn solver_mesh_artifact(
    geometry: &Geometry,
    mesh: &CrossSectionMesh,
    digest: StableDigest,
    mut request: MeshingRequest,
) -> Result<SolverMeshArtifact, CrossSectionError> {
    let invalid = |error: runmat_meshing_core::MeshingContractError| {
        CrossSectionError::Invalid(format!("cross-section solver mesh: {error}"))
    };
    request.dimension = MeshDimension::Surface;
    request.element_order = runmat_meshing_core::ElementOrder::Tet4;
    request.size_controls.clear();
    if mesh.x.len() as u64 > request.limits.maximum_nodes
        || mesh.triangles.len() as u64 > request.limits.maximum_elements
    {
        return Err(CrossSectionError::Invalid(format!(
            "the cross-section mesh has {} nodes and {} triangles, above the admitted {} nodes and {} elements; coarsen `edge_size` or raise the run budget",
            mesh.x.len(),
            mesh.triangles.len(),
            request.limits.maximum_nodes,
            request.limits.maximum_elements
        )));
    }
    let section = entity(PersistentEntityKind::Face, "cross_section");
    let signals = geometry.signal_names();
    let conductors: Vec<PersistentEntityId> = signals
        .iter()
        .map(|name| entity(PersistentEntityKind::Face, &format!("trace:{name}")))
        .collect();
    let ground = entity(PersistentEntityKind::Face, "ground");
    let holders = solve::holders(geometry, mesh);
    let nodes: Vec<SolverMeshNode> = (0..mesh.x.len())
        .map(|index| {
            let mut hasher = Sha256::new();
            hasher.update(NODE_DOMAIN);
            hasher.update(digest.bytes());
            hasher.update((index as u64).to_be_bytes());
            let mut provenance = vec![section.clone()];
            match holders[index] {
                Some(Some(k)) => provenance.push(conductors[k].clone()),
                Some(None) => provenance.push(ground.clone()),
                None => {}
            }
            provenance.sort();
            SolverMeshNode {
                node_id: index as u64 + 1,
                stable_identity: StableDigest::from_bytes(hasher.finalize().into()),
                coordinates_m: runmat_types::physical::Position::new([
                    mesh.x[index],
                    mesh.y[index],
                    0.0,
                ]),
                provenance,
                exact_parameters: Vec::new(),
            }
        })
        .collect();
    let identity_of = |id: u64| nodes[id as usize - 1].stable_identity;
    let boundary_faces: Vec<SolverBoundaryFace> = mesh
        .triangles
        .iter()
        .enumerate()
        .map(|(index, corners)| {
            let node_ids = corners.map(|corner| corner as u64 + 1);
            SolverBoundaryFace {
                face_id: index as u64 + 1,
                stable_identity: solver_boundary_face_identity(node_ids.map(identity_of)),
                order: BoundaryTriangleOrder::Tri3,
                node_ids: node_ids.to_vec(),
                adjacent_volume_element_ids: Vec::new(),
                role: BoundaryFaceRole::Exterior,
                provenance: vec![section.clone()],
            }
        })
        .collect();
    let mut edge_rows = BTreeMap::<[u64; 2], Vec<u64>>::new();
    for face in &boundary_faces {
        for [left, right] in [[0, 1], [1, 2], [2, 0]] {
            let mut pair = [face.node_ids[left], face.node_ids[right]];
            pair.sort_unstable();
            edge_rows.entry(pair).or_default().push(face.face_id);
        }
    }
    let boundary_edges: Vec<SolverBoundaryEdge> = edge_rows
        .into_iter()
        .enumerate()
        .map(|(index, (pair, mut adjacent))| {
            adjacent.sort_unstable();
            adjacent.dedup();
            SolverBoundaryEdge {
                edge_id: index as u64 + 1,
                stable_identity: solver_boundary_edge_identity(pair.map(identity_of)),
                order: BoundaryEdgeOrder::Line2,
                node_ids: pair.to_vec(),
                adjacent_boundary_face_ids: adjacent,
                provenance: vec![section.clone()],
            }
        })
        .collect();
    let field_topologies = [
        ("nodes", FieldTopologyLocation::Node, nodes.len()),
        ("elements", FieldTopologyLocation::VolumeElement, 0),
        (
            "boundary_faces",
            FieldTopologyLocation::BoundaryFace,
            boundary_faces.len(),
        ),
        (
            "boundary_edges",
            FieldTopologyLocation::BoundaryEdge,
            boundary_edges.len(),
        ),
    ]
    .into_iter()
    .map(|(topology_id, location, count)| FieldTopologyMap {
        topology_id: topology_id.into(),
        location,
        ordered_entity_ids: (1..=count as u64).collect(),
    })
    .collect();
    let mut stage = Sha256::new();
    stage.update(STAGE_DOMAIN);
    stage.update(digest.bytes());
    let mut artifact = SolverMeshArtifact {
        schema_version: SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
        canonical_digest: StableDigest::ZERO,
        root_stage_manifest_digest: StableDigest::from_bytes(stage.finalize().into()),
        geometry: cross_section_geometry(digest),
        resolved_request: request,
        topology: SolverMeshTopology {
            nodes,
            volume_elements: Vec::new(),
            neighbors: Vec::new(),
            boundary_faces,
            boundary_edges,
            regions: Vec::new(),
            conformal_interfaces: Vec::new(),
            contacts: Vec::new(),
            field_topologies,
        }
        .into(),
    };
    artifact.seal_canonical_digest().map_err(invalid)?;
    Ok(artifact)
}
