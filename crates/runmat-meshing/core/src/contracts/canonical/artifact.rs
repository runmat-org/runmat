use std::collections::{BTreeMap, BTreeSet};

use super::{
    solver_boundary_edge_identity, solver_boundary_face_identity, solver_volume_element_identity,
    validate_finite, validate_token, CanonicalMeshingContract, ElementOrder, FieldTopologyLocation,
    MeshingContractError, MeshingRequest, PersistentEntityId, PersistentEntityKind,
    SolverMeshArtifact, SolverMeshTopology, StableDigest, ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
};

const MAX_ENTITY_PROVENANCE: usize = 32;

impl SolverMeshArtifact {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        self.validate_canonical()
    }

    pub(super) fn validate_payload(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "analysis mesh artifact schema version",
                format!("expected {ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION}"),
            ));
        }
        self.root_stage_manifest_digest
            .validate_nonzero("artifact.root_stage_manifest_digest")?;
        self.geometry.validate()?;
        self.resolved_request.validate()?;
        self.topology.validate(&self.resolved_request)
    }
}

impl SolverMeshTopology {
    pub(super) fn validate(&self, request: &MeshingRequest) -> Result<(), MeshingContractError> {
        require_nonempty("mesh nodes", &self.nodes)?;
        require_nonempty("volume elements", &self.volume_elements)?;
        if self.nodes.len() as u64 > request.resources.maximum_nodes
            || self.volume_elements.len() as u64 > request.resources.maximum_elements
        {
            return Err(MeshingContractError::invalid(
                "mesh topology",
                "node or element count exceeds the resolved hard budget",
            ));
        }

        let node_ids = ordered_ids("mesh nodes", self.nodes.iter().map(|node| node.node_id))?;
        let element_ids = ordered_ids(
            "volume elements",
            self.volume_elements
                .iter()
                .map(|element| element.element_id),
        )?;
        let face_ids = ordered_ids(
            "boundary faces",
            self.boundary_faces.iter().map(|face| face.face_id),
        )?;
        let edge_ids = ordered_ids(
            "boundary edges",
            self.boundary_edges.iter().map(|edge| edge.edge_id),
        )?;
        let node_identities = self
            .nodes
            .iter()
            .map(|node| (node.node_id, node.stable_identity))
            .collect::<BTreeMap<_, _>>();
        validate_stable_identities(
            "mesh nodes",
            self.nodes.iter().map(|node| node.stable_identity),
        )?;

        for node in &self.nodes {
            for coordinate in node.coordinates_m {
                validate_finite("mesh node coordinate", coordinate)?;
            }
            validate_provenance("mesh node", &node.provenance)?;
            super::artifact_parameters::validate_node_exact_parameters(node)?;
        }
        let mut element_identities = BTreeSet::new();
        for element in &self.volume_elements {
            if element.order != request.element_order
                || element.node_ids.len()
                    != match element.order {
                        ElementOrder::Tet4 => 4,
                        ElementOrder::Tet10 => 10,
                    }
                || !all_references_exist(&element.node_ids, &node_ids)
                || element.node_ids.iter().collect::<BTreeSet<_>>().len() != element.node_ids.len()
            {
                return Err(MeshingContractError::invalid(
                    "volume element",
                    "order or node connectivity is inconsistent",
                ));
            }
            let corners = std::array::from_fn(|index| node_identities[&element.node_ids[index]]);
            if element.stable_identity != solver_volume_element_identity(corners)
                || !element_identities.insert(element.stable_identity)
            {
                return Err(MeshingContractError::invalid(
                    "volume element stable identity",
                    "identity must be unique and derive from the stable corner nodes",
                ));
            }
            validate_region_id(&element.region_id)?;
            validate_provenance("volume element", &element.provenance)?;
        }
        self.validate_boundaries(&node_ids, &node_identities, &element_ids, &face_ids)?;
        super::artifact_order::validate_order_topology(self, request, &node_ids)?;
        super::artifact_classification::validate_classification(self, &element_ids, &face_ids)?;
        self.validate_neighbors(&element_ids)?;
        self.validate_field_topologies(&node_ids, &element_ids, &face_ids, &edge_ids)
    }

    fn validate_boundaries(
        &self,
        node_ids: &BTreeSet<u64>,
        node_identities: &BTreeMap<u64, StableDigest>,
        element_ids: &BTreeSet<u64>,
        face_ids: &BTreeSet<u64>,
    ) -> Result<(), MeshingContractError> {
        let element_nodes = self
            .volume_elements
            .iter()
            .map(|element| {
                (
                    element.element_id,
                    element.node_ids.iter().copied().collect::<BTreeSet<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let boundary_face_nodes = self
            .boundary_faces
            .iter()
            .map(|face| (face.face_id, face.node_ids.as_slice()))
            .collect::<BTreeMap<_, _>>();
        let mut face_identities = BTreeSet::new();
        for face in &self.boundary_faces {
            if face.node_ids.len() != face.order.node_count()
                || face.node_ids.iter().collect::<BTreeSet<_>>().len() != face.node_ids.len()
                || !all_references_exist(&face.node_ids, node_ids)
                || !(1..=2).contains(&face.adjacent_volume_element_ids.len())
                || !strictly_increasing(&face.adjacent_volume_element_ids)
                || !all_references_exist(&face.adjacent_volume_element_ids, element_ids)
                || face.adjacent_volume_element_ids.iter().any(|element| {
                    !face
                        .node_ids
                        .iter()
                        .all(|node| element_nodes[element].contains(node))
                })
                || match face.role {
                    super::BoundaryFaceRole::ConformalInterface => {
                        face.adjacent_volume_element_ids.len() != 2
                    }
                    super::BoundaryFaceRole::Exterior
                    | super::BoundaryFaceRole::ContactPrimary
                    | super::BoundaryFaceRole::ContactSecondary => {
                        face.adjacent_volume_element_ids.len() != 1
                    }
                }
            {
                return Err(MeshingContractError::invalid(
                    "boundary face",
                    "connectivity or adjacency is inconsistent",
                ));
            }
            let corners = std::array::from_fn(|index| node_identities[&face.node_ids[index]]);
            if face.stable_identity != solver_boundary_face_identity(corners)
                || !face_identities.insert(face.stable_identity)
            {
                return Err(MeshingContractError::invalid(
                    "boundary face stable identity",
                    "identity must be unique and derive from the stable corner nodes",
                ));
            }
            validate_provenance("boundary face", &face.provenance)?;
        }
        let mut edge_identities = BTreeSet::new();
        for edge in &self.boundary_edges {
            if edge.node_ids.len() != edge.order.node_count()
                || edge.node_ids[0] >= edge.node_ids[1]
                || edge.node_ids.iter().collect::<BTreeSet<_>>().len() != edge.node_ids.len()
                || !all_references_exist(&edge.node_ids, node_ids)
                || edge.adjacent_boundary_face_ids.is_empty()
                || !strictly_increasing(&edge.adjacent_boundary_face_ids)
                || !all_references_exist(&edge.adjacent_boundary_face_ids, face_ids)
                || edge.adjacent_boundary_face_ids.iter().any(|face_id| {
                    !edge
                        .node_ids
                        .iter()
                        .all(|node| boundary_face_nodes[face_id].contains(node))
                })
            {
                return Err(MeshingContractError::invalid(
                    "boundary edge",
                    "connectivity or adjacency is inconsistent",
                ));
            }
            let endpoints = std::array::from_fn(|index| node_identities[&edge.node_ids[index]]);
            if edge.stable_identity != solver_boundary_edge_identity(endpoints)
                || !edge_identities.insert(edge.stable_identity)
            {
                return Err(MeshingContractError::invalid(
                    "boundary edge stable identity",
                    "identity must be unique and derive from the stable endpoint nodes",
                ));
            }
            validate_provenance("boundary edge", &edge.provenance)?;
        }
        Ok(())
    }

    fn validate_neighbors(&self, element_ids: &BTreeSet<u64>) -> Result<(), MeshingContractError> {
        if self.volume_elements.len().checked_mul(4) != Some(self.neighbors.len()) {
            return Err(MeshingContractError::invalid(
                "mesh neighbors",
                "each tetrahedron must have four ordered neighbor entries",
            ));
        }
        let mut previous = None;
        for neighbor in &self.neighbors {
            let key = (neighbor.element_id, neighbor.local_face_index);
            if neighbor.local_face_index >= 4
                || !element_ids.contains(&neighbor.element_id)
                || neighbor
                    .adjacent_element_id
                    .is_some_and(|id| id == neighbor.element_id || !element_ids.contains(&id))
                || previous.is_some_and(|prior| prior >= key)
            {
                return Err(MeshingContractError::invalid(
                    "mesh neighbors",
                    "entries or references are not canonical and valid",
                ));
            }
            previous = Some(key);
        }
        for neighbor in self
            .neighbors
            .iter()
            .filter(|neighbor| neighbor.adjacent_element_id.is_some())
        {
            let adjacent = neighbor.adjacent_element_id.expect("filtered Some");
            if !self.neighbors.iter().any(|candidate| {
                candidate.element_id == adjacent
                    && candidate.adjacent_element_id == Some(neighbor.element_id)
            }) {
                return Err(MeshingContractError::invalid(
                    "mesh neighbors",
                    "interior adjacency must be reciprocal",
                ));
            }
        }
        Ok(())
    }

    fn validate_field_topologies(
        &self,
        node_ids: &BTreeSet<u64>,
        element_ids: &BTreeSet<u64>,
        face_ids: &BTreeSet<u64>,
        edge_ids: &BTreeSet<u64>,
    ) -> Result<(), MeshingContractError> {
        let expected = BTreeMap::from([
            (FieldTopologyLocation::Node, node_ids),
            (FieldTopologyLocation::VolumeElement, element_ids),
            (FieldTopologyLocation::BoundaryFace, face_ids),
            (FieldTopologyLocation::BoundaryEdge, edge_ids),
        ]);
        if self.field_topologies.len() != expected.len() {
            return Err(MeshingContractError::invalid(
                "field topologies",
                "exactly one map is required for every mesh entity domain",
            ));
        }
        let mut locations = BTreeSet::new();
        for topology in &self.field_topologies {
            validate_token("field topology id", &topology.topology_id, 256)?;
            if !locations.insert(topology.location)
                || &topology
                    .ordered_entity_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    != *expected
                        .get(&topology.location)
                        .expect("all locations mapped")
                || !strictly_increasing(&topology.ordered_entity_ids)
            {
                return Err(MeshingContractError::invalid(
                    "field topology",
                    "mapping must be unique, complete, and canonically ordered",
                ));
            }
        }
        Ok(())
    }
}

fn validate_stable_identities(
    field: &str,
    identities: impl Iterator<Item = StableDigest>,
) -> Result<(), MeshingContractError> {
    let identities = identities.collect::<Vec<_>>();
    if identities.contains(&StableDigest::ZERO)
        || identities.iter().copied().collect::<BTreeSet<_>>().len() != identities.len()
    {
        return Err(MeshingContractError::invalid(
            field,
            "stable identities must be nonzero and unique",
        ));
    }
    Ok(())
}

/// Independently admits a solver topology against its fully resolved request.
pub fn validate_solver_mesh_topology(
    topology: &SolverMeshTopology,
    request: &MeshingRequest,
) -> Result<(), MeshingContractError> {
    request.validate()?;
    topology.validate(request)
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

fn validate_provenance(
    field: &str,
    provenance: &[PersistentEntityId],
) -> Result<(), MeshingContractError> {
    if provenance.is_empty()
        || provenance.len() > MAX_ENTITY_PROVENANCE
        || !strictly_increasing(provenance)
    {
        return Err(MeshingContractError::invalid(
            field,
            "provenance must be non-empty, bounded, unique, and canonically ordered",
        ));
    }
    for entity in provenance {
        entity.validate()?;
    }
    Ok(())
}

fn ordered_ids(
    field: &str,
    ids: impl Iterator<Item = u64>,
) -> Result<BTreeSet<u64>, MeshingContractError> {
    let values: Vec<_> = ids.collect();
    if values.contains(&0) || !strictly_increasing(&values) {
        return Err(MeshingContractError::invalid(
            field,
            "identifiers must be non-zero, unique, and canonically ordered",
        ));
    }
    Ok(values.into_iter().collect())
}

fn require_nonempty<T>(field: &str, values: &[T]) -> Result<(), MeshingContractError> {
    if values.is_empty() {
        return Err(MeshingContractError::invalid(field, "must not be empty"));
    }
    Ok(())
}

fn all_references_exist<T: Ord>(values: &[T], valid: &BTreeSet<T>) -> bool {
    values.iter().all(|value| valid.contains(value))
}

fn strictly_increasing<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}
