use super::{
    artifact_workspace::{reserve, unique},
    solver_boundary_edge_identity, solver_boundary_face_identity, solver_volume_element_identity,
    validate_finite, validate_token, CanonicalMeshingContract, ElementOrder, FieldTopologyLocation,
    MeshDimension, MeshingCancellationSignal, MeshingContractError, MeshingRequest, NeverCancelled,
    PersistentEntityId, PersistentEntityKind, SolverMeshArtifact, SolverMeshTopology, StableDigest,
    SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
};

const MAX_ENTITY_PROVENANCE: usize = 32;

impl SolverMeshArtifact {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        self.validate_canonical()
    }

    /// Solver elements of the requested dimension: tetrahedra, sheet triangles, or segments.
    pub fn generated_element_count(&self) -> u64 {
        (match self.resolved_request.dimension {
            MeshDimension::Solid => self.topology.volume_elements.len(),
            MeshDimension::Surface => self.topology.boundary_faces.len(),
            MeshDimension::Curve => self.topology.boundary_edges.len(),
        }) as u64
    }

    pub(super) fn validate_payload_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), MeshingContractError> {
        self.validate_header()?;
        self.topology.validate(&self.resolved_request, cancellation)
    }

    /// Schema, stage-manifest, geometry and request checks; the topology is checked separately.
    pub(super) fn validate_header(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != SOLVER_MESH_ARTIFACT_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "analysis mesh artifact schema version",
                format!("expected {SOLVER_MESH_ARTIFACT_SCHEMA_VERSION}"),
            ));
        }
        self.root_stage_manifest_digest
            .validate_nonzero("artifact.root_stage_manifest_digest")?;
        self.geometry.validate()?;
        self.resolved_request.validate()
    }
}

impl SolverMeshTopology {
    pub(super) fn validate(
        &self,
        request: &MeshingRequest,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), MeshingContractError> {
        require_nonempty("mesh nodes", &self.nodes)?;
        match request.dimension {
            MeshDimension::Solid => require_nonempty("volume elements", &self.volume_elements)?,
            MeshDimension::Surface | MeshDimension::Curve => {
                // Shell and beam discretizations carry no volume cells or cell classification.
                if !self.volume_elements.is_empty()
                    || !self.neighbors.is_empty()
                    || !self.regions.is_empty()
                    || !self.conformal_interfaces.is_empty()
                    || !self.contacts.is_empty()
                {
                    return Err(MeshingContractError::invalid(
                        "mesh topology",
                        "surface and curve meshes contain no volume elements or cell classification",
                    ));
                }
                let missing = match request.dimension {
                    MeshDimension::Surface => self.boundary_faces.is_empty(),
                    _ => !self.boundary_faces.is_empty() || self.boundary_edges.is_empty(),
                };
                if missing {
                    return Err(MeshingContractError::invalid(
                        "mesh topology",
                        "surface meshes require triangles; curve meshes require only segments",
                    ));
                }
            }
        }
        if self.nodes.len() as u64 > request.limits.maximum_nodes
            || self.volume_elements.len() as u64 > request.limits.maximum_elements
        {
            return Err(MeshingContractError::invalid(
                "mesh topology",
                "node or element count exceeds the resolved hard budget",
            ));
        }

        super::artifact_resources::validation_scratch_bytes(self, request.element_order)?;
        let node_ids = ordered_ids(
            "mesh nodes",
            self.nodes.iter().map(|node| node.node_id),
            self.nodes.len(),
            cancellation,
        )?;
        validation_checkpoint(cancellation)?;
        let element_ids = ordered_ids(
            "volume elements",
            self.volume_elements
                .iter()
                .map(|element| element.element_id),
            self.volume_elements.len(),
            cancellation,
        )?;
        let face_ids = ordered_ids(
            "boundary faces",
            self.boundary_faces.iter().map(|face| face.face_id),
            self.boundary_faces.len(),
            cancellation,
        )?;
        let edge_ids = ordered_ids(
            "boundary edges",
            self.boundary_edges.iter().map(|edge| edge.edge_id),
            self.boundary_edges.len(),
            cancellation,
        )?;
        validation_checkpoint(cancellation)?;
        let mut node_identities = reserve(self.nodes.len(), cancellation)?;
        for node in &self.nodes {
            validation_checkpoint(cancellation)?;
            node_identities.push(node.stable_identity);
        }
        validate_stable_identities("mesh nodes", &mut node_identities, cancellation)?;
        drop(node_identities);

        for node in &self.nodes {
            validation_checkpoint(cancellation)?;
            for coordinate in node.coordinates_m.canonical_value() {
                validate_finite("mesh node coordinate", coordinate)?;
            }
            validate_provenance("mesh node", &node.provenance)?;
            super::artifact_parameters::validate_node_exact_parameters(node)?;
        }
        let mut element_identities = reserve(self.volume_elements.len(), cancellation)?;
        for element in &self.volume_elements {
            validation_checkpoint(cancellation)?;
            if element.order != request.element_order
                || element.node_ids.len()
                    != match element.order {
                        ElementOrder::Tet4 => 4,
                        ElementOrder::Tet10 => 10,
                    }
                || !all_references_exist(&element.node_ids, &node_ids)
                || !unique(&element.node_ids)
            {
                return Err(MeshingContractError::invalid(
                    "volume element",
                    "order or node connectivity is inconsistent",
                ));
            }
            let corners = std::array::from_fn(|index| self.node_identity(element.node_ids[index]));
            if element.stable_identity != solver_volume_element_identity(corners) {
                return Err(MeshingContractError::invalid(
                    "volume element stable identity",
                    "identity must be unique and derive from the stable corner nodes",
                ));
            }
            element_identities.push(element.stable_identity);
            validate_region_id(&element.region_id)?;
            validate_provenance("volume element", &element.provenance)?;
        }
        validate_stable_identities(
            "volume element stable identity",
            &mut element_identities,
            cancellation,
        )?;
        drop(element_identities);
        self.validate_boundaries(
            request.dimension,
            &node_ids,
            &element_ids,
            &face_ids,
            cancellation,
        )?;
        super::artifact_order::validate_order_topology(self, request, &node_ids, cancellation)?;
        super::artifact_classification::validate_classification(
            self,
            &element_ids,
            &face_ids,
            cancellation,
        )?;
        self.validate_neighbors(&element_ids, cancellation)?;
        self.validate_field_topologies(&node_ids, &element_ids, &face_ids, &edge_ids, cancellation)
    }

    fn validate_boundaries(
        &self,
        dimension: MeshDimension,
        node_ids: &[u64],
        element_ids: &[u64],
        face_ids: &[u64],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), MeshingContractError> {
        let mut face_identities = reserve(self.boundary_faces.len(), cancellation)?;
        for face in &self.boundary_faces {
            validation_checkpoint(cancellation)?;
            if face.node_ids.len() != face.order.node_count()
                || !unique(&face.node_ids)
                || !all_references_exist(&face.node_ids, node_ids)
                || match dimension {
                    MeshDimension::Solid => {
                        !(1..=2).contains(&face.adjacent_volume_element_ids.len())
                    }
                    // A sheet triangle is itself the shell element; it bounds no volume.
                    MeshDimension::Surface | MeshDimension::Curve => {
                        !face.adjacent_volume_element_ids.is_empty()
                            || face.role != super::BoundaryFaceRole::Exterior
                    }
                }
                || !strictly_increasing(&face.adjacent_volume_element_ids)
                || !all_references_exist(&face.adjacent_volume_element_ids, element_ids)
                || face.adjacent_volume_element_ids.iter().any(|element| {
                    !face.node_ids.iter().all(|node| {
                        self.volume_elements[element_ids
                            .binary_search(element)
                            .expect("validated element")]
                        .node_ids
                        .contains(node)
                    })
                })
                || (dimension.is_solid()
                    && match face.role {
                        super::BoundaryFaceRole::ConformalInterface => {
                            face.adjacent_volume_element_ids.len() != 2
                        }
                        super::BoundaryFaceRole::Exterior
                        | super::BoundaryFaceRole::ContactPrimary
                        | super::BoundaryFaceRole::ContactSecondary => {
                            face.adjacent_volume_element_ids.len() != 1
                        }
                    })
            {
                return Err(MeshingContractError::invalid(
                    "boundary face",
                    "connectivity or adjacency is inconsistent",
                ));
            }
            let corners = std::array::from_fn(|index| self.node_identity(face.node_ids[index]));
            if face.stable_identity != solver_boundary_face_identity(corners) {
                return Err(MeshingContractError::invalid(
                    "boundary face stable identity",
                    "identity must be unique and derive from the stable corner nodes",
                ));
            }
            face_identities.push(face.stable_identity);
            validate_provenance("boundary face", &face.provenance)?;
        }
        validate_stable_identities(
            "boundary face stable identity",
            &mut face_identities,
            cancellation,
        )?;
        drop(face_identities);
        let mut edge_identities = reserve(self.boundary_edges.len(), cancellation)?;
        for edge in &self.boundary_edges {
            validation_checkpoint(cancellation)?;
            if edge.node_ids.len() != edge.order.node_count()
                || edge.node_ids[0] >= edge.node_ids[1]
                || !unique(&edge.node_ids)
                || !all_references_exist(&edge.node_ids, node_ids)
                || edge.adjacent_boundary_face_ids.is_empty() != (dimension == MeshDimension::Curve)
                || !strictly_increasing(&edge.adjacent_boundary_face_ids)
                || !all_references_exist(&edge.adjacent_boundary_face_ids, face_ids)
                || edge.adjacent_boundary_face_ids.iter().any(|face_id| {
                    !edge.node_ids.iter().all(|node| {
                        self.boundary_faces
                            [face_ids.binary_search(face_id).expect("validated face")]
                        .node_ids
                        .contains(node)
                    })
                })
            {
                return Err(MeshingContractError::invalid(
                    "boundary edge",
                    "connectivity or adjacency is inconsistent",
                ));
            }
            let endpoints = std::array::from_fn(|index| self.node_identity(edge.node_ids[index]));
            if edge.stable_identity != solver_boundary_edge_identity(endpoints) {
                return Err(MeshingContractError::invalid(
                    "boundary edge stable identity",
                    "identity must be unique and derive from the stable endpoint nodes",
                ));
            }
            edge_identities.push(edge.stable_identity);
            validate_provenance("boundary edge", &edge.provenance)?;
        }
        validate_stable_identities(
            "boundary edge stable identity",
            &mut edge_identities,
            cancellation,
        )
    }

    fn node_identity(&self, id: u64) -> StableDigest {
        self.nodes[self
            .nodes
            .binary_search_by_key(&id, |node| node.node_id)
            .expect("validated node")]
        .stable_identity
    }

    fn validate_neighbors(
        &self,
        element_ids: &[u64],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), MeshingContractError> {
        if self.volume_elements.len().checked_mul(4) != Some(self.neighbors.len()) {
            return Err(MeshingContractError::invalid(
                "mesh neighbors",
                "each tetrahedron must have four ordered neighbor entries",
            ));
        }
        let mut previous = None;
        let mut directed_adjacencies = reserve(self.neighbors.len(), cancellation)?;
        for neighbor in &self.neighbors {
            validation_checkpoint(cancellation)?;
            let key = (neighbor.element_id, neighbor.local_face_index);
            if neighbor.local_face_index >= 4
                || element_ids.binary_search(&neighbor.element_id).is_err()
                || neighbor.adjacent_element_id.is_some_and(|id| {
                    id == neighbor.element_id || element_ids.binary_search(&id).is_err()
                })
                || previous.is_some_and(|prior| prior >= key)
            {
                return Err(MeshingContractError::invalid(
                    "mesh neighbors",
                    "entries or references are not canonical and valid",
                ));
            }
            if let Some(adjacent_element_id) = neighbor.adjacent_element_id {
                directed_adjacencies.push((neighbor.element_id, adjacent_element_id));
            }
            previous = Some(key);
        }
        validation_checkpoint(cancellation)?;
        directed_adjacencies.sort_unstable();
        for &(element_id, adjacent_element_id) in &directed_adjacencies {
            validation_checkpoint(cancellation)?;
            if directed_adjacencies
                .binary_search(&(adjacent_element_id, element_id))
                .is_err()
            {
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
        node_ids: &[u64],
        element_ids: &[u64],
        face_ids: &[u64],
        edge_ids: &[u64],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), MeshingContractError> {
        let expected = [
            (FieldTopologyLocation::Node, node_ids),
            (FieldTopologyLocation::VolumeElement, element_ids),
            (FieldTopologyLocation::BoundaryFace, face_ids),
            (FieldTopologyLocation::BoundaryEdge, edge_ids),
        ];
        if self.field_topologies.len() != expected.len() {
            return Err(MeshingContractError::invalid(
                "field topologies",
                "exactly one map is required for every mesh entity domain",
            ));
        }
        let mut locations = [false; 4];
        for topology in &self.field_topologies {
            validation_checkpoint(cancellation)?;
            validate_token("field topology id", &topology.topology_id, 256)?;
            let index = expected
                .iter()
                .position(|(location, _)| *location == topology.location)
                .expect("all locations mapped");
            if std::mem::replace(&mut locations[index], true)
                || topology.ordered_entity_ids.as_slice() != expected[index].1
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
    identities: &mut [StableDigest],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    validation_checkpoint(cancellation)?;
    identities.sort_unstable();
    validation_checkpoint(cancellation)?;
    if identities.first() == Some(&StableDigest::ZERO)
        || identities.windows(2).any(|pair| pair[0] == pair[1])
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
    validate_solver_mesh_topology_with_cancellation(topology, request, &NeverCancelled)
}

/// Independently admits a solver topology while honoring an execution-owned cancellation signal.
pub fn validate_solver_mesh_topology_with_cancellation(
    topology: &SolverMeshTopology,
    request: &MeshingRequest,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    request.validate()?;
    topology.validate(request, cancellation)
}

pub(super) fn validation_checkpoint(
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    if cancellation.is_cancelled() {
        return Err(MeshingContractError::invalid(
            "solver mesh validation",
            "validation was cancelled",
        ));
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
    count: usize,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<u64>, MeshingContractError> {
    let mut values = reserve(count, cancellation)?;
    for id in ids {
        validation_checkpoint(cancellation)?;
        if values.len() == count || id == 0 || values.last().is_some_and(|prior| *prior >= id) {
            return Err(MeshingContractError::invalid(
                field,
                "identifiers must be non-zero, unique, and canonically ordered",
            ));
        }
        values.push(id);
    }
    Ok(values)
}

fn require_nonempty<T>(field: &str, values: &[T]) -> Result<(), MeshingContractError> {
    if values.is_empty() {
        return Err(MeshingContractError::invalid(field, "must not be empty"));
    }
    Ok(())
}

fn all_references_exist<T: Ord>(values: &[T], valid: &[T]) -> bool {
    values
        .iter()
        .all(|value| valid.binary_search(value).is_ok())
}

fn strictly_increasing<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

#[cfg(test)]
mod tests;
