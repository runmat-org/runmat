//! Typed artifact storage is distinct from canonical encode/decode workspace.
use super::{
    artifact_workspace::slots, MeshingContractError, SolverMeshArtifact, SolverNodeExactParameter,
};
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_size::metric::MetricContributionScope;

impl SolverMeshArtifact {
    /// Requested bytes in this value and its owned buffer capacities. Includes
    /// unused Vec/String capacity and nested provenance; excludes allocator bookkeeping,
    /// canonical materialization, validation scratch and any other owner's copies.
    /// This is an allocation inventory, not process RSS or numerical certification.
    pub fn retained_storage_bytes(
        &self,
        maximum: u64,
        checkpoint: &mut impl FnMut() -> Result<(), MeshingContractError>,
    ) -> Result<u64, MeshingContractError> {
        let mut storage = Storage {
            seen: Some(Default::default()),
            bytes: 0,
            maximum,
            checkpoint,
        };
        storage.add(slots::<Self>(1)?)?;
        // Extensions of these owning contracts require an explicit inventory update.
        let Self {
            schema_version: _,
            canonical_digest: _,
            root_stage_manifest_digest: _,
            geometry: _,
            resolved_request: request,
            topology,
        } = self;
        storage.request(request)?;
        storage.topology(topology)?;
        Ok(storage.bytes)
    }
}

impl super::SolverMeshProjection {
    /// Same capacity inventory as the final artifact, without cloning or assembling
    /// a temporary artifact. Includes the projection's inline value and original request.
    /// A projection is built from live exact geometry, whose identity allocations it shares;
    /// it owns handles only.
    pub fn retained_storage_bytes(
        &self,
        maximum: u64,
        checkpoint: &mut impl FnMut() -> Result<(), MeshingContractError>,
    ) -> Result<u64, MeshingContractError> {
        let mut storage = Storage {
            seen: None,
            bytes: 0,
            maximum,
            checkpoint,
        };
        storage.add(slots::<Self>(1)?)?;
        storage.request(&self.resolved_request)?;
        storage.topology(&self.topology)?;
        Ok(storage.bytes)
    }
}
impl super::SolverMeshTopology {
    /// Owned buffer capacity inventory; validation/encoding scratch and allocator
    /// bookkeeping remain separate. No serialized-length proxy is used.
    pub fn retained_storage_bytes(
        &self,
        maximum: u64,
        checkpoint: &mut impl FnMut() -> Result<(), MeshingContractError>,
    ) -> Result<u64, MeshingContractError> {
        let mut storage = Storage {
            seen: Some(Default::default()),
            bytes: 0,
            maximum,
            checkpoint,
        };
        storage.add(slots::<Self>(1)?)?;
        storage.topology(self)?;
        Ok(storage.bytes)
    }
}

impl super::MeshingRequest {
    /// Capacity inventory of the original request, including its inline value.
    /// This bounds a retained clone without a serialized-size multiplier.
    pub fn retained_storage_bytes(
        &self,
        maximum: u64,
        checkpoint: &mut impl FnMut() -> Result<(), MeshingContractError>,
    ) -> Result<u64, MeshingContractError> {
        let mut storage = Storage {
            seen: Some(Default::default()),
            bytes: 0,
            maximum,
            checkpoint,
        };
        storage.add(slots::<Self>(1)?)?;
        storage.request(self)?;
        Ok(storage.bytes)
    }
}

impl super::SolverMeshNode {
    /// Actual owned buffer capacities, excluding the node's inline slot.
    pub fn owned_buffer_bytes(&self) -> Result<u64, MeshingContractError> {
        let mut storage = Storage {
            seen: None,
            bytes: 0,
            maximum: u64::MAX,
            checkpoint: &mut || Ok(()),
        };
        storage.node(self)?;
        Ok(storage.bytes)
    }
}

struct Storage<'a, F> {
    /// Distinct shared identity allocations already charged. `None` inventories one record,
    /// which owns only handles; the shared allocations belong to the geometry and artifact.
    seen: Option<std::collections::HashSet<usize>>,
    bytes: u64,
    maximum: u64,
    checkpoint: &'a mut F,
}
impl<F: FnMut() -> Result<(), MeshingContractError>> Storage<'_, F> {
    fn request(&mut self, request: &super::MeshingRequest) -> Result<(), MeshingContractError> {
        let super::MeshingRequest {
            schema_version: _,
            element_order: _,
            dimension: _,
            deterministic_seed: _,
            algorithms,
            tolerance: _,
            metric,
            quality: _,
            limits: _,
            cancellation: _,
            size_controls,
        } = request;
        for control in size_controls {
            self.add(control.control_id.len() as u64)?;
            self.add(control.region_id.len() as u64)?;
        }
        let super::AlgorithmVersionSet {
            geometry,
            curve,
            surface,
            plc,
            tetrahedron,
            optimization,
            validation,
        } = algorithms;
        for value in [
            geometry,
            curve,
            surface,
            plc,
            tetrahedron,
            optimization,
            validation,
        ] {
            self.string(value)?;
        }
        self.vector(&metric.contributions)?;
        for contribution in &metric.contributions {
            match &contribution.scope {
                MetricContributionScope::Region { region_id } => self.entity(region_id)?,
                MetricContributionScope::Entity { entity_id } => self.entity(entity_id)?,
            }
        }
        Ok(())
    }
    fn topology(
        &mut self,
        topology: &super::SolverMeshTopology,
    ) -> Result<(), MeshingContractError> {
        let super::SolverMeshTopology {
            nodes,
            volume_elements,
            neighbors,
            boundary_faces,
            boundary_edges,
            regions,
            conformal_interfaces,
            contacts,
            field_topologies,
        } = topology;
        self.vector(nodes)?;
        for node in nodes {
            self.node(node)?;
        }
        self.vector(volume_elements)?;
        for element in volume_elements {
            self.vector(&element.node_ids)?;
            self.entity(&element.region_id)?;
            self.entities(&element.provenance)?;
        }
        self.vector(neighbors)?;
        self.vector(boundary_faces)?;
        for face in boundary_faces {
            self.vector(&face.node_ids)?;
            self.vector(&face.adjacent_volume_element_ids)?;
            self.entities(&face.provenance)?;
        }
        self.vector(boundary_edges)?;
        for edge in boundary_edges {
            self.vector(&edge.node_ids)?;
            self.vector(&edge.adjacent_boundary_face_ids)?;
            self.entities(&edge.provenance)?;
        }
        self.vector(regions)?;
        for region in regions {
            self.entity(&region.region_id)?;
            self.vector(&region.element_ids)?;
        }
        self.vector(conformal_interfaces)?;
        for interface in conformal_interfaces {
            self.entity(&interface.source_face_id)?;
            self.entity(&interface.side_a_region_id)?;
            self.entity(&interface.side_b_region_id)?;
            self.vector(&interface.boundary_face_ids)?;
        }
        self.vector(contacts)?;
        for contact in contacts {
            self.entity(&contact.contact_id)?;
            self.vector(&contact.primary_boundary_face_ids)?;
            self.vector(&contact.secondary_boundary_face_ids)?;
        }
        self.vector(field_topologies)?;
        for field in field_topologies {
            self.string(&field.topology_id)?;
            self.vector(&field.ordered_entity_ids)?;
        }
        Ok(())
    }
    fn node(&mut self, node: &super::SolverMeshNode) -> Result<(), MeshingContractError> {
        self.entities(&node.provenance)?;
        self.vector(&node.exact_parameters)?;
        for parameter in &node.exact_parameters {
            match parameter {
                SolverNodeExactParameter::Curve { source_edge_id, .. } => {
                    self.entity(source_edge_id)?
                }
                SolverNodeExactParameter::Surface { source_face_id, .. } => {
                    self.entity(source_face_id)?
                }
            }
        }
        Ok(())
    }
    fn add(&mut self, bytes: u64) -> Result<(), MeshingContractError> {
        (self.checkpoint)()?;
        let next = self
            .bytes
            .checked_add(bytes)
            .filter(|bytes| *bytes <= self.maximum)
            .ok_or_else(|| {
                MeshingContractError::invalid(
                    "mesh retained storage",
                    "owned buffer byte inventory exceeds its ceiling or address space",
                )
            })?;
        self.bytes = next;
        Ok(())
    }
    fn vector<T>(&mut self, values: &Vec<T>) -> Result<(), MeshingContractError> {
        self.add(slots::<T>(values.capacity())?)
    }
    fn string(&mut self, value: &String) -> Result<(), MeshingContractError> {
        self.add(slots::<u8>(value.capacity())?)
    }
    /// Records hold a pointer (inside their slot); each distinct shared identity
    /// allocation is charged once per inventory.
    fn entity(&mut self, value: &PersistentEntityId) -> Result<(), MeshingContractError> {
        let Some(seen) = self.seen.as_mut() else {
            return Ok(());
        };
        if seen.insert(value.shared_address()) {
            self.add(value.shared_heap_bytes())?;
        }
        Ok(())
    }
    fn entities(&mut self, values: &Vec<PersistentEntityId>) -> Result<(), MeshingContractError> {
        self.vector(values)?;
        for value in values {
            self.entity(value)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

impl super::MeshingPartitionDescriptor {
    /// Original capacity inventory bounds the two owned endpoint identity clones.
    pub fn owned_buffer_bytes(&self) -> Result<u64, MeshingContractError> {
        let mut storage = Storage {
            seen: None,
            bytes: 0,
            maximum: u64::MAX,
            checkpoint: &mut || Ok(()),
        };
        if let Some(range) = &self.entity_range {
            storage.entity(&range.first)?;
            storage.entity(&range.last)?;
        }
        Ok(storage.bytes)
    }
}
