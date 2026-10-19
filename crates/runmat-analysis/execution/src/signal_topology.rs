//! Exact, physics-neutral row binding for post-mesh immutable signal materialization.
use runmat_analysis_core::{FieldLocation, PhysicalSignal, SignalShape};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

mod resolution;
pub use resolution::SignalTopologyResolution;

/// Canonical mesh-domain rows, not a physics recovery/quadrature convention.
/// Transported bindings must pass `validate_for` against the authoritative mesh.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SignalTopologyBindingWire")]
pub struct SignalTopologyBinding {
    mesh_digest: Digest,
    location: FieldLocation,
    entity_count: usize,
    component_count: usize,
    order_digest: Digest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SignalTopologyBindingWire {
    mesh_digest: Digest,
    location: FieldLocation,
    entity_count: usize,
    component_count: usize,
    order_digest: Digest,
}

impl TryFrom<SignalTopologyBindingWire> for SignalTopologyBinding {
    type Error = AnalysisBridgeError;

    fn try_from(value: SignalTopologyBindingWire) -> AnalysisBridgeResult<Self> {
        let result = Self {
            mesh_digest: value.mesh_digest,
            location: value.location,
            entity_count: value.entity_count,
            component_count: value.component_count,
            order_digest: value.order_digest,
        };
        result.validate_dimensions()?;
        Ok(result)
    }
}

impl SignalTopologyBinding {
    /// Ordinary scalar/vector/tensor shapes have no topology binding.
    /// An authored topology identifier must already name this exact mesh digest.
    pub fn derive(
        signal: &PhysicalSignal,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisBridgeResult<Option<Self>> {
        if matches!(signal.shape(), SignalShape::DeferredTopologyField { .. }) {
            return Err(invalid(
                "deferred topology selector requires an explicit mesh resolution transition",
            ));
        }
        let SignalShape::TopologyField {
            topology_id,
            location,
            component_count,
        } = signal.shape()
        else {
            return Ok(None);
        };
        mesh.validate_canonical()
            .map_err(|error| invalid(error.to_string()))?;
        let mesh_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
        if topology_id.parse::<Digest>()? != mesh_digest {
            return Err(AnalysisBridgeError::Identity(
                "signal topology names another canonical mesh",
            ));
        }
        let entity_count = match location {
            FieldLocation::Node => mesh.topology.nodes.len(),
            FieldLocation::Cell | FieldLocation::Element => mesh.topology.volume_elements.len(),
            FieldLocation::Face => mesh.topology.boundary_faces.len(),
            FieldLocation::Edge => mesh.topology.boundary_edges.len(),
            _ => {
                return Err(invalid(
                    "field location has no authoritative canonical mesh row inventory",
                ))
            }
        };
        let mut result = Self {
            mesh_digest,
            location: *location,
            entity_count,
            component_count: *component_count,
            order_digest: Digest::sha256([]),
        };
        result.validate_dimensions()?;
        // These domain inventories are canonically ordered by mesh validation and
        // match AnalysisProjection's node/element/face order. No sorting or reindexing.
        result.order_digest = match location {
            FieldLocation::Node => result.row_digest(
                mesh.topology
                    .nodes
                    .iter()
                    .map(|row| (row.node_id, row.stable_identity)),
            )?,
            FieldLocation::Cell | FieldLocation::Element => result.row_digest(
                mesh.topology
                    .volume_elements
                    .iter()
                    .map(|row| (row.element_id, row.stable_identity)),
            )?,
            FieldLocation::Face => result.row_digest(
                mesh.topology
                    .boundary_faces
                    .iter()
                    .map(|row| (row.face_id, row.stable_identity)),
            )?,
            FieldLocation::Edge => result.row_digest(
                mesh.topology
                    .boundary_edges
                    .iter()
                    .map(|row| (row.edge_id, row.stable_identity)),
            )?,
            _ => unreachable!("location admitted above"),
        };
        Ok(Some(result))
    }

    /// Checks source/shape association without claiming mesh-order authentication.
    pub fn validate_shape(&self, signal: &PhysicalSignal) -> AnalysisBridgeResult<()> {
        self.validate_dimensions()?;
        let SignalShape::TopologyField {
            topology_id,
            location,
            component_count,
        } = signal.shape()
        else {
            return Err(invalid("non-topology signal carries a topology binding"));
        };
        if topology_id.parse::<Digest>()? != self.mesh_digest
            || *location != self.location
            || *component_count != self.component_count
        {
            return Err(AnalysisBridgeError::Identity(
                "signal shape differs from its topology binding",
            ));
        }
        Ok(())
    }

    /// Re-derives exact count and row identity; transported counts/order are not authority.
    pub fn validate_for(
        &self,
        signal: &PhysicalSignal,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisBridgeResult<()> {
        self.validate_shape(signal)?;
        if Self::derive(signal, mesh)?.as_ref() != Some(self) {
            return Err(AnalysisBridgeError::Identity(
                "signal binding differs from authoritative mesh rows",
            ));
        }
        Ok(())
    }

    pub fn mesh_digest(&self) -> Digest {
        self.mesh_digest
    }
    pub fn location(&self) -> FieldLocation {
        self.location
    }
    pub fn entity_count(&self) -> usize {
        self.entity_count
    }
    pub fn component_count(&self) -> usize {
        self.component_count
    }
    pub fn order_digest(&self) -> Digest {
        self.order_digest
    }

    pub fn scalar_component_count(&self) -> AnalysisBridgeResult<usize> {
        self.entity_count
            .checked_mul(self.component_count)
            .ok_or_else(|| invalid("topology signal scalar cardinality overflows this host"))
    }

    fn validate_dimensions(&self) -> AnalysisBridgeResult<()> {
        if self.entity_count == 0 || self.component_count == 0 {
            return Err(invalid(
                "topology signal requires positive row/component counts",
            ));
        }
        location_tag(self.location)?;
        self.scalar_component_count()?;
        Ok(())
    }

    /// Ordered fixed-width digest segments avoid a copied row inventory/encoded tree.
    fn row_digest(
        &self,
        rows: impl Iterator<Item = (u64, StableDigest)>,
    ) -> AnalysisBridgeResult<Digest> {
        let count = u64::try_from(self.entity_count)
            .map_err(|_| invalid("entity count exceeds u64"))?
            .to_le_bytes();
        let width = u64::try_from(self.component_count)
            .map_err(|_| invalid("component count exceeds u64"))?
            .to_le_bytes();
        let location = [location_tag(self.location)?];
        let header = Digest::sha256_parts([
            b"runmat-signal-topology-order/v1\0".as_slice(),
            self.mesh_digest.bytes().as_slice(),
            &location,
            &count,
            &width,
        ]);
        let rows = rows.map(|(id, identity)| {
            *Digest::sha256_parts([id.to_le_bytes().as_slice(), identity.bytes().as_slice()])
                .bytes()
        });
        Ok(Digest::sha256_parts(
            std::iter::once(*header.bytes()).chain(rows),
        ))
    }
}

fn location_tag(location: FieldLocation) -> AnalysisBridgeResult<u8> {
    match location {
        FieldLocation::Node => Ok(1),
        FieldLocation::Cell => Ok(2),
        FieldLocation::Element => Ok(3),
        FieldLocation::Face => Ok(4),
        FieldLocation::Edge => Ok(5),
        _ => Err(invalid(
            "field location has no authoritative canonical mesh row inventory",
        )),
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}

#[cfg(test)]
mod tests;
