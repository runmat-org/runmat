//! Count-only construction inventory, composed with the mesh/codec owners' estimates.
//! Requested slots exclude borrowed inputs, allocator bookkeeping and measured RSS.
use super::*;
use runmat_canonical_codec::profile_serialized_shape;
use serde::{ser::SerializeSeq, Serializer};

#[derive(Debug)]
pub(crate) struct NodeSpaceWorkspaceProfile {
    pub node_slots: usize,
    pub requested_vector_bytes: u64,
    pub identity_materialization_bytes: u64,
    pub validation_bytes: u64,
}
struct Repeated<'a, T> {
    value: &'a T,
    count: usize,
}
impl<T: Serialize> Serialize for Repeated<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.count))?;
        for ordinal in 0..self.count {
            if ordinal % 1024 == 0 {
                storage::check_cancelled().map_err(serde::ser::Error::custom)?;
            }
            seq.serialize_element(self.value)?;
        }
        seq.end()
    }
}
pub(crate) fn node_space_workspace_profile(
    mesh: Option<&SolverMeshArtifact>,
    structural: Option<&StructuralModel>,
) -> Result<NodeSpaceWorkspaceProfile, StructuralNodeSpaceError> {
    storage::check_cancelled()?;
    let solid = mesh.map_or(0, |m| m.topology.nodes.len());
    let explicit = structural.map_or(0, |s| s.nodes.len());
    let appended = structural.map_or(0, |s| {
        s.nodes
            .iter()
            .filter(|n| mesh.is_none() || n.solid_attachment.is_none())
            .count()
    });
    let node_slots = storage::count_add(solid, appended)?;
    let mut requested_vector_bytes = storage::add(
        storage::slots::<CanonicalStructuralNode>(node_slots)?,
        storage::slots::<usize>(explicit)?,
    )?;
    requested_vector_bytes = storage::add(
        requested_vector_bytes,
        storage::slots::<StableDigest>(node_slots)?,
    )?;
    if mesh.is_some() && structural.is_some() {
        requested_vector_bytes = storage::add(
            requested_vector_bytes,
            storage::slots::<Option<u32>>(solid)?,
        )?;
    }
    let validation_bytes = mesh
        .map(|m| {
            m.validation_resident_bytes_estimate()
                .map_err(|e| StructuralNodeSpaceError::InvalidMesh(e.to_string()))
        })
        .transpose()?
        .unwrap_or(0);
    let mut identity_materialization_bytes = structural
        .map(super::super::explicit_identity_resource_bytes)
        .transpose()
        .map_err(|e| match e {
            super::super::identity::ExplicitStructuralTopologyError::Storage(error) => error.into(),
            other => StructuralNodeSpaceError::Identity(other.to_string()),
        })?
        .unwrap_or(0);
    if mesh.is_some() && structural.is_some() {
        // Only the inventory is serialized here. Placeholder values never enter identity
        // or physics; the shared profiler bounds every scalar by its maximal wire width.
        let node = CanonicalStructuralNode {
            identity: StableDigest::ZERO,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([0.0; 3]),
            owns_solid_translation: true,
            owns_explicit_rotation: true,
        };
        let shape = (
            StableDigest::ZERO,
            Digest::from_bytes([0; 32]),
            Repeated {
                value: &node,
                count: node_slots,
            },
            Repeated {
                value: &0usize,
                count: explicit,
            },
        );
        let limits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);
        identity_materialization_bytes = identity_materialization_bytes.max(
            profile_serialized_shape(&shape, limits)
                .map_err(|e| StructuralNodeSpaceError::Identity(e.to_string()))?
                .materialization_resident_bytes_estimate,
        );
    }
    let profile = NodeSpaceWorkspaceProfile {
        node_slots,
        requested_vector_bytes,
        identity_materialization_bytes,
        validation_bytes,
    };
    // Check the composed requested envelope before constructing any owned node-space
    // buffer. This is addressability, not policy admission or measured RSS.
    usize::try_from(profile.total_requested_bytes()?)
        .map_err(|_| TopologyStorageError::Capacity)?;
    Ok(profile)
}
impl NodeSpaceWorkspaceProfile {
    pub(crate) fn total_requested_bytes(&self) -> Result<u64, TopologyStorageError> {
        storage::add(
            self.requested_vector_bytes,
            self.validation_bytes
                .max(self.identity_materialization_bytes),
        )
    }
}
