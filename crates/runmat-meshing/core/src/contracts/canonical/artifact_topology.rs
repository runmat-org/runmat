//! Shared, copy-on-write solver topology that carries its artifact's admitted identity.
//!
//! A solver mesh is validated and digested once, when it is sealed or decoded. The result
//! (the identity digest of the topology under one schema, geometry and resolved request) is
//! kept with the topology itself, so every later `validate_canonical*` on the same content is
//! an equality check instead of a full topology walk, re-encode and SHA-256.
//!
//! Soundness: the topology is reachable for writing only through `DerefMut`, which detaches
//! the shared allocation and drops the admitted identity before handing out the reference.
//! The identity also records the header it was derived under, so a changed schema, geometry
//! or request field misses it and takes the full validation path. Clones share the
//! allocation and the identity, which also makes artifact clones O(1).

use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync::{Arc, OnceLock},
};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{GeometryRevisionRef, MeshingRequest, SolverMeshTopology, StableDigest};

/// Identity of a validated payload: the header it was validated under and its digest.
#[derive(Debug)]
pub(super) struct AdmittedIdentity {
    schema_version: u16,
    geometry: GeometryRevisionRef,
    resolved_request: MeshingRequest,
    digest: StableDigest,
    /// Canonical encoded size of the sealed artifact, derived on first request.
    encoded_len: OnceLock<u64>,
}

impl AdmittedIdentity {
    pub(super) fn digest(&self) -> StableDigest {
        self.digest
    }

    pub(super) fn encoded_len(&self) -> &OnceLock<u64> {
        &self.encoded_len
    }
}

struct TopologyCell {
    topology: SolverMeshTopology,
    identity: OnceLock<AdmittedIdentity>,
}

impl Clone for TopologyCell {
    /// A detached copy is about to be written; it never inherits the admitted identity.
    fn clone(&self) -> Self {
        Self {
            topology: self.topology.clone(),
            identity: OnceLock::new(),
        }
    }
}

/// Solver topology held by a `SolverMeshArtifact`.
#[derive(Clone)]
pub struct SharedSolverMeshTopology {
    cell: Arc<TopologyCell>,
}

impl SharedSolverMeshTopology {
    pub fn new(topology: SolverMeshTopology) -> Self {
        Self {
            cell: Arc::new(TopologyCell {
                topology,
                identity: OnceLock::new(),
            }),
        }
    }

    /// The owned topology; copies only when another artifact still shares it.
    pub fn into_topology(self) -> SolverMeshTopology {
        match Arc::try_unwrap(self.cell) {
            Ok(cell) => cell.topology,
            Err(shared) => shared.topology.clone(),
        }
    }

    /// The admitted identity when it was derived under exactly this header.
    pub(super) fn admitted(
        &self,
        schema_version: u16,
        geometry: &GeometryRevisionRef,
        resolved_request: &MeshingRequest,
    ) -> Option<&AdmittedIdentity> {
        self.cell.identity.get().filter(|identity| {
            identity.schema_version == schema_version
                && identity.geometry == *geometry
                && identity.resolved_request == *resolved_request
        })
    }

    /// Records the identity of a payload that has just passed full validation.
    pub(super) fn admit(
        &self,
        schema_version: u16,
        geometry: &GeometryRevisionRef,
        resolved_request: &MeshingRequest,
        digest: StableDigest,
    ) {
        let _ = self.cell.identity.set(AdmittedIdentity {
            schema_version,
            geometry: geometry.clone(),
            resolved_request: resolved_request.clone(),
            digest,
            encoded_len: OnceLock::new(),
        });
    }

    /// True when both handles share one allocation (and therefore one identity).
    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cell, &other.cell)
    }
}

impl From<SolverMeshTopology> for SharedSolverMeshTopology {
    fn from(topology: SolverMeshTopology) -> Self {
        Self::new(topology)
    }
}

impl Deref for SharedSolverMeshTopology {
    type Target = SolverMeshTopology;

    fn deref(&self) -> &SolverMeshTopology {
        &self.cell.topology
    }
}

impl DerefMut for SharedSolverMeshTopology {
    /// Detaches a shared topology and drops its admitted identity before any write.
    fn deref_mut(&mut self) -> &mut SolverMeshTopology {
        let cell = Arc::make_mut(&mut self.cell);
        cell.identity.take();
        &mut cell.topology
    }
}

impl PartialEq for SharedSolverMeshTopology {
    fn eq(&self, other: &Self) -> bool {
        self.shares_allocation_with(other) || self.cell.topology == other.cell.topology
    }
}

impl PartialEq<SolverMeshTopology> for SharedSolverMeshTopology {
    fn eq(&self, other: &SolverMeshTopology) -> bool {
        self.cell.topology == *other
    }
}

impl fmt::Debug for SharedSolverMeshTopology {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cell.topology.fmt(formatter)
    }
}

impl Serialize for SharedSolverMeshTopology {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.cell.topology.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SharedSolverMeshTopology {
    /// Decoded content is never trusted: it starts without an admitted identity.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        SolverMeshTopology::deserialize(deserializer).map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use crate::fixtures::canonical_tetrahedron_solver_mesh;
    use crate::{CanonicalMeshingContract, ElementOrder, StableDigest};

    #[test]
    fn sealed_artifacts_carry_their_identity_and_writes_drop_it() {
        let artifact = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let identity = |artifact: &crate::SolverMeshArtifact| {
            artifact
                .topology
                .admitted(
                    artifact.schema_version,
                    &artifact.geometry,
                    &artifact.resolved_request,
                )
                .map(|identity| identity.digest())
        };
        // Sealing admits the identity; clones share the allocation and the identity.
        assert_eq!(identity(&artifact), Some(artifact.canonical_digest));
        let clone = artifact.clone();
        assert!(clone.topology.shares_allocation_with(&artifact.topology));
        assert_eq!(identity(&clone), Some(artifact.canonical_digest));
        clone.validate().unwrap();

        // A claimed digest is still checked against the admitted identity.
        let mut forged = artifact.clone();
        forged.canonical_digest = StableDigest::from_bytes([7; 32]);
        assert!(forged.validate().is_err());

        // A header change misses the identity and takes full validation.
        let mut renamed = artifact.clone();
        renamed.resolved_request.limits.maximum_nodes += 1;
        assert_eq!(identity(&renamed), None);
        assert!(renamed.validate().is_err());

        // Any topology write detaches the allocation and drops the identity first.
        let mut written = artifact.clone();
        written.topology.nodes[0].coordinates_m = written.topology.nodes[1].coordinates_m;
        assert!(!written.topology.shares_allocation_with(&artifact.topology));
        assert_eq!(identity(&written), None);
        assert!(written.validate().is_err());
        assert_eq!(identity(&artifact), Some(artifact.canonical_digest));

        // Decoded content is admitted by its own full validation, never trusted.
        let decoded =
            crate::SolverMeshArtifact::canonical_decode(&artifact.canonical_encode().unwrap())
                .unwrap();
        assert!(!decoded.topology.shares_allocation_with(&artifact.topology));
        assert_eq!(identity(&decoded), Some(artifact.canonical_digest));
        assert_eq!(
            decoded.canonical_encoded_len().unwrap(),
            artifact.canonical_encode().unwrap().len() as u64
        );
    }
}
