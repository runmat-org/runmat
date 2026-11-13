//! Material-independent exact sparse complex support over the free (ungrounded) edges.
//! Canonical geometry projection only; no physical matrices/form/source integration.
use super::{geometry, invalid, topology, MaxwellError};
use crate::{algebra::SparsePattern, topology::AnalysisProjection};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const PREFIX: &[u8] = b"runmat-maxwell-system-plan/v2\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::platform_content(256, 16));
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellSystemPlan {
    mesh_digest: StableDigest,
    projection_digest: Digest,
    edge_node_identities: Vec<[StableDigest; 2]>,
    grounded_faces: Vec<StableDigest>,
    free_edge_indices: Vec<usize>,
    /// Complex operator pattern over free edges (one entry per complex coefficient).
    pattern: SparsePattern,
}
impl MaxwellSystemPlan {
    pub fn derive(
        mesh: &SolverMeshArtifact,
        grounded_faces: &[StableDigest],
    ) -> Result<Self, MaxwellError> {
        let projection = AnalysisProjection::derive(mesh).map_err(topology)?;
        let edges = geometry::spaces(&projection)?;
        let grounded = geometry::grounded_edges(&projection, &edges, grounded_faces)?;
        let free_edge_indices: Vec<_> = (0..edges.edges().len())
            .filter(|i| !grounded.contains(i))
            .collect();
        let row_by_edge: BTreeMap<_, _> = free_edge_indices
            .iter()
            .enumerate()
            .map(|(row, &edge)| (edge, row))
            .collect();
        let mut rows = vec![BTreeSet::new(); free_edge_indices.len()];
        for index in 0..projection.elements.len() {
            let local = edges
                .element_edge_indices(index)
                .ok_or_else(|| topology("missing element edge space"))?;
            for edge in local {
                let Some(&row) = row_by_edge.get(&edge) else {
                    continue;
                };
                for other in local {
                    if let Some(&column) = row_by_edge.get(&other) {
                        rows[row].insert(column);
                    }
                }
            }
        }
        let mut offsets = vec![0];
        let mut columns = Vec::new();
        for row in rows {
            for column in row {
                columns.push(u64::try_from(column).map_err(invalid)?);
            }
            offsets.push(u64::try_from(columns.len()).map_err(invalid)?);
        }
        let dimension = u64::try_from(free_edge_indices.len()).map_err(invalid)?;
        Ok(Self {
            mesh_digest: mesh.canonical_digest,
            projection_digest: projection.projection_digest,
            edge_node_identities: edges
                .edges()
                .iter()
                .map(|edge| edge.node_identities)
                .collect(),
            grounded_faces: grounded_faces.to_vec(),
            free_edge_indices,
            pattern: SparsePattern::try_new(dimension, dimension, offsets, columns)
                .map_err(invalid)?,
        })
    }
    pub fn validate_for(&self, mesh: &SolverMeshArtifact) -> Result<(), MaxwellError> {
        if self != &Self::derive(mesh, &self.grounded_faces)? {
            return Err(invalid(
                "Maxwell system plan differs from exact canonical support",
            ));
        }
        Ok(())
    }
    pub fn free_edge_indices(&self) -> &[usize] {
        &self.free_edge_indices
    }
    pub fn edge_node_identities(&self) -> &[[StableDigest; 2]] {
        &self.edge_node_identities
    }
    pub fn pattern(&self) -> &SparsePattern {
        &self.pattern
    }
    pub fn grounded_faces(&self) -> &[StableDigest] {
        &self.grounded_faces
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, MaxwellError> {
        encode_contract(PREFIX, "maxwell-system-plan", self, LIMITS).map_err(invalid)
    }
    /// Structural decode only; validate_for binds it to the original canonical mesh.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, MaxwellError> {
        decode_contract(PREFIX, "maxwell-system-plan", bytes, LIMITS).map_err(invalid)
    }
    pub fn identity(&self) -> Result<Digest, MaxwellError> {
        Ok(Digest::sha256(self.canonical_bytes()?))
    }
}
