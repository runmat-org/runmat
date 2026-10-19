//! Material-independent exact sparse complex-block support.
//! Canonical geometry projection only; no physical matrices/form/source integration.
use super::{geometry, invalid, topology, MaxwellError};
use crate::{algebra::SparsePattern, topology::AnalysisProjection};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const PREFIX: &[u8] = b"runmat-maxwell-system-plan/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(512 * 1024 * 1024, 64_000_000, 256, 16);
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellSystemPlan {
    mesh_digest: StableDigest,
    projection_digest: Digest,
    edge_node_identities: Vec<[StableDigest; 2]>,
    free_edge_indices: Vec<usize>,
    block_pattern: SparsePattern,
}
impl MaxwellSystemPlan {
    pub fn derive(mesh: &SolverMeshArtifact) -> Result<Self, MaxwellError> {
        let projection = AnalysisProjection::derive(mesh).map_err(topology)?;
        let (edges, exterior) = geometry::spaces(&projection)?;
        let exterior: BTreeSet<_> = exterior.into_iter().collect();
        let free_edge_indices: Vec<_> = (0..edges.edges().len())
            .filter(|i| !exterior.contains(i))
            .collect();
        let free = free_edge_indices.len();
        let dimension = free
            .checked_mul(2)
            .ok_or_else(|| invalid("Maxwell dimension overflow"))?;
        let row_by_edge: BTreeMap<_, _> = free_edge_indices
            .iter()
            .enumerate()
            .map(|(row, &edge)| (edge, row))
            .collect();
        let mut rows = vec![BTreeSet::new(); dimension];
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
                        for component in 0..2 {
                            rows[row + component * free].insert(column);
                            rows[row + component * free].insert(column + free);
                        }
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
        let dimension = u64::try_from(dimension).map_err(invalid)?;
        Ok(Self {
            mesh_digest: mesh.canonical_digest,
            projection_digest: projection.projection_digest,
            edge_node_identities: edges
                .edges()
                .iter()
                .map(|edge| edge.node_identities)
                .collect(),
            free_edge_indices,
            block_pattern: SparsePattern::try_new(dimension, dimension, offsets, columns)
                .map_err(invalid)?,
        })
    }
    pub fn validate_for(&self, mesh: &SolverMeshArtifact) -> Result<(), MaxwellError> {
        if self != &Self::derive(mesh)? {
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
    pub fn block_pattern(&self) -> &SparsePattern {
        &self.block_pattern
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
