//! Counted coordinate/identity indexes. Borrow physical nodes; retain only compact keys/values.
use super::{
    profile::{check_cancelled, reserve},
    ConstraintPlanError, SolverMeshArtifact, StableDigest,
};

pub(super) struct NodeCoordinates(Vec<(StableDigest, [f64; 3])>);
impl NodeCoordinates {
    pub(super) fn derive(
        values: impl Iterator<Item = (StableDigest, [f64; 3])>,
        count: usize,
    ) -> Result<Self, ConstraintPlanError> {
        let mut index = reserve(count)?;
        for value in values {
            if index.len() % 1024 == 0 {
                check_cancelled()?;
            }
            if index.len() == count {
                return Err(invalid());
            }
            index.push(value);
        }
        if index.len() != count {
            return Err(invalid());
        }
        index.sort_unstable_by_key(|(key, _)| *key);
        if index.windows(2).any(|p| p[0].0 == p[1].0) {
            return Err(invalid());
        }
        Ok(Self(index))
    }
    pub(super) fn get(&self, key: StableDigest) -> Option<[f64; 3]> {
        self.0
            .binary_search_by_key(&key, |(key, _)| *key)
            .ok()
            .map(|i| self.0[i].1)
    }
}

pub(super) struct NodeIdentities(Vec<(u64, StableDigest)>);
impl NodeIdentities {
    pub(super) fn derive(mesh: &SolverMeshArtifact) -> Result<Self, ConstraintPlanError> {
        let mut index = reserve(mesh.topology.nodes.len())?;
        for (ordinal, node) in mesh.topology.nodes.iter().enumerate() {
            if ordinal % 1024 == 0 {
                check_cancelled()?;
            }
            index.push((node.node_id, node.stable_identity));
        }
        index.sort_unstable_by_key(|(key, _)| *key);
        if index.windows(2).any(|p| p[0].0 == p[1].0) {
            return Err(invalid());
        }
        Ok(Self(index))
    }
    pub(super) fn get(&self, key: u64) -> Option<StableDigest> {
        self.0
            .binary_search_by_key(&key, |(key, _)| *key)
            .ok()
            .map(|i| self.0[i].1)
    }
}
fn invalid() -> ConstraintPlanError {
    ConstraintPlanError::Invalid(
        "pairing node index has duplicate keys or an inconsistent count".into(),
    )
}
