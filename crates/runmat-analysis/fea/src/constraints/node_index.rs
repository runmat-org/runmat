//! Compact counted indexes for canonical constraint node data.
use super::{
    workspace::{check_cancelled, reserve},
    ConstraintPlanError,
};
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use std::ops::Index;

pub(super) struct NodeValues<T>(Vec<(StableDigest, T)>);
impl<T> NodeValues<T> {
    pub(super) fn derive(
        values: impl Iterator<Item = (StableDigest, T)>,
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
    pub(super) fn get(&self, key: &StableDigest) -> Option<&T> {
        self.0
            .binary_search_by_key(key, |(key, _)| *key)
            .ok()
            .map(|i| &self.0[i].1)
    }
    pub(super) fn get_mut(&mut self, key: &StableDigest) -> Option<&mut T> {
        self.0
            .binary_search_by_key(key, |(key, _)| *key)
            .ok()
            .map(|i| &mut self.0[i].1)
    }
    pub(super) fn entries(&self) -> std::slice::Iter<'_, (StableDigest, T)> {
        self.0.iter()
    }
    pub(super) fn into_entries(self) -> std::vec::IntoIter<(StableDigest, T)> {
        self.0.into_iter()
    }
}
impl<T> Index<&StableDigest> for NodeValues<T> {
    type Output = T;
    fn index(&self, key: &StableDigest) -> &T {
        self.get(key).expect("validated constraint node index")
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
        "constraint node index has duplicate keys or an inconsistent count".into(),
    )
}
