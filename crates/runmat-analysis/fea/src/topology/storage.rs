//! Counted construction storage shared by topology and its numerical consumers.
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use std::{borrow::Borrow, mem::size_of, ops::Index};
use thiserror::Error;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub(crate) enum TopologyStorageError {
    #[error("topology construction cancelled")]
    Cancelled,
    #[error("topology requested storage exceeds addressable slots")]
    Capacity,
    #[error("topology index has duplicate keys or an inconsistent count")]
    Index,
}
pub(crate) fn check_cancelled() -> Result<(), TopologyStorageError> {
    if crate::is_fea_cancelled() {
        Err(TopologyStorageError::Cancelled)
    } else {
        Ok(())
    }
}
pub(crate) fn reserve<T>(count: usize) -> Result<Vec<T>, TopologyStorageError> {
    check_cancelled()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| TopologyStorageError::Capacity)?;
    Ok(values)
}
pub(crate) fn multiply(a: usize, b: usize) -> Result<usize, TopologyStorageError> {
    a.checked_mul(b).ok_or(TopologyStorageError::Capacity)
}
pub(crate) fn count_add(a: usize, b: usize) -> Result<usize, TopologyStorageError> {
    a.checked_add(b).ok_or(TopologyStorageError::Capacity)
}
pub(crate) fn add(a: u64, b: u64) -> Result<u64, TopologyStorageError> {
    a.checked_add(b).ok_or(TopologyStorageError::Capacity)
}
pub(crate) fn slots<T>(count: usize) -> Result<u64, TopologyStorageError> {
    let bytes = multiply(count, size_of::<T>())?;
    if bytes > isize::MAX as usize {
        return Err(TopologyStorageError::Capacity);
    }
    u64::try_from(bytes).map_err(|_| TopologyStorageError::Capacity)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SortedValues<K, T>(Vec<(K, T)>);
impl<K: Copy + Ord, T> SortedValues<K, T> {
    pub(crate) fn derive(
        values: impl Iterator<Item = (K, T)>,
        count: usize,
    ) -> Result<Self, TopologyStorageError> {
        let mut entries = reserve(count)?;
        for value in values {
            if entries.len() % 1024 == 0 {
                check_cancelled()?;
            }
            if entries.len() == count {
                return Err(TopologyStorageError::Index);
            }
            entries.push(value);
        }
        if entries.len() != count {
            return Err(TopologyStorageError::Index);
        }
        Self::from_entries(entries)
    }
    /// Validate an already counted inventory without allocating a second copy.
    pub(crate) fn from_entries(mut entries: Vec<(K, T)>) -> Result<Self, TopologyStorageError> {
        check_cancelled()?;
        entries.sort_unstable_by_key(|v| v.0);
        if entries.windows(2).any(|p| p[0].0 == p[1].0) {
            return Err(TopologyStorageError::Index);
        }
        check_cancelled()?;
        Ok(Self(entries))
    }
    pub(crate) fn get<Q: Ord + ?Sized>(&self, key: &Q) -> Option<&T>
    where
        K: Borrow<Q>,
    {
        self.0
            .binary_search_by(|v| v.0.borrow().cmp(key))
            .ok()
            .map(|i| &self.0[i].1)
    }
    pub(crate) fn get_mut<Q: Ord + ?Sized>(&mut self, key: &Q) -> Option<&mut T>
    where
        K: Borrow<Q>,
    {
        self.0
            .binary_search_by(|v| v.0.borrow().cmp(key))
            .ok()
            .map(|i| &mut self.0[i].1)
    }
    pub(crate) fn entries_mut(&mut self) -> std::slice::IterMut<'_, (K, T)> {
        self.0.iter_mut()
    }
    pub(crate) fn entries(&self) -> std::slice::Iter<'_, (K, T)> {
        self.0.iter()
    }
    pub(crate) fn keys(&self) -> impl Iterator<Item = &K> {
        self.0.iter().map(|v| &v.0)
    }
    pub(crate) fn into_entries(self) -> std::vec::IntoIter<(K, T)> {
        self.0.into_iter()
    }
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}
impl<K: Copy + Ord, T> Index<&K> for SortedValues<K, T> {
    type Output = T;
    fn index(&self, key: &K) -> &T {
        self.get(key).expect("validated topology index")
    }
}
pub(crate) type CanonicalNodeValues<T> = SortedValues<StableDigest, T>;
pub(crate) fn solver_node_identities(
    mesh: &SolverMeshArtifact,
) -> Result<SortedValues<u64, StableDigest>, TopologyStorageError> {
    SortedValues::derive(
        mesh.topology
            .nodes
            .iter()
            .map(|n| (n.node_id, n.stable_identity)),
        mesh.topology.nodes.len(),
    )
}

mod dense;
pub(crate) use dense::{DenseIndexUnion, DenseUnionError};
mod grouped;
mod keys;
pub(crate) use keys::SortedKeys;

#[cfg(test)]
mod tests;

impl From<TopologyStorageError> for super::AnalysisProjectionError {
    fn from(error: TopologyStorageError) -> Self {
        match error {
            TopologyStorageError::Cancelled => Self::Cancelled,
            other => Self::ResourceEstimate(other.to_string()),
        }
    }
}
