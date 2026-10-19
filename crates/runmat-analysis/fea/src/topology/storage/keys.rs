//! Sorted set over an already counted/reserved source inventory; duplicates are aliases.
use super::{check_cancelled, TopologyStorageError};

pub(crate) struct SortedKeys<K>(Vec<K>);
impl<K: Ord> SortedKeys<K> {
    pub(crate) fn from_entries(mut entries: Vec<K>) -> Result<Self, TopologyStorageError> {
        check_cancelled()?;
        entries.sort_unstable();
        entries.dedup();
        check_cancelled()?;
        Ok(Self(entries))
    }
    pub(crate) fn contains(&self, key: &K) -> bool {
        self.0.binary_search(key).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_sparse_keys_and_duplicate_aliases_share_one_set() {
        let keys = SortedKeys::from_entries(vec![u64::MAX, 0, 13, 13, 0]).unwrap();
        assert_eq!(keys.0, [0, 13, u64::MAX]);
        assert!(keys.contains(&u64::MAX));
        assert!(!keys.contains(&12));
    }
}
