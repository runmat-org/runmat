//! Counted multivalue indices retain aliases without allocator-dependent trees.
use super::*;

impl<K: Copy + Ord, T: Copy + Ord> SortedValues<K, Vec<T>> {
    pub(crate) fn derive_grouped(
        values: impl Iterator<Item = (K, T)>,
        count: usize,
    ) -> Result<Self, TopologyStorageError> {
        let mut pairs = reserve(count)?;
        for value in values {
            if pairs.len() % 1024 == 0 {
                check_cancelled()?;
            }
            if pairs.len() == count {
                return Err(TopologyStorageError::Index);
            }
            pairs.push(value);
        }
        if pairs.len() != count {
            return Err(TopologyStorageError::Index);
        }
        pairs.sort_unstable();
        pairs.dedup();
        let group_count = pairs
            .iter()
            .enumerate()
            .filter(|(i, p)| *i == 0 || pairs[*i - 1].0 != p.0)
            .count();
        let mut groups = reserve(group_count)?;
        let mut start = 0;
        while start < pairs.len() {
            check_cancelled()?;
            let key = pairs[start].0;
            let end = start + pairs[start..].partition_point(|p| p.0 == key);
            let mut values = reserve(end - start)?;
            values.extend(pairs[start..end].iter().map(|p| p.1));
            groups.push((key, values));
            start = end;
        }
        Ok(Self(groups))
    }

    /// Requested construction heap: sorted input pairs coexist with final
    /// group headers and values. All-distinct keys bound every grouping.
    pub(crate) fn grouped_construction_resident_bytes(
        count: usize,
    ) -> Result<u64, TopologyStorageError> {
        add(
            add(slots::<(K, T)>(count)?, slots::<(K, Vec<T>)>(count)?)?,
            slots::<T>(count)?,
        )
    }
}

#[cfg(test)]
mod tests;
