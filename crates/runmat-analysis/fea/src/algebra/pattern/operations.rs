use super::{SparsePattern, SparsePatternError};

/// Exact canonical support counts; no symmetry or numerical-value claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparsePatternStatistics {
    pub rows: usize,
    pub columns: usize,
    pub nonzeros: usize,
    pub maximum_row_nonzeros: usize,
}
impl SparsePatternStatistics {
    pub fn resident_bytes(self) -> Result<u64, SparsePatternError> {
        self.rows
            .checked_add(1)
            .and_then(|n| n.checked_add(self.nonzeros))
            .and_then(|n| u64::try_from(n).ok())
            .and_then(|n| n.checked_mul(8))
            .ok_or(SparsePatternError::DimensionOverflow)
    }
}

impl SparsePattern {
    /// Requested heap for one independently owned CSR, including its shared owners.
    /// Logical clones share these allocations; callers charge a backing owner once.
    pub(crate) fn retained_storage_bytes(&self) -> Result<u64, SparsePatternError> {
        Self::storage_bytes_for_capacities(
            self.row_offsets.capacity(),
            self.column_indices.capacity(),
        )
    }

    pub(crate) fn retained_storage_bytes_for_counts(
        rows: usize,
        nonzeros: usize,
    ) -> Result<u64, SparsePatternError> {
        Self::storage_bytes_for_capacities(
            rows.checked_add(1)
                .ok_or(SparsePatternError::DimensionOverflow)?,
            nonzeros,
        )
    }

    fn storage_bytes_for_capacities(
        offsets: usize,
        indices: usize,
    ) -> Result<u64, SparsePatternError> {
        let bytes = super::super::storage::shared_vector_storage_bytes::<u64>;
        bytes(offsets)
            .and_then(|offsets| offsets.checked_add(bytes(indices)?))
            .ok_or(SparsePatternError::DimensionOverflow)
    }

    /// Copies canonical index buffers with fallible allocation and no geometric growth.
    pub fn try_clone(&self) -> Result<Self, SparsePatternError> {
        let mut row_offsets = reserve(self.row_offsets.len())?;
        row_offsets.extend_from_slice(&self.row_offsets);
        let mut column_indices = reserve(self.column_indices.len())?;
        column_indices.extend_from_slice(&self.column_indices);
        Ok(Self {
            rows: self.rows,
            columns: self.columns,
            index_width: self.index_width,
            row_offsets: std::sync::Arc::new(row_offsets),
            column_indices: std::sync::Arc::new(column_indices),
        })
    }

    /// Count the original ordered union without allocating support buffers.
    pub fn union_statistics(
        &self,
        other: &Self,
    ) -> Result<SparsePatternStatistics, SparsePatternError> {
        if self.rows != other.rows || self.columns != other.columns {
            return Err(SparsePatternError::ShapeMismatch);
        }
        let mut result = SparsePatternStatistics {
            rows: self.rows(),
            columns: self.columns(),
            nonzeros: 0,
            maximum_row_nonzeros: 0,
        };
        for row in 0..self.rows() {
            let width = merged_columns(self, other, row).count();
            result.nonzeros = result
                .nonzeros
                .checked_add(width)
                .ok_or(SparsePatternError::DimensionOverflow)?;
            result.maximum_row_nonzeros = result.maximum_row_nonzeros.max(width);
        }
        Ok(result)
    }
    pub(crate) fn union_columns<'a>(
        &'a self,
        other: &'a Self,
        row: usize,
    ) -> impl Iterator<Item = u64> + 'a {
        merged_columns(self, other, row)
    }

    /// Deterministic row-wise union; reserve the actual union size, not both inputs' sum.
    pub fn union(&self, other: &Self) -> Result<Self, SparsePatternError> {
        let count = self.union_statistics(other)?.nonzeros;
        let mut row_offsets = reserve(
            self.rows()
                .checked_add(1)
                .ok_or(SparsePatternError::DimensionOverflow)?,
        )?;
        let mut column_indices = reserve(count)?;
        row_offsets.push(0);
        for row in 0..self.rows() {
            column_indices.extend(merged_columns(self, other, row));
            row_offsets.push(
                u64::try_from(column_indices.len())
                    .map_err(|_| SparsePatternError::DimensionOverflow)?,
            );
        }
        Self::try_new(self.rows, self.columns, row_offsets, column_indices)
    }
}

fn merged_columns<'a>(
    left: &'a SparsePattern,
    right: &'a SparsePattern,
    row: usize,
) -> impl Iterator<Item = u64> + 'a {
    let mut left = left.column_indices[left.row_range(row)]
        .iter()
        .copied()
        .peekable();
    let mut right = right.column_indices[right.row_range(row)]
        .iter()
        .copied()
        .peekable();
    std::iter::from_fn(move || match (left.peek(), right.peek()) {
        (Some(a), Some(b)) if a < b => left.next(),
        (Some(a), Some(b)) if b < a => right.next(),
        (Some(_), Some(_)) => {
            right.next();
            left.next()
        }
        (Some(_), None) => left.next(),
        (None, Some(_)) => right.next(),
        (None, None) => None,
    })
}

fn reserve(elements: usize) -> Result<Vec<u64>, SparsePatternError> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(elements)
        .map_err(|_| SparsePatternError::StorageAllocation { elements })?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_is_canonical_for_overlap_empty_rows_and_rectangular_patterns() {
        let left = SparsePattern::try_new(3, 5, vec![0, 2, 2, 3], vec![0, 3, 4]).unwrap();
        let right = SparsePattern::try_new(3, 5, vec![0, 2, 3, 4], vec![1, 3, 2, 4]).unwrap();
        let expected = SparsePattern::try_new(3, 5, vec![0, 3, 4, 5], vec![0, 1, 3, 2, 4]).unwrap();
        assert_eq!(left.union(&right).unwrap(), expected);
        assert_eq!(right.union(&left).unwrap(), expected);
        assert_eq!(left.union(&left).unwrap(), left);
        assert_eq!(
            left.try_clone().unwrap().identity().unwrap(),
            left.identity().unwrap()
        );
        let empty = SparsePattern::try_new(0, 0, vec![0], vec![]).unwrap();
        assert_eq!(empty.union(&empty).unwrap(), empty);
        assert!(matches!(
            left.union(&empty),
            Err(SparsePatternError::ShapeMismatch)
        ));
        assert_eq!(
            reserve(usize::MAX),
            Err(SparsePatternError::StorageAllocation {
                elements: usize::MAX
            })
        );
    }
}
