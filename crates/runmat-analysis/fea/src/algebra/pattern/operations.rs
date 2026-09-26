use super::{SparsePattern, SparsePatternError};

impl SparsePattern {
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
            row_offsets,
            column_indices,
        })
    }

    /// Deterministic row-wise union; reserve the actual union size, not both inputs' sum.
    pub fn union(&self, other: &Self) -> Result<Self, SparsePatternError> {
        if self.rows != other.rows || self.columns != other.columns {
            return Err(SparsePatternError::ShapeMismatch);
        }
        let count = (0..self.rows()).try_fold(0_usize, |count, row| {
            count
                .checked_add(merged_columns(self, other, row).count())
                .ok_or(SparsePatternError::DimensionOverflow)
        })?;
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
