//! Canonical CSR from an owned, pre-admitted coordinate inventory. No tree-node storage.
use super::*;
impl SparsePattern {
    /// Consumes caller-admitted coordinates, sorts and removes duplicate support in place.
    /// Peak requested payload includes this input plus the two returned CSR buffers.
    pub fn from_coordinates(
        rows: usize,
        columns: usize,
        mut entries: Vec<(usize, usize)>,
    ) -> Result<Self, SparsePatternError> {
        let offset_count = rows
            .checked_add(1)
            .ok_or(SparsePatternError::DimensionOverflow)?;
        for (entry, &(row, column)) in entries.iter().enumerate() {
            if row >= rows {
                return Err(SparsePatternError::RowOutOfBounds { entry, row, rows });
            }
            if column >= columns {
                return Err(SparsePatternError::ColumnOutOfBounds {
                    entry,
                    column: column as u64,
                    columns: columns as u64,
                });
            }
        }
        entries.sort_unstable();
        entries.dedup();
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(offset_count).map_err(|_| {
            SparsePatternError::StorageAllocation {
                elements: offset_count,
            }
        })?;
        let mut indices = Vec::new();
        indices.try_reserve_exact(entries.len()).map_err(|_| {
            SparsePatternError::StorageAllocation {
                elements: entries.len(),
            }
        })?;
        offsets.push(0);
        let mut cursor = 0;
        for row in 0..rows {
            while cursor < entries.len() && entries[cursor].0 == row {
                indices.push(entries[cursor].1 as u64);
                cursor += 1;
            }
            offsets.push(indices.len() as u64);
        }
        Self::try_new(rows as u64, columns as u64, offsets, indices)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unordered_duplicate_coordinates_retain_rectangular_empty_rows_and_stable_identity() {
        let entries = vec![(2, 4), (0, 3), (2, 4), (0, 0), (0, 3)];
        let expected = SparsePattern::try_new(4, 5, vec![0, 2, 2, 3, 3], vec![0, 3, 4]).unwrap();
        let actual = SparsePattern::from_coordinates(4, 5, entries.clone()).unwrap();
        assert_eq!(actual, expected);
        let reverse =
            SparsePattern::from_coordinates(4, 5, entries.into_iter().rev().collect()).unwrap();
        assert_eq!(reverse.identity().unwrap(), expected.identity().unwrap());
        assert_eq!(
            SparsePattern::from_coordinates(0, 0, vec![])
                .unwrap()
                .row_offsets(),
            &[0]
        );
        assert!(matches!(
            SparsePattern::from_coordinates(1, 2, vec![(1, 0)]),
            Err(SparsePatternError::RowOutOfBounds { .. })
        ));
        assert!(matches!(
            SparsePattern::from_coordinates(1, 2, vec![(0, 2)]),
            Err(SparsePatternError::ColumnOutOfBounds { .. })
        ));
        assert_eq!(
            SparsePattern::from_coordinates(usize::MAX, 0, vec![]),
            Err(SparsePatternError::DimensionOverflow)
        );
    }
}
