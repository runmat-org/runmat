use serde::{Deserialize, Serialize};
use std::fmt;

use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_execution::Digest;

mod coordinates;
mod operations;
pub use operations::SparsePatternStatistics;

const PATTERN_IDENTITY_PREFIX: &[u8] = b"runmat-fea-sparse-pattern/v1\0";
const PATTERN_IDENTITY_DOMAIN: &str = "analysis.fea.sparse-pattern/v1";
const PATTERN_IDENTITY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 64_000_000, 256, 8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseIndexWidth {
    U64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparsePatternError {
    DimensionOverflow,
    StorageAllocation {
        elements: usize,
    },
    OffsetExceedsHostWidth {
        entry: usize,
        offset: u64,
    },
    ColumnExceedsHostWidth {
        entry: usize,
        column: u64,
    },
    InvalidRowOffsetCount {
        expected: usize,
        actual: usize,
    },
    FirstRowOffsetNotZero,
    NonMonotonicRowOffsets {
        row: usize,
    },
    EntryCountMismatch {
        offsets: usize,
        columns: usize,
    },
    ColumnOutOfBounds {
        entry: usize,
        column: u64,
        columns: u64,
    },
    ColumnsNotStrictlyIncreasing {
        row: usize,
    },
    ShapeMismatch,
    RowOutOfBounds {
        entry: usize,
        row: usize,
        rows: usize,
    },
}

impl fmt::Display for SparsePatternError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid canonical sparse pattern: {self:?}")
    }
}

impl std::error::Error for SparsePatternError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SparsePatternData")]
pub struct SparsePattern {
    rows: u64,
    columns: u64,
    index_width: SparseIndexWidth,
    row_offsets: std::sync::Arc<Vec<u64>>,
    column_indices: std::sync::Arc<Vec<u64>>,
}

#[derive(Deserialize)]
struct SparsePatternData {
    rows: u64,
    columns: u64,
    index_width: SparseIndexWidth,
    row_offsets: Vec<u64>,
    column_indices: Vec<u64>,
}

impl TryFrom<SparsePatternData> for SparsePattern {
    type Error = SparsePatternError;

    fn try_from(data: SparsePatternData) -> Result<Self, Self::Error> {
        if data.index_width != SparseIndexWidth::U64 {
            return Err(SparsePatternError::DimensionOverflow);
        }
        Self::try_new(
            data.rows,
            data.columns,
            data.row_offsets,
            data.column_indices,
        )
    }
}

impl SparsePattern {
    pub fn try_new(
        rows: u64,
        columns: u64,
        row_offsets: Vec<u64>,
        column_indices: Vec<u64>,
    ) -> Result<Self, SparsePatternError> {
        let row_count = usize::try_from(rows).map_err(|_| SparsePatternError::DimensionOverflow)?;
        usize::try_from(columns).map_err(|_| SparsePatternError::DimensionOverflow)?;
        let expected = row_count
            .checked_add(1)
            .ok_or(SparsePatternError::DimensionOverflow)?;
        if row_offsets.len() != expected {
            return Err(SparsePatternError::InvalidRowOffsetCount {
                expected,
                actual: row_offsets.len(),
            });
        }
        if row_offsets.first().copied() != Some(0) {
            return Err(SparsePatternError::FirstRowOffsetNotZero);
        }
        for row in 0..row_count {
            if row_offsets[row] > row_offsets[row + 1] {
                return Err(SparsePatternError::NonMonotonicRowOffsets { row });
            }
        }
        for (entry, &offset) in row_offsets.iter().enumerate() {
            if usize::try_from(offset).is_err() {
                return Err(SparsePatternError::OffsetExceedsHostWidth { entry, offset });
            }
        }
        let entries = usize::try_from(*row_offsets.last().unwrap_or(&0))
            .map_err(|_| SparsePatternError::DimensionOverflow)?;
        if entries != column_indices.len() {
            return Err(SparsePatternError::EntryCountMismatch {
                offsets: entries,
                columns: column_indices.len(),
            });
        }
        for (entry, &column) in column_indices.iter().enumerate() {
            if usize::try_from(column).is_err() {
                return Err(SparsePatternError::ColumnExceedsHostWidth { entry, column });
            }
            if column >= columns {
                return Err(SparsePatternError::ColumnOutOfBounds {
                    entry,
                    column,
                    columns,
                });
            }
        }
        for row in 0..row_count {
            let start = usize::try_from(row_offsets[row])
                .map_err(|_| SparsePatternError::DimensionOverflow)?;
            let end = usize::try_from(row_offsets[row + 1])
                .map_err(|_| SparsePatternError::DimensionOverflow)?;
            if column_indices[start..end]
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(SparsePatternError::ColumnsNotStrictlyIncreasing { row });
            }
        }
        Ok(Self {
            rows,
            columns,
            index_width: SparseIndexWidth::U64,
            row_offsets: std::sync::Arc::new(row_offsets),
            column_indices: std::sync::Arc::new(column_indices),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn index_width(&self) -> SparseIndexWidth {
        self.index_width
    }

    pub fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        let start = self.row_offsets[row] as usize;
        let end = self.row_offsets[row + 1] as usize;
        start..end
    }

    pub fn column(&self, entry: usize) -> usize {
        self.column_indices[entry] as usize
    }

    pub fn row_offsets(&self) -> &[u64] {
        &self.row_offsets
    }

    pub fn column_indices(&self) -> &[u64] {
        &self.column_indices
    }

    pub(crate) fn into_index_buffers(self) -> Result<(Vec<u64>, Vec<u64>), SparsePatternError> {
        fn take(buffer: std::sync::Arc<Vec<u64>>) -> Result<Vec<u64>, SparsePatternError> {
            match std::sync::Arc::try_unwrap(buffer) {
                Ok(unique) => Ok(unique),
                Err(shared) => {
                    let mut owned = Vec::new();
                    owned.try_reserve_exact(shared.len()).map_err(|_| {
                        SparsePatternError::StorageAllocation {
                            elements: shared.len(),
                        }
                    })?;
                    owned.extend_from_slice(&shared);
                    Ok(owned)
                }
            }
        }
        Ok((take(self.row_offsets)?, take(self.column_indices)?))
    }

    /// Find the first assembled entry not covered by this planned upper-bound pattern.
    /// Both row sets are canonical, so a merge walk uses no additional allocation.
    pub fn first_uncovered_entry(
        &self,
        assembled: &Self,
    ) -> Result<Option<(usize, usize)>, SparsePatternError> {
        if self.rows != assembled.rows || self.columns != assembled.columns {
            return Err(SparsePatternError::ShapeMismatch);
        }
        for row in 0..self.rows() {
            let planned = &self.column_indices[self.row_range(row)];
            let actual = &assembled.column_indices[assembled.row_range(row)];
            let mut planned_index = 0;
            for &column in actual {
                while planned
                    .get(planned_index)
                    .is_some_and(|&entry| entry < column)
                {
                    planned_index += 1;
                }
                if planned.get(planned_index) != Some(&column) {
                    return Ok(Some((row, column as usize)));
                }
            }
        }
        Ok(None)
    }

    /// Stable identity for symbolic sparsity reuse across devices and workers.
    pub fn identity(&self) -> Result<Digest, CanonicalCodecError> {
        // Streamed: a mesh-sized pattern is hashed without materializing its encoding.
        crate::contracts::canonical_digest::canonical_contract_digest(
            PATTERN_IDENTITY_PREFIX,
            PATTERN_IDENTITY_DOMAIN,
            self,
            PATTERN_IDENTITY_LIMITS,
        )
    }

    // Allocation-aware union and clone operations live in a focused child module.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_pattern_is_rejected() {
        assert!(matches!(
            SparsePattern::try_new(2, 2, vec![0, 2, 1], vec![0]),
            Err(SparsePatternError::NonMonotonicRowOffsets { row: 1 })
        ));
        assert!(matches!(
            SparsePattern::try_new(1, 2, vec![0, 2], vec![1, 1]),
            Err(SparsePatternError::ColumnsNotStrictlyIncreasing { row: 0 })
        ));
    }

    #[test]
    fn pattern_identity_is_deterministic_and_sensitive_to_structure() {
        let first = SparsePattern::try_new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
        let same = SparsePattern::try_new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
        let changed = SparsePattern::try_new(2, 2, vec![0, 2, 2], vec![0, 1]).unwrap();
        assert_eq!(first.identity().unwrap(), same.identity().unwrap());
        assert_ne!(first.identity().unwrap(), changed.identity().unwrap());
    }

    #[test]
    fn deserialization_cannot_bypass_pattern_validation() {
        let pattern = SparsePattern::try_new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
        let mut encoded = serde_json::to_value(pattern).unwrap();
        encoded["row_offsets"] = serde_json::json!([0, 2, 1]);
        assert!(serde_json::from_value::<SparsePattern>(encoded).is_err());
    }

    #[test]
    fn pattern_union_is_canonical_and_rejects_shape_mismatch() {
        let left = SparsePattern::try_new(3, 3, vec![0, 2, 3, 4], vec![0, 2, 1, 2]).unwrap();
        let right = SparsePattern::try_new(3, 3, vec![0, 1, 3, 4], vec![1, 0, 1, 2]).unwrap();
        let expected =
            SparsePattern::try_new(3, 3, vec![0, 3, 5, 6], vec![0, 1, 2, 0, 1, 2]).unwrap();
        assert_eq!(left.union(&right).unwrap(), expected);
        assert_eq!(right.union(&left).unwrap(), expected);
        assert_eq!(left.union(&left).unwrap(), left);

        let other_shape = SparsePattern::try_new(1, 1, vec![0, 1], vec![0]).unwrap();
        assert_eq!(
            left.union(&other_shape),
            Err(SparsePatternError::ShapeMismatch)
        );
    }

    #[test]
    fn planned_pattern_rejects_the_first_uncovered_assembled_entry() {
        let planned =
            SparsePattern::try_new(3, 3, vec![0, 2, 4, 6], vec![0, 2, 0, 1, 1, 2]).unwrap();
        let covered = SparsePattern::try_new(3, 3, vec![0, 1, 2, 3], vec![2, 0, 2]).unwrap();
        let uncovered = SparsePattern::try_new(3, 3, vec![0, 1, 3, 4], vec![2, 1, 2, 2]).unwrap();
        assert_eq!(planned.first_uncovered_entry(&covered), Ok(None));
        assert_eq!(planned.first_uncovered_entry(&uncovered), Ok(Some((1, 2))));
        let wrong_shape = SparsePattern::try_new(1, 1, vec![0, 1], vec![0]).unwrap();
        assert_eq!(
            planned.first_uncovered_entry(&wrong_shape),
            Err(SparsePatternError::ShapeMismatch)
        );
    }
}
