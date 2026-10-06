use super::super::HostIndexBuffer;
use std::num::NonZeroUsize;

/// Incremental validator for one owned canonical sparse CSC topology.
///
/// This type owns no scheduling policy. Callers choose how many index entries
/// to validate per turn and may retain it across a cooperative suspension.
pub struct SparseCscValidation {
    rows: usize,
    cols: usize,
    col_ptrs: HostIndexBuffer,
    row_indices: HostIndexBuffer,
    values_len: usize,
    pointer: usize,
    column: usize,
    offset: usize,
}

/// Sealed proof that an owned CSC topology has passed every structural check.
/// Its fields are private so unchecked sparse tensors cannot be constructed by
/// callers in another crate.
pub struct ValidatedSparseCscParts {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) col_ptrs: HostIndexBuffer,
    pub(super) row_indices: HostIndexBuffer,
    pub(super) values_len: usize,
}

impl SparseCscValidation {
    pub fn new(
        rows: usize,
        cols: usize,
        col_ptrs: HostIndexBuffer,
        row_indices: HostIndexBuffer,
        values_len: usize,
    ) -> Result<Self, String> {
        if col_ptrs.len() != cols.saturating_add(1) {
            return Err(format!(
                "SparseTensor col_ptrs length {} doesn't match cols {}",
                col_ptrs.len(),
                cols
            ));
        }
        if row_indices.len() != values_len {
            return Err(format!(
                "SparseTensor row index length {} doesn't match value length {}",
                row_indices.len(),
                values_len
            ));
        }
        if col_ptrs.first().copied().unwrap_or(usize::MAX) != 0 {
            return Err("SparseTensor col_ptrs must start at 0".to_string());
        }
        if col_ptrs.last().copied().unwrap_or(usize::MAX) != values_len {
            return Err("SparseTensor final col_ptr must equal nnz".to_string());
        }
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values_len,
            pointer: 0,
            column: 0,
            offset: 0,
        })
    }

    /// Validate at most `maximum_work` items. One work item is either one
    /// adjacent column-pointer relation, one row index, or one empty-column
    /// transition. Returns true only after both ordered phases are complete.
    pub fn advance(&mut self, maximum_work: NonZeroUsize) -> Result<bool, String> {
        let mut remaining = maximum_work.get();
        while self.pointer < self.cols {
            if self.col_ptrs[self.pointer] > self.col_ptrs[self.pointer + 1] {
                return Err("SparseTensor col_ptrs must be nondecreasing".to_string());
            }
            self.pointer += 1;
            remaining -= 1;
            if remaining == 0 {
                if self.pointer == self.cols && self.row_indices.is_empty() {
                    self.column = self.cols;
                    return Ok(true);
                }
                return Ok(false);
            }
        }
        while self.column < self.cols {
            let start = self.col_ptrs[self.column];
            let end = self.col_ptrs[self.column + 1];
            let cursor = start.saturating_add(self.offset);
            if cursor >= end {
                self.column += 1;
                self.offset = 0;
                remaining -= 1;
                if remaining == 0 {
                    return Ok(self.column == self.cols);
                }
                continue;
            }
            let limit = end.min(cursor.saturating_add(remaining));
            let mut previous = if cursor == start {
                None
            } else {
                Some(self.row_indices[cursor - 1])
            };
            for &row in &self.row_indices[cursor..limit] {
                if row >= self.rows {
                    return Err(format!(
                        "SparseTensor row index {row} exceeds rows {}",
                        self.rows
                    ));
                }
                if previous.is_some_and(|prior| prior >= row) {
                    return Err("SparseTensor row indices must be sorted and unique".to_string());
                }
                previous = Some(row);
            }
            let completed = limit - cursor;
            self.offset += completed;
            remaining -= completed;
            if remaining == 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Consume a completed cursor without doing additional validation work.
    pub fn into_validated(self) -> Result<ValidatedSparseCscParts, Self> {
        if self.pointer != self.cols || self.column != self.cols {
            return Err(self);
        }
        Ok(self.validated_parts())
    }

    /// Compatibility drain for synchronous constructors. Cooperative callers
    /// use `advance` followed by `into_validated` so completion is O(1).
    pub fn finish_synchronously(mut self) -> Result<ValidatedSparseCscParts, String> {
        let unbounded = NonZeroUsize::new(usize::MAX).expect("usize::MAX is nonzero");
        while !self.advance(unbounded)? {}
        Ok(self.validated_parts())
    }

    fn validated_parts(self) -> ValidatedSparseCscParts {
        ValidatedSparseCscParts {
            rows: self.rows,
            cols: self.cols,
            col_ptrs: self.col_ptrs,
            row_indices: self.row_indices,
            values_len: self.values_len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(
        rows: usize,
        cols: usize,
        pointers: Vec<usize>,
        indices: Vec<usize>,
        values: usize,
        quantum: usize,
    ) -> Result<ValidatedSparseCscParts, String> {
        let mut validation =
            SparseCscValidation::new(rows, cols, pointers.into(), indices.into(), values)?;
        let quantum = NonZeroUsize::new(quantum).expect("test quantum is nonzero");
        while !validation.advance(quantum)? {}
        validation
            .into_validated()
            .map_err(|_| "validation unexpectedly incomplete".to_string())
    }

    #[test]
    fn incremental_and_single_turn_validation_accept_the_same_canonical_topology() {
        let incremental =
            validate(4, 3, vec![0, 2, 2, 4], vec![0, 3, 1, 2], 4, 1).expect("incremental topology");
        let single = validate(4, 3, vec![0, 2, 2, 4], vec![0, 3, 1, 2], 4, usize::MAX)
            .expect("single-turn topology");
        assert_eq!(incremental.rows, single.rows);
        assert_eq!(incremental.cols, single.cols);
        assert_eq!(&*incremental.col_ptrs, &*single.col_ptrs);
        assert_eq!(&*incremental.row_indices, &*single.row_indices);
    }

    #[test]
    fn malformed_counts_pointers_and_rows_fail_at_exact_chunk_boundaries() {
        assert!(SparseCscValidation::new(2, 1, vec![0].into(), Vec::new().into(), 0).is_err());
        assert!(SparseCscValidation::new(2, 1, vec![1, 1].into(), vec![0].into(), 1).is_err());
        assert!(SparseCscValidation::new(2, 1, vec![0, 0].into(), vec![0].into(), 1).is_err());

        let mut decreasing =
            SparseCscValidation::new(3, 3, vec![0, 2, 1, 2].into(), vec![0, 1].into(), 2)
                .expect("header defers pointer ordering");
        assert!(!decreasing.advance(NonZeroUsize::new(1).unwrap()).unwrap());
        assert!(decreasing.advance(NonZeroUsize::new(1).unwrap()).is_err());

        let mut duplicate =
            SparseCscValidation::new(3, 1, vec![0, 3].into(), vec![0, 1, 1].into(), 3).unwrap();
        assert!(!duplicate.advance(NonZeroUsize::new(2).unwrap()).unwrap());
        assert!(!duplicate.advance(NonZeroUsize::new(1).unwrap()).unwrap());
        assert!(duplicate.advance(NonZeroUsize::new(1).unwrap()).is_err());

        let mut out_of_range =
            SparseCscValidation::new(2, 1, vec![0, 2].into(), vec![0, 2].into(), 2).unwrap();
        assert!(!out_of_range.advance(NonZeroUsize::new(1).unwrap()).unwrap());
        assert!(!out_of_range.advance(NonZeroUsize::new(1).unwrap()).unwrap());
        assert!(out_of_range.advance(NonZeroUsize::new(1).unwrap()).is_err());

        let mut multiply_invalid =
            SparseCscValidation::new(1, 3, vec![0, 1, 0, 1].into(), vec![usize::MAX].into(), 1)
                .unwrap();
        let error = multiply_invalid
            .advance(NonZeroUsize::new(3).unwrap())
            .expect_err("pointer precedence before invalid row");
        assert_eq!(error, "SparseTensor col_ptrs must be nondecreasing");
    }

    #[test]
    fn empty_columns_consume_validation_work() {
        let mut validation =
            SparseCscValidation::new(1, 4, vec![0, 0, 0, 0, 0].into(), Vec::new().into(), 0)
                .unwrap();
        assert!(!validation.advance(NonZeroUsize::new(2).unwrap()).unwrap());
        assert!(validation.advance(NonZeroUsize::new(2).unwrap()).unwrap());
    }

    #[test]
    fn validated_constructor_preserves_class_and_shared_index_owners() {
        use crate::{HostNumericBuffer, NumericStorage, SparseTensor};

        let pointers = HostIndexBuffer::new(vec![0, 1, 2]);
        let rows = HostIndexBuffer::new(vec![0, 1]);
        let pointer_owner = pointers.clone();
        let row_owner = rows.clone();
        let mut validation = SparseCscValidation::new(2, 2, pointers, rows, 2).unwrap();
        while !validation.advance(NonZeroUsize::new(1).unwrap()).unwrap() {}
        let parts = validation.into_validated().ok().unwrap();
        let actual = SparseTensor::from_validated_numeric(
            parts,
            HostNumericBuffer::from_numeric_storage(NumericStorage::F32(vec![1.0, 2.0])),
        )
        .unwrap();
        let expected =
            SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        assert_eq!(actual, expected);
        assert!(actual.col_ptrs.shares_allocation_with(&pointer_owner));
        assert!(actual.row_indices.shares_allocation_with(&row_owner));
    }
}
