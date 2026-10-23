//! Allocation-free traversal of the canonical CSC merge. Callers own the
//! sorted index/ordinal table and final storage, never a second value payload.
use super::SparseTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseCscMergeEvent {
    Stored { row: usize, entry: usize },
    Update { row: usize, ordinal: usize },
    ColumnEnd { column: usize },
}

impl SparseTensor {
    /// The canonical retained-column/row projection used by CSC deletion.
    /// Mapping is pure; callers may repeat this for exact count and fill.
    pub fn visit_rebuilt_csc<E>(
        &self,
        source_columns: impl IntoIterator<Item = usize>,
        mut map_row: impl FnMut(usize) -> Option<usize>,
        mut visit: impl FnMut(SparseCscMergeEvent) -> Result<(), E>,
        mut checkpoint: impl FnMut() -> Result<(), E>,
        invalid: impl Fn(String) -> E,
    ) -> Result<(), E> {
        for (column, source_column) in source_columns.into_iter().enumerate() {
            checkpoint()?;
            if source_column >= self.cols {
                return Err(invalid(
                    "SparseTensor deletion column index exceeds dimension".into(),
                ));
            }
            for entry in self.col_ptrs[source_column]..self.col_ptrs[source_column + 1] {
                checkpoint()?;
                if let Some(row) = map_row(self.row_indices[entry]) {
                    visit(SparseCscMergeEvent::Stored { row, entry })?;
                }
            }
            visit(SparseCscMergeEvent::ColumnEnd { column })?;
        }
        Ok(())
    }
    /// Visit one CSC merge in column/row order. Updates must be sorted by
    /// (linear index, source ordinal); the final ordinal wins for duplicates.
    /// Zero elision belongs to the supplied typed value consumer. Count/fill
    /// passes can reuse this same traversal without copying any values.
    pub fn visit_merged_linear_updates<E>(
        &self,
        rows: usize,
        cols: usize,
        updates: &[(usize, usize)],
        mut visit: impl FnMut(SparseCscMergeEvent) -> Result<(), E>,
        invalid: impl Fn(String) -> E,
        mut checkpoint: impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        if rows < self.rows || cols < self.cols {
            return Err(invalid(format!(
                "SparseTensor cannot shrink shape ({}, {}) to ({rows}, {cols})",
                self.rows, self.cols
            )));
        }
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid("SparseTensor assignment dimensions overflow usize".into()))?;
        let mut previous = None;
        for &(linear, ordinal) in updates {
            checkpoint()?;
            if linear >= total {
                return Err(invalid(format!(
                    "SparseTensor assignment linear index {linear} exceeds {total} elements"
                )));
            }
            if previous.is_some_and(|key| key > (linear, ordinal)) {
                return Err(invalid("SparseTensor update order is not canonical".into()));
            }
            previous = Some((linear, ordinal));
        }
        let mut cursor = 0;
        for column in 0..cols {
            checkpoint()?;
            let mut stored = if column < self.cols {
                self.col_ptrs[column]
            } else {
                self.nnz()
            };
            let stored_end = if column < self.cols {
                self.col_ptrs[column + 1]
            } else {
                self.nnz()
            };
            while cursor < updates.len() && updates[cursor].0 / rows == column {
                checkpoint()?;
                let linear = updates[cursor].0;
                let row = linear % rows;
                let mut last = cursor;
                while last + 1 < updates.len() && updates[last + 1].0 == linear {
                    checkpoint()?;
                    last += 1;
                }
                while stored < stored_end && self.row_indices[stored] < row {
                    checkpoint()?;
                    visit(SparseCscMergeEvent::Stored {
                        row: self.row_indices[stored],
                        entry: stored,
                    })?;
                    stored += 1;
                }
                if stored < stored_end && self.row_indices[stored] == row {
                    stored += 1;
                }
                visit(SparseCscMergeEvent::Update {
                    row,
                    ordinal: updates[last].1,
                })?;
                cursor = last + 1;
            }
            while stored < stored_end {
                checkpoint()?;
                visit(SparseCscMergeEvent::Stored {
                    row: self.row_indices[stored],
                    entry: stored,
                })?;
                stored += 1;
            }
            visit(SparseCscMergeEvent::ColumnEnd { column })?;
        }
        Ok(())
    }
}
