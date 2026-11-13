//! Compact row index over an immutable original CSC constant. Shared by
//! coefficient proposal and terminal pullback; no dense row-pointer mirror.
use super::ScalarCooperation;
use crate::{BuiltinResult, RuntimeError};
#[derive(Clone, Copy)]
pub(crate) struct Entry {
    pub row: usize,
    pub column: usize,
    pub entry: usize,
}
pub(crate) struct CscRows {
    entries: Vec<Entry>,
}
impl CscRows {
    pub(crate) async fn prepare(
        columns: usize,
        pointers: &[usize],
        rows: &[usize],
        allocation: impl Fn(runmat_value::ExactHostNumericAllocationError) -> RuntimeError,
        reserve: impl Fn(std::collections::TryReserveError) -> RuntimeError,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut entries = runmat_value::exact_capacity_vec(rows.len()).map_err(allocation)?;
        for column in 0..columns {
            for entry in pointers[column]..pointers[column + 1] {
                entries.push(Entry {
                    row: rows[entry],
                    column,
                    entry,
                });
                cooperation.completed(1).await?;
            }
            cooperation.completed(1).await?;
        }
        use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
        let mut sort = StableSortWork::new(&mut entries, |a, b| {
            (a.row, a.column).cmp(&(b.row, b.column))
        });
        loop {
            match sort.advance().map_err(|source| reserve(source.0))? {
                SortProgress::Pending(work) => cooperation.completed(work).await?,
                SortProgress::Complete(work) => {
                    cooperation.completed(work).await?;
                    break;
                }
            }
        }
        drop(sort);
        Ok(Self { entries })
    }
    pub(crate) fn row(&self, row: usize) -> &[Entry] {
        let first = self.entries.partition_point(|entry| entry.row < row);
        let last = self.entries.partition_point(|entry| entry.row <= row);
        &self.entries[first..last]
    }
}
