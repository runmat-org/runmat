//! Existing-sheet resizing and cell overlay retain the caller's cooperation owner.

use super::*;

impl XlsTable {
    pub(in crate::builtins::io::tabular::xlswrite) async fn overlay(
        self,
        start: RangeStart,
        incoming: XlsTable,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        let required_rows = start.row.checked_add(incoming.rows).ok_or_else(|| {
            xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range row overflow")
        })?;
        let required_cols = start.col.checked_add(incoming.cols).ok_or_else(|| {
            xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range column overflow")
        })?;
        checked_cell_count(required_rows, required_cols, "worksheet")?;
        let mut result = if required_rows > self.rows || required_cols > self.cols {
            let rows = required_rows.max(self.rows);
            let cols = required_cols.max(self.cols);
            self.resize(rows, cols, cooperation, selected).await?
        } else {
            self
        };
        let mut work = 0;
        for (index, value) in incoming.cells.into_iter().enumerate() {
            let row = index / incoming.cols;
            let col = index % incoming.cols;
            let dest = (start.row + row) * result.cols + start.col + col;
            result.cells[dest] = value;
            work += 1;
            if work == 1024 {
                checkpoint_table(cooperation, selected, work).await?;
                work = 0;
            }
        }
        checkpoint_table(cooperation, selected, work).await?;
        Ok(result)
    }

    async fn resize(
        self,
        rows: usize,
        cols: usize,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        checked_cell_count(rows, cols, "worksheet")?;
        let count = rows * cols;
        let mut resized = Vec::new();
        resized.try_reserve_exact(count).map_err(|source| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: worksheet resize allocation failed",
                source,
            )
        })?;
        while resized.len() < count {
            let chunk = (count - resized.len()).min(1024);
            resized.resize(resized.len() + chunk, CellValue::Empty);
            checkpoint_table(cooperation, selected, chunk).await?;
        }
        let previous_cols = self.cols;
        let previous_cells = self.cells;
        let mut work = 0;
        for (index, value) in previous_cells.into_iter().enumerate() {
            let row = index / previous_cols;
            let col = index % previous_cols;
            resized[row * cols + col] = value;
            work += 1;
            if work == 1024 {
                checkpoint_table(cooperation, selected, work).await?;
                work = 0;
            }
        }
        checkpoint_table(cooperation, selected, work).await?;
        Ok(Self {
            rows,
            cols,
            cells: resized,
        })
    }
}

async fn checkpoint_table(
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    work: usize,
) -> BuiltinResult<()> {
    cooperation.completed(work).await?;
    selected.check().map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: captured filesystem authority was revoked",
            source,
        )
    })
}

#[cfg(test)]
#[path = "overlay/tests.rs"]
mod tests;
