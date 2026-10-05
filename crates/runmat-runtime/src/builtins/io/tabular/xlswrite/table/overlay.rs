//! Existing-sheet resizing and cell overlay retain the caller's cooperation owner.

use super::*;

impl XlsTable {
    pub(in crate::builtins::io::tabular::xlswrite) async fn overlay(
        &mut self,
        start: RangeStart,
        incoming: &XlsTable,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<()> {
        let required_rows = start.row.checked_add(incoming.rows).ok_or_else(|| {
            xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range row overflow")
        })?;
        let required_cols = start.col.checked_add(incoming.cols).ok_or_else(|| {
            xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range column overflow")
        })?;
        checked_cell_count(required_rows, required_cols, "worksheet")?;
        if required_rows > self.rows || required_cols > self.cols {
            self.resize(
                required_rows.max(self.rows),
                required_cols.max(self.cols),
                cooperation,
                selected,
            )
            .await?;
        }
        let mut work = 0;
        for row in 0..incoming.rows {
            for col in 0..incoming.cols {
                let dest = (start.row + row) * self.cols + start.col + col;
                self.cells[dest] = incoming.cells[row * incoming.cols + col].clone();
                work += 1;
                if work == 1024 {
                    checkpoint_table(cooperation, selected, work).await?;
                    work = 0;
                }
            }
        }
        checkpoint_table(cooperation, selected, work).await?;
        Ok(())
    }

    async fn resize(
        &mut self,
        rows: usize,
        cols: usize,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<()> {
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
        let mut work = 0;
        for row in 0..self.rows {
            for col in 0..self.cols {
                resized[row * cols + col] = self.cells[row * self.cols + col].clone();
                work += 1;
                if work == 1024 {
                    checkpoint_table(cooperation, selected, work).await?;
                    work = 0;
                }
            }
        }
        checkpoint_table(cooperation, selected, work).await?;
        self.rows = rows;
        self.cols = cols;
        self.cells = resized;
        Ok(())
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
