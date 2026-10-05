//! Projection of an already-parsed Calamine sheet into the owned XLS table.

use super::*;

impl XlsTable {
    pub(in crate::builtins::io::tabular::xlswrite) async fn from_calamine_range_cooperative(
        range: &calamine::Range<SpreadsheetData>,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        let Some((start_row, start_col)) = range.start() else {
            return Ok(Self::empty());
        };
        let Some((end_row, end_col)) = range.end() else {
            return Ok(Self::empty());
        };
        let rows = end_row
            .checked_add(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                xlswrite_error_with(
                    &XLSWRITE_ERROR_DATA,
                    "xlswrite: existing sheet row count is too large",
                )
            })?;
        let cols = end_col
            .checked_add(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                xlswrite_error_with(
                    &XLSWRITE_ERROR_DATA,
                    "xlswrite: existing sheet column count is too large",
                )
            })?;
        let count = checked_cell_count(rows, cols, "existing worksheet")?;
        let mut cells = Vec::new();
        cells.try_reserve_exact(count).map_err(|source| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: existing worksheet allocation failed",
                source,
            )
        })?;
        while cells.len() < count {
            let chunk = (count - cells.len()).min(1024);
            cells.resize(cells.len() + chunk, CellValue::Empty);
            checkpoint_input(cooperation, selected, chunk).await?;
        }
        let mut work = 0;
        for row in start_row..=end_row {
            for col in start_col..=end_col {
                let row_usize = row as usize;
                let col_usize = col as usize;
                cells[row_usize * cols + col_usize] = spreadsheet_cell_to_cell_value(
                    range
                        .get_value((row, col))
                        .unwrap_or(&SpreadsheetData::Empty),
                    cooperation,
                    selected,
                )
                .await?;
                work += 1;
                if work == 1024 {
                    checkpoint_input(cooperation, selected, work).await?;
                    work = 0;
                }
            }
        }
        checkpoint_input(cooperation, selected, work).await?;
        Ok(Self { rows, cols, cells })
    }
}

async fn spreadsheet_cell_to_cell_value(
    cell: &SpreadsheetData,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
) -> BuiltinResult<CellValue> {
    Ok(match cell {
        SpreadsheetData::Empty => CellValue::Empty,
        SpreadsheetData::Int(value) => CellValue::Number(*value as f64),
        SpreadsheetData::Float(value) => CellValue::Number(*value),
        SpreadsheetData::String(value) => {
            CellValue::Text(copy_cell_text(value, cooperation, selected).await?)
        }
        SpreadsheetData::Bool(value) => CellValue::Boolean(*value),
        SpreadsheetData::DateTime(value) => CellValue::Number(value.as_f64()),
        SpreadsheetData::DateTimeIso(value) | SpreadsheetData::DurationIso(value) => {
            CellValue::Text(copy_cell_text(value, cooperation, selected).await?)
        }
        SpreadsheetData::Error(value) => CellValue::Text(value.to_string()),
    })
}

async fn copy_cell_text(
    text: &str,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
) -> BuiltinResult<String> {
    let mut copy = String::new();
    copy.try_reserve(text.len()).map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: existing worksheet text allocation failed",
            source,
        )
    })?;
    let mut work = 0;
    for character in text.chars() {
        copy.push(character);
        work += character.len_utf8();
        if work >= 4096 {
            checkpoint_input(cooperation, selected, work).await?;
            work = 0;
        }
    }
    checkpoint_input(cooperation, selected, work).await?;
    Ok(copy)
}
