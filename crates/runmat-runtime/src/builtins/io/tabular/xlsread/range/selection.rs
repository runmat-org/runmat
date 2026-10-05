//! Trim implicit worksheet selection without changing explicit range geometry.

use super::*;

pub(super) async fn trim_empty_edges(
    rows: Vec<Vec<SpreadsheetData>>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
) -> BuiltinResult<Vec<Vec<SpreadsheetData>>> {
    if rows.is_empty() {
        return Ok(rows);
    }
    let row_count = rows.len();
    let col_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut first_row = None;
    let mut last_row = None;
    let mut first_col = None;
    let mut last_col = None;

    for row in 0..row_count {
        for col in 0..col_count {
            let cell = rows
                .get(row)
                .and_then(|values| values.get(col))
                .unwrap_or(&SpreadsheetData::Empty);
            if !matches!(cell, SpreadsheetData::Empty) {
                first_row = Some(first_row.map_or(row, |current: usize| current.min(row)));
                last_row = Some(last_row.map_or(row, |current: usize| current.max(row)));
                first_col = Some(first_col.map_or(col, |current: usize| current.min(col)));
                last_col = Some(last_col.map_or(col, |current: usize| current.max(col)));
            }
        }
        checkpoint_cells(cooperation, selected, col_count).await?;
    }

    let (Some(first_row), Some(last_row), Some(first_col), Some(last_col)) =
        (first_row, last_row, first_col, last_col)
    else {
        return Ok(Vec::new());
    };
    let mut trimmed = Vec::new();
    for row in rows.iter().take(last_row + 1).skip(first_row) {
        let mut out = Vec::new();
        for col in first_col..=last_col {
            let cell = match row.get(col) {
                Some(cell) => clone_spreadsheet_cell(cell, cooperation, selected).await?,
                None => SpreadsheetData::Empty,
            };
            out.push(cell);
        }
        trimmed.push(out);
        checkpoint_cells(cooperation, selected, last_col - first_col + 1).await?;
    }
    Ok(trimmed)
}
