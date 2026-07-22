use super::*;

#[derive(Clone, Debug)]
pub(in crate::builtins::table) enum ImportCell {
    Empty,
    Text(String),
    Number(f64),
    Logical(bool),
    DateTime(f64),
    Error(String),
}

impl ImportCell {
    pub(in crate::builtins::table) fn from_text(text: String) -> Self {
        if text.trim().is_empty() {
            Self::Empty
        } else {
            Self::Text(text)
        }
    }

    pub(in crate::builtins::table) fn display_text(&self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Text(text) => text.clone(),
            Self::Number(value) => format_key_number(*value),
            Self::Logical(value) => value.to_string(),
            Self::DateTime(serial) => format_key_number(*serial),
            Self::Error(text) => text.clone(),
        }
    }

    pub(in crate::builtins::table) fn is_missing(&self, options: &ReadTableOptions) -> bool {
        match self {
            Self::Empty => true,
            Self::Text(text) => options.is_missing(text),
            _ => false,
        }
    }

    pub(in crate::builtins::table) fn is_likely_data_token(
        &self,
        options: &ReadTableOptions,
    ) -> bool {
        match self {
            Self::Number(_) | Self::Logical(_) | Self::DateTime(_) => true,
            Self::Empty => false,
            Self::Text(text) => {
                let token = unquote(text.trim()).trim();
                options.is_missing(token)
                    || parse_numeric(token).is_some()
                    || parse_logical(token).is_some()
                    || parse_iso_datetime_to_datenum(token).is_some()
            }
            Self::Error(_) => true,
        }
    }
}

pub(in crate::builtins::table) fn spreadsheet_cell_to_import(cell: &SpreadsheetData) -> ImportCell {
    match cell {
        SpreadsheetData::Empty => ImportCell::Empty,
        SpreadsheetData::Int(value) => ImportCell::Number(*value as f64),
        SpreadsheetData::Float(value) => ImportCell::Number(*value),
        SpreadsheetData::String(text) => ImportCell::Text(text.clone()),
        SpreadsheetData::Bool(value) => ImportCell::Logical(*value),
        SpreadsheetData::DateTime(value) => value
            .as_datetime()
            .map(crate::builtins::datetime::datenum_from_naive)
            .map(ImportCell::DateTime)
            .unwrap_or_else(|| ImportCell::Number(value.as_f64())),
        SpreadsheetData::DateTimeIso(text) => parse_iso_datetime_to_datenum(text)
            .map(ImportCell::DateTime)
            .unwrap_or_else(|| ImportCell::Text(text.clone())),
        SpreadsheetData::DurationIso(text) => ImportCell::Text(text.clone()),
        SpreadsheetData::Error(err) => ImportCell::Error(err.to_string()),
    }
}

pub(in crate::builtins::table) fn spreadsheet_range_to_rows(
    range: &calamine::Range<SpreadsheetData>,
    options: &ReadTableOptions,
) -> BuiltinResult<Vec<Vec<ImportCell>>> {
    if range.is_empty() {
        return Ok(Vec::new());
    }
    let Some((range_start_row, range_start_col)) = range.start() else {
        return Ok(Vec::new());
    };
    let Some((range_end_row, range_end_col)) = range.end() else {
        return Ok(Vec::new());
    };
    let start_row = options
        .range
        .map(|spec| checked_u32(spec.start_row, "Range row"))
        .transpose()?
        .unwrap_or(range_start_row);
    let start_col = options
        .range
        .map(|spec| checked_u32(spec.start_col, "Range column"))
        .transpose()?
        .unwrap_or(range_start_col);
    let end_row = options
        .range
        .and_then(|spec| spec.end_row)
        .map(|row| checked_u32(row, "Range row"))
        .transpose()?
        .unwrap_or(range_end_row);
    let end_col = options
        .range
        .and_then(|spec| spec.end_col)
        .map(|col| checked_u32(col, "Range column"))
        .transpose()?
        .unwrap_or(range_end_col);
    if start_row > end_row || start_col > end_col {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for row_idx in start_row..=end_row {
        let mut row = Vec::new();
        for col_idx in start_col..=end_col {
            row.push(
                range
                    .get_value((row_idx, col_idx))
                    .map(spreadsheet_cell_to_import)
                    .unwrap_or(ImportCell::Empty),
            );
        }
        if matches!(options.empty_line_rule, EmptyLineRule::Skip)
            && row.iter().all(|cell| cell.is_missing(options))
        {
            continue;
        }
        rows.push(row);
    }
    if options.num_header_lines > 0 {
        Ok(rows.into_iter().skip(options.num_header_lines).collect())
    } else {
        Ok(rows)
    }
}

pub(in crate::builtins::table) fn checked_u32(value: usize, context: &str) -> BuiltinResult<u32> {
    u32::try_from(value).map_err(|_| invalid_argument(format!("readtable: {context} overflow")))
}
