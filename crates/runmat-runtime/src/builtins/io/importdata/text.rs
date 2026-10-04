use super::*;
use runmat_value::{CellArray, StructValue, Tensor};

pub(super) struct ImportTextResult {
    pub(super) value: Value,
    pub(super) delimiter: String,
    pub(super) header_lines: usize,
}

#[derive(Debug, Clone)]
struct ImportedText {
    data: Vec<Vec<f64>>,
    textdata: Vec<Vec<String>>,
    colheaders: Vec<String>,
    rowheaders: Vec<String>,
}

pub(super) fn import_text_data(
    text: &str,
    delimiter: Option<&str>,
    header_lines: Option<usize>,
) -> BuiltinResult<ImportTextResult> {
    let lines: Vec<&str> = text.lines().collect();
    let nonempty: Vec<(usize, &str)> = lines
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .collect();
    if nonempty.is_empty() {
        return Ok(ImportTextResult {
            value: Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).map_err(|err| {
                importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}"))
            })?),
            delimiter: delimiter.unwrap_or_default().to_string(),
            header_lines: header_lines.unwrap_or(0),
        });
    }

    let delimiter = delimiter
        .map(Delimiter::Explicit)
        .unwrap_or_else(|| detect_delimiter(nonempty.iter().map(|(_, line)| *line)));
    let delimiter_out = match &delimiter {
        Delimiter::Whitespace => " ".to_string(),
        Delimiter::Explicit(value) => (*value).to_string(),
    };
    let records: Vec<(usize, Vec<String>)> = nonempty
        .iter()
        .map(|(idx, line)| (*idx, split_record(line, &delimiter)))
        .collect();

    let data_start = header_lines.unwrap_or_else(|| infer_header_lines(&records));
    if data_start > records.len() {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: headerlinesIn exceeds number of non-empty lines",
        ));
    }

    let header_records: Vec<Vec<String>> = records[..data_start]
        .iter()
        .map(|(_, record)| record.clone())
        .collect();
    let data_records = &records[data_start..];

    let imported = parse_numeric_records(data_records, &header_records)?;
    let tensor = rows_to_tensor(&imported.data)?;
    if imported.textdata.is_empty()
        && imported.colheaders.is_empty()
        && imported.rowheaders.is_empty()
    {
        return Ok(ImportTextResult {
            value: Value::Tensor(tensor),
            delimiter: delimiter_out,
            header_lines: data_start,
        });
    }

    let mut out = StructValue::new();
    out.insert("data", Value::Tensor(tensor));
    if !imported.textdata.is_empty() {
        out.insert("textdata", cell_from_rows(&imported.textdata)?);
    }
    if !imported.colheaders.is_empty() {
        out.insert("colheaders", cell_from_row(&imported.colheaders)?);
    }
    if !imported.rowheaders.is_empty() {
        out.insert("rowheaders", cell_from_col(&imported.rowheaders)?);
    }
    Ok(ImportTextResult {
        value: Value::Struct(out),
        delimiter: delimiter_out,
        header_lines: data_start,
    })
}

fn parse_numeric_records(
    data_records: &[(usize, Vec<String>)],
    header_records: &[Vec<String>],
) -> BuiltinResult<ImportedText> {
    if data_records.is_empty() {
        return Ok(ImportedText {
            data: Vec::new(),
            textdata: header_records.to_vec(),
            colheaders: header_records.last().cloned().unwrap_or_default(),
            rowheaders: Vec::new(),
        });
    }

    let first = &data_records[0].1;
    let row_header_cols = infer_row_header_cols(data_records);
    let numeric_cols = first.len().saturating_sub(row_header_cols);
    if numeric_cols == 0 {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_PARSE,
            "importdata: no numeric columns found",
        ));
    }

    let mut rows = Vec::with_capacity(data_records.len());
    let mut rowheaders = Vec::new();
    for (line_idx, record) in data_records {
        let expected_cols = row_header_cols + numeric_cols;
        if record.len() != expected_cols {
            return Err(importdata_error_with(
                &IMPORTDATA_ERROR_PARSE,
                format!(
                    "importdata: row {} has {} columns, expected {}",
                    line_idx + 1,
                    record.len(),
                    expected_cols
                ),
            ));
        }
        if row_header_cols > 0 {
            rowheaders.push(record[..row_header_cols].join(" "));
        }
        let mut row = Vec::with_capacity(numeric_cols);
        for (col, token) in record[row_header_cols..row_header_cols + numeric_cols]
            .iter()
            .enumerate()
        {
            row.push(parse_numeric_token(token).ok_or_else(|| {
                importdata_error_with(
                    &IMPORTDATA_ERROR_PARSE,
                    format!(
                        "importdata: nonnumeric token '{}' at row {}, column {}",
                        token,
                        line_idx + 1,
                        row_header_cols + col + 1
                    ),
                )
            })?);
        }
        rows.push(row);
    }

    let mut colheaders = Vec::new();
    if let Some(last_header) = header_records.last() {
        if last_header.len() >= row_header_cols + numeric_cols {
            colheaders = last_header[row_header_cols..row_header_cols + numeric_cols].to_vec();
        } else if last_header.len() == numeric_cols {
            colheaders = last_header.clone();
        }
    }

    Ok(ImportedText {
        data: rows,
        textdata: header_records.to_vec(),
        colheaders,
        rowheaders,
    })
}

fn infer_row_header_cols(records: &[(usize, Vec<String>)]) -> usize {
    let Some(first) = records.first() else {
        return 0;
    };
    if first.1.len() < 2 || parse_numeric_token(&first.1[0]).is_some() {
        return 0;
    }
    if records.iter().all(|(_, row)| {
        row.len() == first.1.len()
            && parse_numeric_token(&row[0]).is_none()
            && row[1..]
                .iter()
                .all(|token| parse_numeric_token(token).is_some())
    }) {
        1
    } else {
        0
    }
}

fn infer_header_lines(records: &[(usize, Vec<String>)]) -> usize {
    records
        .iter()
        .position(|(_, row)| is_numeric_data_row(row))
        .unwrap_or(records.len())
}

fn is_numeric_data_row(row: &[String]) -> bool {
    if row.is_empty() {
        return false;
    }
    if row.iter().all(|token| parse_numeric_token(token).is_some()) {
        return true;
    }
    row.len() > 1
        && parse_numeric_token(&row[0]).is_none()
        && row[1..]
            .iter()
            .all(|token| parse_numeric_token(token).is_some())
}

fn rows_to_tensor(rows: &[Vec<f64>]) -> BuiltinResult<Tensor> {
    let row_count = rows.len();
    let col_count = rows.first().map(|row| row.len()).unwrap_or(0);
    if rows.iter().any(|row| row.len() != col_count) {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_PARSE,
            "importdata: numeric rows have inconsistent column counts",
        ));
    }
    let mut data = Vec::with_capacity(row_count * col_count);
    for col in 0..col_count {
        for row in rows {
            data.push(row[col]);
        }
    }
    Tensor::new(data, vec![row_count, col_count])
        .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Delimiter<'a> {
    Whitespace,
    Explicit(&'a str),
}

fn detect_delimiter<'a>(lines: impl Iterator<Item = &'a str>) -> Delimiter<'static> {
    let candidates = [",", "\t", ";", "|"];
    let sample: Vec<&str> = lines.take(12).collect();
    let mut best: Option<(&str, usize, usize)> = None;
    for candidate in candidates {
        let counts: Vec<usize> = sample
            .iter()
            .map(|line| split_record(line, &Delimiter::Explicit(candidate)).len())
            .filter(|count| *count > 1)
            .collect();
        if counts.is_empty() {
            continue;
        }
        let consistent = counts.iter().filter(|count| **count == counts[0]).count();
        let score = (consistent, counts[0]);
        if best
            .map(|(_, best_consistent, best_cols)| score > (best_consistent, best_cols))
            .unwrap_or(true)
        {
            best = Some((candidate, consistent, counts[0]));
        }
    }
    best.map(|(candidate, _, _)| Delimiter::Explicit(candidate))
        .unwrap_or(Delimiter::Whitespace)
}

fn split_record(line: &str, delimiter: &Delimiter<'_>) -> Vec<String> {
    match delimiter {
        Delimiter::Whitespace => line
            .split_whitespace()
            .map(|token| unquote(token.trim()))
            .filter(|token| !token.is_empty())
            .collect(),
        Delimiter::Explicit(delimiter) => split_explicit(line, delimiter),
    }
}

fn split_explicit(line: &str, delimiter: &str) -> Vec<String> {
    if delimiter.is_empty() {
        return vec![line.trim().to_string()];
    }
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut idx = 0usize;
    while idx < line.len() {
        let Some(ch) = line[idx..].chars().next() else {
            break;
        };
        if ch == '"' {
            if in_quotes && line[idx + ch.len_utf8()..].starts_with('"') {
                current.push('"');
                idx += ch.len_utf8() * 2;
                continue;
            }
            in_quotes = !in_quotes;
            idx += ch.len_utf8();
            continue;
        }
        if !in_quotes && line[idx..].starts_with(delimiter) {
            fields.push(unquote(current.trim()));
            current.clear();
            idx += delimiter.len();
            continue;
        }
        current.push(ch);
        idx += ch.len_utf8();
    }
    fields.push(unquote(current.trim()));
    fields
}

fn unquote(token: &str) -> String {
    let trimmed = token.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        trimmed[1..trimmed.len() - 1].replace("\"\"", "\"")
    } else {
        trimmed.to_string()
    }
}

fn parse_numeric_token(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Some(f64::NAN);
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "nan" => Some(f64::NAN),
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => trimmed.parse::<f64>().ok(),
    }
}

fn cell_from_rows(rows: &[Vec<String>]) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let col_count = rows.iter().map(|row| row.len()).max().unwrap_or(0);
    let mut values = Vec::with_capacity(row_count * col_count);
    for row in rows {
        for col in 0..col_count {
            values.push(Value::String(row.get(col).cloned().unwrap_or_default()));
        }
    }
    CellArray::new(values, row_count, col_count)
        .map(Value::Cell)
        .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

fn cell_from_row(values: &[String]) -> BuiltinResult<Value> {
    CellArray::new(
        values.iter().cloned().map(Value::String).collect(),
        1,
        values.len(),
    )
    .map(Value::Cell)
    .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

fn cell_from_col(values: &[String]) -> BuiltinResult<Value> {
    CellArray::new(
        values.iter().cloned().map(Value::String).collect(),
        values.len(),
        1,
    )
    .map(Value::Cell)
    .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}
