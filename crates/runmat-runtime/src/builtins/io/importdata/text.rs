use super::*;
use runmat_value::{CellArray, Tensor};

#[path = "text/lines.rs"]
mod lines;
use lines::collect_nonempty_lines;

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

pub(super) async fn import_text_data(
    text: &str,
    delimiter: Option<&str>,
    header_lines: Option<usize>,
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<ImportTextResult> {
    let nonempty = collect_nonempty_lines(text, cooperation).await?;
    if nonempty.is_empty() {
        return Ok(ImportTextResult {
            value: Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).map_err(|err| {
                importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}"))
            })?),
            delimiter: delimiter.unwrap_or_default().to_string(),
            header_lines: header_lines.unwrap_or(0),
        });
    }

    let delimiter = match delimiter {
        Some(value) => Delimiter::Explicit(value),
        None => detect_delimiter(nonempty.iter().map(|(_, line)| *line), cooperation).await?,
    };
    let delimiter_out = match &delimiter {
        Delimiter::Whitespace => " ".to_string(),
        Delimiter::Explicit(value) => (*value).to_string(),
    };
    let mut records = Vec::with_capacity(nonempty.len());
    for (idx, line) in &nonempty {
        records.push((*idx, split_record(line, &delimiter, cooperation).await?));
    }

    let data_start = match header_lines {
        Some(value) => value,
        None => infer_header_lines(&records, cooperation).await?,
    };
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

    let imported = parse_numeric_records(data_records, &header_records, cooperation).await?;
    let tensor = rows_to_tensor(&imported.data, cooperation).await?;
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

    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    out.insert("data", Value::Tensor(tensor));
    if !imported.textdata.is_empty() {
        out.insert(
            "textdata",
            cell_from_rows(&imported.textdata, cooperation).await?,
        );
    }
    if !imported.colheaders.is_empty() {
        out.insert(
            "colheaders",
            cell_from_row(&imported.colheaders, cooperation).await?,
        );
    }
    if !imported.rowheaders.is_empty() {
        out.insert(
            "rowheaders",
            cell_from_col(&imported.rowheaders, cooperation).await?,
        );
    }
    Ok(ImportTextResult {
        value: Value::Struct(out.finish()),
        delimiter: delimiter_out,
        header_lines: data_start,
    })
}

async fn parse_numeric_records(
    data_records: &[(usize, Vec<String>)],
    header_records: &[Vec<String>],
    cooperation: &mut cooperation::ImportCooperation,
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
    let row_header_cols = infer_row_header_cols(data_records, cooperation).await?;
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
            cooperation.completed(token.len()).await?;
        }
        rows.push(row);
        cooperation.completed(record.len()).await?;
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

async fn infer_row_header_cols(
    records: &[(usize, Vec<String>)],
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<usize> {
    let Some(first) = records.first() else {
        return Ok(0);
    };
    if first.1.len() < 2 || parse_numeric_token(&first.1[0]).is_some() {
        return Ok(0);
    }
    for (_, row) in records {
        if row.len() != first.1.len() || parse_numeric_token(&row[0]).is_some() {
            return Ok(0);
        }
        for token in &row[1..] {
            if parse_numeric_token(token).is_none() {
                return Ok(0);
            }
            cooperation.completed(token.len()).await?;
        }
    }
    Ok(1)
}

async fn infer_header_lines(
    records: &[(usize, Vec<String>)],
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<usize> {
    for (index, (_, row)) in records.iter().enumerate() {
        if is_numeric_data_row(row, cooperation).await? {
            return Ok(index);
        }
        cooperation
            .completed(row.iter().map(String::len).sum())
            .await?;
    }
    Ok(records.len())
}

async fn is_numeric_data_row(
    row: &[String],
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<bool> {
    if row.is_empty() {
        return Ok(false);
    }
    let mut all_numeric = true;
    for token in row {
        all_numeric &= parse_numeric_token(token).is_some();
        cooperation.completed(token.len()).await?;
    }
    if all_numeric {
        return Ok(true);
    }
    if row.len() <= 1 || parse_numeric_token(&row[0]).is_some() {
        return Ok(false);
    }
    for token in &row[1..] {
        if parse_numeric_token(token).is_none() {
            return Ok(false);
        }
        cooperation.completed(token.len()).await?;
    }
    Ok(true)
}

async fn rows_to_tensor(
    rows: &[Vec<f64>],
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<Tensor> {
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
        cooperation.completed(row_count).await?;
    }
    Tensor::new(data, vec![row_count, col_count])
        .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Delimiter<'a> {
    Whitespace,
    Explicit(&'a str),
}

async fn detect_delimiter<'a>(
    lines: impl Iterator<Item = &'a str>,
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<Delimiter<'static>> {
    let candidates = [",", "\t", ";", "|"];
    let sample: Vec<&str> = lines.take(12).collect();
    let mut best: Option<(&str, usize, usize)> = None;
    for candidate in candidates {
        let mut counts = Vec::new();
        for line in &sample {
            let count = split_record(line, &Delimiter::Explicit(candidate), cooperation)
                .await?
                .len();
            if count > 1 {
                counts.push(count);
            }
        }
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
    Ok(best
        .map(|(candidate, _, _)| Delimiter::Explicit(candidate))
        .unwrap_or(Delimiter::Whitespace))
}

async fn split_record(
    line: &str,
    delimiter: &Delimiter<'_>,
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<Vec<String>> {
    match delimiter {
        Delimiter::Whitespace => {
            let mut fields = Vec::new();
            let mut current = String::new();
            for ch in line.chars() {
                if ch.is_whitespace() {
                    if !current.is_empty() {
                        let token = unquote(current.trim());
                        if !token.is_empty() {
                            fields.push(token);
                        }
                        current.clear();
                    }
                } else {
                    current.push(ch);
                }
                cooperation.completed(ch.len_utf8()).await?;
            }
            if !current.is_empty() {
                let token = unquote(current.trim());
                if !token.is_empty() {
                    fields.push(token);
                }
            }
            Ok(fields)
        }
        Delimiter::Explicit(delimiter) => split_explicit(line, delimiter, cooperation).await,
    }
}

async fn split_explicit(
    line: &str,
    delimiter: &str,
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<Vec<String>> {
    if delimiter.is_empty() {
        for chunk in line.as_bytes().chunks(4096) {
            cooperation.completed(chunk.len()).await?;
        }
        return Ok(vec![line.trim().to_string()]);
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
                cooperation.completed(ch.len_utf8() * 2).await?;
                continue;
            }
            in_quotes = !in_quotes;
            idx += ch.len_utf8();
            cooperation.completed(ch.len_utf8()).await?;
            continue;
        }
        if !in_quotes && line[idx..].starts_with(delimiter) {
            fields.push(unquote(current.trim()));
            current.clear();
            idx += delimiter.len();
            cooperation.completed(delimiter.len()).await?;
            continue;
        }
        current.push(ch);
        idx += ch.len_utf8();
        cooperation.completed(ch.len_utf8()).await?;
    }
    fields.push(unquote(current.trim()));
    Ok(fields)
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

async fn cell_from_rows(
    rows: &[Vec<String>],
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let col_count = rows.iter().map(|row| row.len()).max().unwrap_or(0);
    let mut values = Vec::with_capacity(row_count * col_count);
    for row in rows {
        for col in 0..col_count {
            values.push(Value::string(row.get(col).cloned().unwrap_or_default()));
        }
        cooperation.completed(col_count).await?;
    }
    CellArray::new(values, row_count, col_count)
        .map(Value::Cell)
        .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

async fn cell_from_row(
    values: &[String],
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<Value> {
    let mut entries = Vec::with_capacity(values.len());
    for value in values {
        entries.push(Value::string(value.clone()));
        cooperation.completed(value.len()).await?;
    }
    CellArray::new(entries, 1, values.len())
        .map(Value::Cell)
        .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

async fn cell_from_col(
    values: &[String],
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<Value> {
    let mut entries = Vec::with_capacity(values.len());
    for value in values {
        entries.push(Value::string(value.clone()));
        cooperation.completed(value.len()).await?;
    }
    CellArray::new(entries, values.len(), 1)
        .map(Value::Cell)
        .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}
