use super::*;
use calamine::Data as SpreadsheetData;

pub(super) struct XlsReadResult {
    num: Option<Value>,
    txt: Option<Value>,
    raw: Option<Value>,
}

#[derive(Clone, Copy)]
pub(super) struct XlsReadOutputs {
    num: bool,
    txt: bool,
    raw: bool,
    implicit: bool,
}

impl XlsReadOutputs {
    pub(super) fn requested() -> BuiltinResult<Self> {
        match crate::output_count::current_output_count() {
            Some(0) => Ok(Self {
                num: false,
                txt: false,
                raw: false,
                implicit: false,
            }),
            Some(1) => Ok(Self {
                num: true,
                txt: false,
                raw: false,
                implicit: false,
            }),
            None => Ok(Self {
                num: true,
                txt: false,
                raw: false,
                implicit: true,
            }),
            Some(2) => Ok(Self {
                num: true,
                txt: true,
                raw: false,
                implicit: false,
            }),
            Some(3) => Ok(Self {
                num: true,
                txt: true,
                raw: true,
                implicit: false,
            }),
            Some(n) => Err(xlsread_error_with(
                &XLSREAD_ERROR_OUTPUT_COUNT,
                format!("xlsread: expected at most 3 outputs, got {n}"),
            )),
        }
    }
}

impl XlsReadResult {
    pub(super) fn from_cells(
        rows: Vec<Vec<SpreadsheetData>>,
        outputs: XlsReadOutputs,
    ) -> BuiltinResult<Self> {
        let num = if outputs.num {
            Some(build_numeric_output(&rows)?)
        } else {
            None
        };
        let txt = if outputs.txt {
            Some(build_text_output(&rows)?)
        } else {
            None
        };
        let raw = if outputs.raw {
            Some(build_raw_output(&rows)?)
        } else {
            None
        };
        Ok(Self { num, txt, raw })
    }

    pub(super) fn into_requested_value(self, outputs: XlsReadOutputs) -> BuiltinResult<Value> {
        if outputs.implicit {
            return Ok(self.num.expect("num output"));
        }
        match (outputs.num, outputs.txt, outputs.raw) {
            (false, false, false) => Ok(Value::OutputList(Vec::new())),
            (true, false, false) => Ok(Value::OutputList(vec![self.num.expect("num output")])),
            (true, true, false) => Ok(Value::OutputList(vec![
                self.num.expect("num output"),
                self.txt.expect("txt output"),
            ])),
            (true, true, true) => Ok(Value::OutputList(vec![
                self.num.expect("num output"),
                self.txt.expect("txt output"),
                self.raw.expect("raw output"),
            ])),
            _ => Err(xlsread_error_with(
                &XLSREAD_ERROR_OUTPUT_COUNT,
                "xlsread: invalid requested output state",
            )),
        }
    }
}

fn build_numeric_output(rows: &[Vec<SpreadsheetData>]) -> BuiltinResult<Value> {
    let Some((first_row, last_row, first_col, last_col)) = numeric_bounds(rows) else {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).map_err(|err| {
            xlsread_error_with(&XLSREAD_ERROR_TENSOR_BUILD, format!("xlsread: {err}"))
        })?;
        return Ok(Value::Tensor(tensor));
    };
    let row_count = last_row - first_row + 1;
    let col_count = last_col - first_col + 1;
    let data_len = row_count.checked_mul(col_count).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_TENSOR_BUILD,
            "xlsread: numeric output is too large",
        )
    })?;
    let mut data = vec![f64::NAN; data_len];
    for out_row in 0..row_count {
        for out_col in 0..col_count {
            let source_row = first_row + out_row;
            let source_col = first_col + out_col;
            let value = rows
                .get(source_row)
                .and_then(|row| row.get(source_col))
                .and_then(cell_to_numeric)
                .unwrap_or(f64::NAN);
            data[out_row + out_col * row_count] = value;
        }
    }
    let tensor = Tensor::new(data, vec![row_count, col_count]).map_err(|err| {
        xlsread_error_with(&XLSREAD_ERROR_TENSOR_BUILD, format!("xlsread: {err}"))
    })?;
    Ok(Value::Tensor(tensor))
}

fn numeric_bounds(rows: &[Vec<SpreadsheetData>]) -> Option<(usize, usize, usize, usize)> {
    let mut first_row = None;
    let mut last_row = None;
    let mut first_col = None;
    let mut last_col = None;
    for (row_idx, row) in rows.iter().enumerate() {
        for (col_idx, cell) in row.iter().enumerate() {
            if cell_to_numeric(cell).is_some() {
                first_row = Some(first_row.map_or(row_idx, |current: usize| current.min(row_idx)));
                last_row = Some(last_row.map_or(row_idx, |current: usize| current.max(row_idx)));
                first_col = Some(first_col.map_or(col_idx, |current: usize| current.min(col_idx)));
                last_col = Some(last_col.map_or(col_idx, |current: usize| current.max(col_idx)));
            }
        }
    }
    match (first_row, last_row, first_col, last_col) {
        (Some(first_row), Some(last_row), Some(first_col), Some(last_col)) => {
            Some((first_row, last_row, first_col, last_col))
        }
        _ => None,
    }
}

fn build_text_output(rows: &[Vec<SpreadsheetData>]) -> BuiltinResult<Value> {
    let (row_count, col_count) = rectangular_shape(rows);
    let mut values = Vec::with_capacity(row_count * col_count);
    for row_idx in 0..row_count {
        for col_idx in 0..col_count {
            let text = rows
                .get(row_idx)
                .and_then(|row| row.get(col_idx))
                .and_then(cell_to_text)
                .unwrap_or_default();
            values.push(Value::from(text));
        }
    }
    crate::make_cell(values, row_count, col_count)
        .map_err(|err| xlsread_error_with(&XLSREAD_ERROR_TENSOR_BUILD, format!("xlsread: {err}")))
}

fn build_raw_output(rows: &[Vec<SpreadsheetData>]) -> BuiltinResult<Value> {
    let (row_count, col_count) = rectangular_shape(rows);
    let mut values = Vec::with_capacity(row_count * col_count);
    for row_idx in 0..row_count {
        for col_idx in 0..col_count {
            let value = rows
                .get(row_idx)
                .and_then(|row| row.get(col_idx))
                .map(cell_to_raw)
                .unwrap_or(Value::Num(f64::NAN));
            values.push(value);
        }
    }
    crate::make_cell(values, row_count, col_count)
        .map_err(|err| xlsread_error_with(&XLSREAD_ERROR_TENSOR_BUILD, format!("xlsread: {err}")))
}

fn rectangular_shape(rows: &[Vec<SpreadsheetData>]) -> (usize, usize) {
    let row_count = rows.len();
    let col_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    (row_count, col_count)
}

fn cell_to_numeric(cell: &SpreadsheetData) -> Option<f64> {
    match cell {
        SpreadsheetData::Int(value) => Some(*value as f64),
        SpreadsheetData::Float(value) => Some(*value),
        SpreadsheetData::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        SpreadsheetData::DateTime(value) => Some(value.as_f64()),
        _ => None,
    }
}

fn cell_to_text(cell: &SpreadsheetData) -> Option<String> {
    match cell {
        SpreadsheetData::String(text)
        | SpreadsheetData::DateTimeIso(text)
        | SpreadsheetData::DurationIso(text) => Some(text.clone()),
        SpreadsheetData::Error(err) => Some(err.to_string()),
        _ => None,
    }
}

fn cell_to_raw(cell: &SpreadsheetData) -> Value {
    match cell {
        SpreadsheetData::Int(value) => Value::Num(*value as f64),
        SpreadsheetData::Float(value) => Value::Num(*value),
        SpreadsheetData::Bool(value) => Value::Bool(*value),
        SpreadsheetData::DateTime(value) => Value::Num(value.as_f64()),
        SpreadsheetData::String(text)
        | SpreadsheetData::DateTimeIso(text)
        | SpreadsheetData::DurationIso(text) => Value::from(text.clone()),
        SpreadsheetData::Error(err) => Value::from(err.to_string()),
        SpreadsheetData::Empty => Value::Num(f64::NAN),
    }
}
