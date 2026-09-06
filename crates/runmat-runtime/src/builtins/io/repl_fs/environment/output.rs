use runmat_value::{CellArray, CharArray, LogicalArray, StringArray, Value};

use super::names::{CellNameKind, EnvironmentNames};

impl EnvironmentNames {
    pub(super) fn logical_output(&self, values: Vec<bool>) -> Result<Value, String> {
        let shape = match self {
            Self::CharacterRow(_) | Self::StringScalar(_) => {
                return values
                    .first()
                    .copied()
                    .map(Value::Bool)
                    .ok_or_else(|| "missing scalar environment result".to_string());
            }
            Self::Strings { shape, .. } => shape.clone(),
            Self::Cells { rows, cols, .. } => vec![*rows, *cols],
            Self::CharacterMatrix { rows, .. } => vec![*rows, 1],
        };
        LogicalArray::new(values.into_iter().map(u8::from).collect(), shape)
            .map(Value::LogicalArray)
    }

    pub(super) fn text_output(&self, values: Vec<String>) -> Result<Value, String> {
        match self {
            Self::CharacterRow(_) | Self::StringScalar(_) => Ok(Value::CharArray(
                CharArray::new_row(values.first().map_or("", String::as_str)),
            )),
            Self::Strings { shape, .. } => {
                StringArray::new(values, shape.clone()).map(Value::StringArray)
            }
            Self::Cells {
                kinds, rows, cols, ..
            } => cell_text_output(values, kinds, *rows, *cols),
            Self::CharacterMatrix { rows, .. } => {
                character_matrix(values, *rows).map(Value::CharArray)
            }
        }
    }
}

fn cell_text_output(
    values: Vec<String>,
    kinds: &[CellNameKind],
    rows: usize,
    cols: usize,
) -> Result<Value, String> {
    let data = values
        .into_iter()
        .zip(kinds)
        .map(|(value, kind)| match kind {
            CellNameKind::Character => Value::CharArray(CharArray::new_row(&value)),
            CellNameKind::String => Value::String(value),
        })
        .collect();
    CellArray::new(data, rows, cols).map(Value::Cell)
}

fn character_matrix(rows: Vec<String>, row_count: usize) -> Result<CharArray, String> {
    let width = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(row_count.saturating_mul(width));
    for row in rows {
        let mut chars = row.chars();
        for _ in 0..width {
            data.push(chars.next().unwrap_or(' '));
        }
    }
    CharArray::new(data, row_count, width)
}
