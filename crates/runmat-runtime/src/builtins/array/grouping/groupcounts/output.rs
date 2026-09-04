use runmat_value::{CellArray, CharArray, LogicalArray, StringArray, Tensor, Value};

use crate::builtins::array::grouping::keys::{GroupIndex, KeyAtom};
use crate::builtins::array::grouping::variables::{select_group_rows, GroupColumn};
use crate::builtins::table::{categorical_levels_for_observations, table_from_columns};
use crate::BuiltinResult;

use super::error;
use super::input::{GroupCountInput, OutputForm};

pub(super) fn build(input: &GroupCountInput, index: &GroupIndex) -> BuiltinResult<Value> {
    let counts = index
        .row_groups
        .iter()
        .map(|rows| rows.len() as f64)
        .collect::<Vec<_>>();
    let percentages = counts
        .iter()
        .map(|count| {
            if input.rows == 0 {
                0.0
            } else {
                count * 100.0 / input.rows as f64
            }
        })
        .collect::<Vec<_>>();
    let count_value = column(counts)?;
    let percent_value = column(percentages)?;
    let mut labels = label_values(&input.columns, index)?;

    match &input.output {
        OutputForm::Array => {
            let groups = match labels.len() {
                1 => labels
                    .pop()
                    .ok_or_else(|| error::internal("groupcounts: missing label column"))?,
                _ => Value::Cell(
                    CellArray::new(labels, 1, input.columns.len()).map_err(error::internal)?,
                ),
            };
            super::super::requested_outputs::finish(vec![count_value, groups, percent_value])
        }
        OutputForm::Table { names } => {
            let mut output_names = names.clone();
            output_names.extend(["GroupCount".into(), "Percent".into()]);
            let mut values = labels;
            values.extend([count_value, percent_value]);
            table_from_columns(output_names, values)
        }
    }
}

fn column(values: Vec<f64>) -> BuiltinResult<Value> {
    let rows = values.len();
    Tensor::new(values, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(error::internal)
}

fn label_values(columns: &[GroupColumn], index: &GroupIndex) -> BuiltinResult<Vec<Value>> {
    columns
        .iter()
        .enumerate()
        .map(|(position, column)| column_labels(column, position, index))
        .collect()
}

fn column_labels(
    column: &GroupColumn,
    position: usize,
    index: &GroupIndex,
) -> BuiltinResult<Value> {
    let atoms = index
        .keys
        .iter()
        .map(|key| {
            key.get(position)
                .cloned()
                .ok_or_else(|| error::internal("groupcounts: malformed group key"))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    match &column.value {
        Value::Object(value) if value.is_class(runmat_types::standard::CATEGORICAL) => {
            categorical_labels(value, atoms)
        }
        Value::LogicalArray(_) | Value::Bool(_) => logical_labels(atoms),
        Value::StringArray(_) | Value::String(_) => StringArray::new(
            atoms.into_iter().map(|atom| atom.label()).collect(),
            vec![index.keys.len(), 1],
        )
        .map(Value::StringArray)
        .map_err(error::internal),
        Value::Cell(_) => cellstr_labels(atoms),
        _ => select_observed_rows(column, &atoms),
    }
}

fn categorical_labels(
    prototype: &runmat_value::ObjectInstance,
    atoms: Vec<KeyAtom>,
) -> BuiltinResult<Value> {
    let labels = atoms
        .into_iter()
        .map(|atom| match atom {
            KeyAtom::Text(value) => Ok(Some(value)),
            KeyAtom::Missing => Ok(None),
            _ => Err(error::internal(
                "groupcounts: invalid categorical group key",
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    categorical_levels_for_observations(prototype, &labels)
}

fn logical_labels(atoms: Vec<KeyAtom>) -> BuiltinResult<Value> {
    let values: Vec<u8> = atoms
        .into_iter()
        .map(|atom| match atom {
            KeyAtom::Logical(value) => Ok(u8::from(value)),
            _ => Err(error::internal("groupcounts: invalid logical group key")),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let rows = values.len();
    LogicalArray::new(values, vec![rows, 1])
        .map(Value::LogicalArray)
        .map_err(error::internal)
}

fn cellstr_labels(atoms: Vec<KeyAtom>) -> BuiltinResult<Value> {
    let values: Vec<Value> = atoms
        .into_iter()
        .map(|atom| match atom {
            KeyAtom::Text(value) => Value::CharArray(CharArray::new_row(&value)),
            KeyAtom::Missing => Value::CharArray(CharArray::new_row("")),
            value => Value::CharArray(CharArray::new_row(&value.label())),
        })
        .collect();
    let rows = values.len();
    CellArray::new(values, rows, 1)
        .map(Value::Cell)
        .map_err(error::internal)
}

fn select_observed_rows(column: &GroupColumn, atoms: &[KeyAtom]) -> BuiltinResult<Value> {
    let rows = atoms
        .iter()
        .map(|target| {
            column
                .atoms()
                .iter()
                .position(|atom| atom == target)
                .ok_or_else(|| error::internal("groupcounts: group label has no source value"))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    select_group_rows(&column.value, &rows).map_err(error::internal)
}
