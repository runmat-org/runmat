use runmat_value::Value;

use crate::builtins::array::grouping::variables::{columns_from_arguments, GroupColumn};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::table::{
    is_tabular_object, parse_variable_selector_for_object, table_height,
    table_variable_names_from_object, table_variables,
};
use crate::BuiltinResult;

use super::error;

pub(super) enum FindGroupsInput {
    Vectors {
        columns: Vec<GroupColumn>,
        output_shape: Vec<usize>,
    },
    Table {
        columns: Vec<GroupColumn>,
        rows: usize,
    },
}

impl FindGroupsInput {
    pub(super) fn prepare(values: Vec<Value>) -> BuiltinResult<Self> {
        let Some(first) = values.first() else {
            return Err(error::invalid(
                "findgroups: expected at least one grouping input",
            ));
        };
        if matches!(first, Value::Object(object) if is_tabular_object(object)) {
            return table(values);
        }
        vectors(values)
    }

    pub(super) fn columns(&self) -> &[GroupColumn] {
        match self {
            Self::Vectors { columns, .. } | Self::Table { columns, .. } => columns,
        }
    }
}

fn vectors(values: Vec<Value>) -> BuiltinResult<FindGroupsInput> {
    let output_shape = vector_output_shape(&values[0])?;
    for value in values.iter().skip(1) {
        if vector_output_shape(value)? != output_shape {
            return Err(error::invalid(
                "findgroups: grouping inputs must have matching sizes and orientations",
            ));
        }
    }
    let columns = columns_from_arguments(values, true).map_err(error::invalid)?;
    Ok(FindGroupsInput::Vectors {
        columns,
        output_shape,
    })
}

fn table(values: Vec<Value>) -> BuiltinResult<FindGroupsInput> {
    if values.len() > 2 {
        return Err(error::invalid(
            "findgroups: the table form accepts at most one selector",
        ));
    }
    let Value::Object(object) = &values[0] else {
        unreachable!("table form was checked by the caller")
    };
    let names = table_variable_names_from_object(object)?;
    let selected = parse_variable_selector_for_object(values.get(1), object, &names)?;
    let variables = table_variables(object)?;
    let rows = table_height(object)?;
    let columns = selected
        .into_iter()
        .map(|name| {
            let value =
                variables.fields.get(&name).cloned().ok_or_else(|| {
                    error::invalid(format!("findgroups: unknown variable '{name}'"))
                })?;
            let column = GroupColumn::vector(name.clone(), value).map_err(error::invalid)?;
            if column.rows != rows {
                return Err(error::invalid(format!(
                    "findgroups: table variable '{name}' has {} rows; expected {rows}",
                    column.rows
                )));
            }
            Ok(column)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(FindGroupsInput::Table { columns, rows })
}

fn vector_output_shape(value: &Value) -> BuiltinResult<Vec<usize>> {
    let shape = match value {
        Value::Tensor(value) => value.shape.clone(),
        Value::LogicalArray(value) => value.shape.clone(),
        Value::StringArray(value) => value.shape.clone(),
        Value::Cell(value) => vec![value.rows, value.cols],
        Value::GpuTensor(value) => value.shape.clone(),
        Value::Object(object) if object.is_class(runmat_types::standard::DATETIME) => {
            crate::builtins::datetime::serials_from_datetime_value(value)?.shape
        }
        Value::Object(object) if object.is_class(runmat_types::standard::DURATION) => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)?.shape
        }
        Value::Object(object) if object.is_class(runmat_types::standard::CALENDAR_DURATION) => {
            let (months, days) =
                crate::builtins::datetime::calendar_duration_tensors_from_value(value)?;
            if months.shape != days.shape {
                return Err(error::invalid(
                    "findgroups: calendarDuration component shapes do not match",
                ));
            }
            months.shape
        }
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            let rows = crate::builtins::table::categorical_observation_labels(object)?.len();
            vec![rows, 1]
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => vec![1, 1],
        Value::SparseTensor(_) => {
            return Err(error::invalid(
                "findgroups: sparse grouping inputs are not supported",
            ))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(error::invalid(
                "findgroups: complex grouping inputs are not supported",
            ))
        }
        other => {
            return Err(error::invalid(format!(
                "findgroups: unsupported grouping input {other:?}"
            )))
        }
    };
    if shape.iter().filter(|dimension| **dimension > 1).count() <= 1 {
        return Ok(shape);
    }
    let rows = shape.first().copied().unwrap_or_else(|| match value {
        Value::Tensor(value) => tensor_utils::tensor_element_len(value),
        _ => 1,
    });
    Ok(vec![rows, 1])
}
