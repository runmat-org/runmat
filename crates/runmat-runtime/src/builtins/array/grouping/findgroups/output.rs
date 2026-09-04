use runmat_value::{Tensor, Value};

use crate::builtins::array::grouping::keys::GroupIndex;
use crate::builtins::array::grouping::variables::label_columns;
use crate::builtins::table::table_from_columns;
use crate::BuiltinResult;

use super::error;
use super::input::FindGroupsInput;

pub(super) fn build(input: &FindGroupsInput, index: &GroupIndex) -> BuiltinResult<Value> {
    let output_shape = match input {
        FindGroupsInput::Vectors { output_shape, .. } => output_shape.clone(),
        FindGroupsInput::Table { rows, .. } => vec![*rows, 1],
    };
    let g = Tensor::new(index.ids.clone(), output_shape)
        .map(Value::Tensor)
        .map_err(error::internal)?;
    let labels = label_columns(input.columns(), index).map_err(error::internal)?;
    let mut outputs = vec![g];
    match input {
        FindGroupsInput::Vectors { .. } => outputs.extend(labels),
        FindGroupsInput::Table { columns, .. } => {
            let names = columns.iter().map(|column| column.name.clone()).collect();
            outputs.push(table_from_columns(names, labels)?);
        }
    }
    super::super::requested_outputs::finish(outputs)
}
