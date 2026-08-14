use serde::{Deserialize, Serialize};

use crate::object::dispatch::call_object_index_descriptor_method_with_outputs;
use crate::object::indexing::{ObjectIndexDescriptor, ObjectIndexSelector};
use crate::{runtime_error::semantic_error, RuntimeError};
use runmat_value::Value;

/// Describes one source-level call argument after lowering.
///
/// The descriptor is shared by bytecode and native execution. It captures only
/// language-level expansion semantics and deliberately contains no operand-stack
/// or instruction-decoding state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentSpec {
    pub is_expand: bool,
    pub num_indices: usize,
    pub expand_all: bool,
}

/// One source-level argument after an executor has materialized its operands.
///
/// VM stack order and Native IR value identities stay in their executors; the
/// expansion rules themselves are shared Runtime language semantics.
#[derive(Debug, Clone)]
pub enum MaterializedArgument {
    Single(Value),
    Expansion {
        base: Value,
        indices: Vec<Value>,
        expand_all: bool,
    },
}

pub async fn expand_arguments(
    arguments: Vec<MaterializedArgument>,
) -> Result<Vec<Value>, RuntimeError> {
    let mut expanded_arguments = Vec::new();
    for argument in arguments {
        match argument {
            MaterializedArgument::Single(value) => expanded_arguments.push(value),
            MaterializedArgument::Expansion {
                base,
                indices,
                expand_all,
            } => {
                let values = if expand_all {
                    match base {
                        Value::OutputList(outputs) => outputs,
                        Value::Cell(cell) => crate::object::cell::expand_all_cell_values(&cell)?,
                        base @ (Value::Object(_) | Value::HandleObject(_)) => {
                            expand_brace_values(base, &[], None).await?
                        }
                        _ => {
                            return Err(semantic_error(
                                "InvalidExpandAllTarget",
                                "Comma-separated-list expansion requires a cell array, output list, or object",
                            ));
                        }
                    }
                } else {
                    match (base, indices.len()) {
                        (Value::Cell(cell), 1 | 2) => {
                            crate::object::cell::expand_cell_indices(&cell, &indices)?
                        }
                        (Value::OutputList(outputs), 1 | 2) => {
                            let cols = outputs.len();
                            let cell = runmat_value::CellArray::new(outputs, 1, cols).map_err(
                                |error| {
                                    semantic_error(
                                        "ShapeMismatch",
                                        format!("output-list expansion: {error}"),
                                    )
                                },
                            )?;
                            crate::object::cell::expand_cell_indices(&cell, &indices)?
                        }
                        (base @ (Value::Object(_) | Value::HandleObject(_)), _) => {
                            expand_brace_values(base, &indices, None).await?
                        }
                        _ => {
                            return Err(semantic_error(
                                "InvalidExpandTarget",
                                "Indexed comma-separated-list expansion requires a cell array, output list, or object",
                            ));
                        }
                    }
                };
                expanded_arguments.extend(values);
            }
        }
    }
    Ok(expanded_arguments)
}

pub async fn expand_brace_values(
    base: Value,
    indices: &[Value],
    pad_to_outputs: Option<usize>,
) -> Result<Vec<Value>, RuntimeError> {
    let mut values = match base {
        Value::Cell(cell) => {
            if indices.is_empty() {
                if let Some(output_count) = pad_to_outputs {
                    crate::object::cell::expand_cell_values(&cell, &[], output_count)?
                } else {
                    crate::object::cell::expand_all_cell_values(&cell)?
                }
            } else {
                crate::object::cell::expand_cell_indices(&cell, indices)?
            }
        }
        base @ (Value::Object(_) | Value::HandleObject(_)) => {
            let value = call_object_index_descriptor_method_with_outputs(
                ObjectIndexDescriptor::subsref_brace(
                    base,
                    ObjectIndexSelector::IndexValues {
                        values: indices.to_vec(),
                    },
                ),
                pad_to_outputs.unwrap_or(1),
            )
            .await?;
            match value {
                Value::OutputList(values) => values,
                value => vec![value],
            }
        }
        _ => {
            return Err(semantic_error(
                "CellExpansionOnNonCell",
                "Cell expansion on non-cell",
            ));
        }
    };
    if let Some(output_count) = pad_to_outputs {
        if values.len() > output_count {
            values.truncate(output_count);
        } else {
            values.resize(output_count, Value::Num(0.0));
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_value::{CellArray, Tensor, Value};

    use super::{expand_arguments, MaterializedArgument};

    #[test]
    fn expansion_preserves_source_and_comma_list_order() {
        let cell = CellArray::new(vec![Value::Num(2.0), Value::Num(3.0)], 1, 2).unwrap();
        let values = block_on(expand_arguments(vec![
            MaterializedArgument::Single(Value::Num(1.0)),
            MaterializedArgument::Expansion {
                base: Value::Cell(cell),
                indices: Vec::new(),
                expand_all: true,
            },
            MaterializedArgument::Single(Value::Num(4.0)),
        ]))
        .expect("expand arguments");
        assert_eq!(
            values,
            vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
            ]
        );
    }

    #[test]
    fn output_list_index_expansion_uses_cell_index_semantics() {
        let values = block_on(expand_arguments(vec![MaterializedArgument::Expansion {
            base: Value::OutputList(vec![Value::Num(9.0), Value::Num(2.0)]),
            indices: vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
            )],
            expand_all: false,
        }]))
        .expect("expand output list");
        assert_eq!(values, vec![Value::Num(9.0), Value::Num(2.0)]);
    }

    #[test]
    fn invalid_expansion_retains_stable_identifier() {
        let error = block_on(expand_arguments(vec![MaterializedArgument::Expansion {
            base: Value::Num(1.0),
            indices: Vec::new(),
            expand_all: true,
        }]))
        .expect_err("numeric expansion must fail");
        assert_eq!(error.identifier(), Some("RunMat:InvalidExpandAllTarget"));
    }
}
