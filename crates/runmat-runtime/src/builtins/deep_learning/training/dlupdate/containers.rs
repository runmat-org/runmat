//! Shape-preserving cell and structure traversal.
use super::*;

pub(super) async fn dlupdate_struct_array(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    reference: &StructArray,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let arrays = values
        .iter()
        .map(|value| match value {
            Value::StructArray(array)
                if array.shape() == reference.shape()
                    && array.field_names().eq(reference.field_names()) =>
            {
                Ok(array)
            }
            Value::StructArray(_) => Err(deep_learning_error(
                "dlupdate",
                "dlupdate: structure-array parameter trees must have matching shapes and ordered fields",
            )),
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching structure-array tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let field_names = reference.field_names().cloned().collect::<Vec<_>>();
    let capacity = field_names
        .len()
        .checked_mul(reference.len())
        .ok_or_else(|| deep_learning_error("dlupdate", "dlupdate: structure array is too large"))?;
    let mut outputs = (0..requested_outputs)
        .map(|_| Vec::with_capacity(capacity))
        .collect::<Vec<_>>();
    for name in &field_names {
        for index in 0..reference.len() {
            let fields = arrays
                .iter()
                .map(|array| {
                    array
                        .field_values(name)
                        .and_then(|column| column.get(index))
                        .cloned()
                        .ok_or_else(|| {
                            deep_learning_error(
                                "dlupdate",
                                "dlupdate: structure-array field storage is incomplete",
                            )
                        })
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            let updated =
                dlupdate_node(runtime, function, &fields, requested_outputs, depth).await?;
            for (output, value) in outputs.iter_mut().zip(updated) {
                output.push(value);
            }
        }
    }
    outputs
        .into_iter()
        .map(|values| {
            StructArray::normalize_field_major(
                field_names.clone(),
                values,
                reference.shape().to_vec(),
            )
            .map_err(|error| deep_learning_error("dlupdate", error))
        })
        .collect()
}

pub(super) async fn dlupdate_struct(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    reference: &StructValue,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let field_names = reference.field_names().cloned().collect::<Vec<_>>();
    let structs = values
        .iter()
        .map(|value| match value {
            Value::Struct(st) => {
                let names = st.field_names().cloned().collect::<Vec<_>>();
                if names == field_names {
                    Ok(st)
                } else {
                    Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: struct parameter trees must have the same fields in the same order",
                    ))
                }
            }
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching struct tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    let mut outputs = (0..requested_outputs)
        .map(|_| StructValue::new())
        .collect::<Vec<_>>();
    for name in field_names {
        let field_values = structs
            .iter()
            .map(|st| {
                st.fields
                    .get(&name)
                    .cloned()
                    .unwrap_or(Value::Num(f64::NAN))
            })
            .collect::<Vec<_>>();
        let updated =
            dlupdate_node(runtime, function, &field_values, requested_outputs, depth).await?;
        for (out, value) in outputs.iter_mut().zip(updated.into_iter()) {
            out.insert(name.clone(), value);
        }
    }

    Ok(outputs.into_iter().map(Value::Struct).collect())
}

pub(super) async fn dlupdate_cell(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    reference: &CellArray,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let cells = values
        .iter()
        .map(|value| match value {
            Value::Cell(cell) if cell.shape == reference.shape => Ok(cell),
            Value::Cell(_) => Err(deep_learning_error(
                "dlupdate",
                "dlupdate: cell parameter trees must have the same shape",
            )),
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching cell tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    let mut outputs = (0..requested_outputs)
        .map(|_| Vec::with_capacity(reference.data.len()))
        .collect::<Vec<_>>();
    for idx in 0..reference.data.len() {
        let elements = cells
            .iter()
            .map(|cell| cell.data[idx].clone())
            .collect::<Vec<_>>();
        let updated = dlupdate_node(runtime, function, &elements, requested_outputs, depth).await?;
        for (out, value) in outputs.iter_mut().zip(updated.into_iter()) {
            out.push(value);
        }
    }

    outputs
        .into_iter()
        .map(|data| {
            CellArray::new_with_shape(data, reference.shape.clone())
                .map(Value::Cell)
                .map_err(|err| deep_learning_error("dlupdate", err))
        })
        .collect()
}
