//! Shape-preserving cell and structure traversal.
use super::*;

pub(super) async fn dlupdate_struct_array(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    reference: &StructArray,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<RootedValues> {
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
    let field_names = reference
        .field_names()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let capacity = field_names
        .len()
        .checked_mul(reference.len())
        .ok_or_else(|| deep_learning_error("dlupdate", "dlupdate: structure array is too large"))?;
    let mut outputs = (0..requested_outputs)
        .map(|_| RootedValuesBuilder::with_capacity(capacity))
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
            updated.with_owned_output_storage(|updated| {
                let updated = into_owned_outputs(updated)?;
                for (output, value) in outputs.iter_mut().zip(updated) {
                    output
                        .push(value)
                        .map_err(|error| error.into_runtime_error())?;
                }
                Ok::<_, crate::RuntimeError>(())
            })?;
        }
    }
    let mut rebuilt = RootedValuesBuilder::with_capacity(requested_outputs);
    for values in outputs {
        let values = values
            .finish()
            .map_err(|error| error.into_runtime_error())?;
        let result = values.with_unbounded_values(|values| {
            let value = StructArray::normalize_field_major(
                field_names.clone(),
                values,
                reference.shape().to_vec(),
            )
            .map_err(|error| deep_learning_error("dlupdate", error))?;
            RootedValue::new(value).map_err(|error| error.into_runtime_error())
        })??;
        result.with_owned(|value| {
            rebuilt
                .push(value)
                .map_err(|error| error.into_runtime_error())
        })?;
    }
    rebuilt.finish().map_err(|error| error.into_runtime_error())
}

pub(super) async fn dlupdate_struct(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    reference: &StructValue,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<RootedValues> {
    let field_names = reference
        .field_names()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let structs = values
        .iter()
        .map(|value| match value {
            Value::Struct(st) => {
                let names = st.field_names().map(ToOwned::to_owned).collect::<Vec<_>>();
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
        .map(|_| RootedValuesBuilder::with_capacity(field_names.len()))
        .collect::<Vec<_>>();
    for name in &field_names {
        let field_values = structs
            .iter()
            .map(|st| st.fields.get(name).cloned().unwrap_or(Value::Num(f64::NAN)))
            .collect::<Vec<_>>();
        let updated =
            dlupdate_node(runtime, function, &field_values, requested_outputs, depth).await?;
        updated.with_owned_output_storage(|updated| {
            let updated = into_owned_outputs(updated)?;
            for (out, value) in outputs.iter_mut().zip(updated) {
                out.push(value)
                    .map_err(|error| error.into_runtime_error())?;
            }
            Ok::<_, crate::RuntimeError>(())
        })?;
    }

    let mut rebuilt = RootedValuesBuilder::with_capacity(requested_outputs);
    for fields in outputs {
        let fields = fields
            .finish()
            .map_err(|error| error.into_runtime_error())?;
        let result = if let Some(runtime) = runtime {
            crate::execution::construct_struct_in_context(
                runtime,
                field_names.iter().map(String::as_str),
                fields,
            )?
        } else {
            fields.with_owned_output_storage(|fields| {
                use crate::context::host_allocation::{
                    HostAllocationPolicy, HostAllocationReceipt,
                };
                if HostAllocationReceipt::policy(None).map_err(|source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        None,
                        "RunMat:dlupdate:Structure",
                        "structure destination policy is unavailable",
                        source,
                    )
                })? != HostAllocationPolicy::Unbounded
                {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "structure destination has no selected managed context",
                    ));
                }
                let fields = into_owned_outputs(fields)?;
                let mut out = runmat_value::UnboundedStructValueBuilder::new();
                for (name, value) in field_names.iter().cloned().zip(fields) {
                    out.insert(name, value);
                }
                RootedValue::new(Value::Struct(out.finish()))
                    .map_err(|error| error.into_runtime_error())
            })?
        };
        result.with_owned(|value| {
            rebuilt
                .push(value)
                .map_err(|error| error.into_runtime_error())
        })?;
    }
    rebuilt.finish().map_err(|error| error.into_runtime_error())
}

pub(super) async fn dlupdate_cell(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    reference: &CellArray,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<RootedValues> {
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
        .map(|_| RootedValuesBuilder::with_capacity(reference.data().len()))
        .collect::<Vec<_>>();
    for idx in 0..reference.data().len() {
        let elements = cells
            .iter()
            .map(|cell| cell.data()[idx].clone())
            .collect::<Vec<_>>();
        let updated = dlupdate_node(runtime, function, &elements, requested_outputs, depth).await?;
        updated.with_owned_output_storage(|updated| {
            let updated = into_owned_outputs(updated)?;
            for (out, value) in outputs.iter_mut().zip(updated) {
                out.push(value)
                    .map_err(|error| error.into_runtime_error())?;
            }
            Ok::<_, crate::RuntimeError>(())
        })?;
    }

    let mut rebuilt = RootedValuesBuilder::with_capacity(requested_outputs);
    for data in outputs {
        let data = data.finish().map_err(|error| error.into_runtime_error())?;
        let result = data.with_unbounded_values(|data| {
            let value = CellArray::new_with_shape(data, reference.shape.as_slice().to_vec())
                .map(Value::Cell)
                .map_err(|err| deep_learning_error("dlupdate", err))?;
            RootedValue::new(value).map_err(|error| error.into_runtime_error())
        })??;
        result.with_owned(|value| {
            rebuilt
                .push(value)
                .map_err(|error| error.into_runtime_error())
        })?;
    }
    rebuilt.finish().map_err(|error| error.into_runtime_error())
}
