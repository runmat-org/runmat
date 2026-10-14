use super::*;

use super::assembly::{root_cell, root_struct, root_struct_array, SerializationPhase};

enum Step {
    Enter(Value, usize),
    Struct(Vec<String>),
    Cell(Vec<usize>, usize),
    StructArray(Vec<String>, Vec<usize>, usize),
    Object {
        class_name: runmat_types::ClassIdentity,
        kind: &'static str,
        had_saveobj: bool,
        payload_lease: Option<RootedValueLease>,
    },
}

pub(super) async fn prepare_value_for_save(value: Value) -> crate::BuiltinResult<RootedValue> {
    let _input_root =
        RootedValueLease::for_traced(&value).map_err(|error| error.into_runtime_error())?;
    let mut work = vec![Step::Enter(value, 0)];
    let mut results = Vec::<RootedValue>::new();
    while let Some(step) = work.pop() {
        match step {
            Step::Enter(value, depth) => {
                if depth > MAX_SERIALIZATION_DEPTH {
                    return Err(serialization_error(
                        SAVEOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_RECURSION,
                        "nested object serialization exceeded the supported depth",
                    ));
                }
                match value {
                    receiver @ (Value::Object(_) | Value::HandleObject(_)) => {
                        ensure_no_native_payload(&receiver)?;
                        let class_name = object_class_name(&receiver).ok_or_else(|| {
                            serialization_error(
                                SAVEOBJ_METHOD.display_name(),
                                &SERIALIZATION_ERROR_ARGUMENT,
                                "object receiver is missing class metadata",
                            )
                        })?;
                        let kind = if matches!(receiver, Value::HandleObject(_)) {
                            SERIALIZED_KIND_HANDLE
                        } else {
                            SERIALIZED_KIND_VALUE
                        };
                        let saved = call_saveobj_if_available(receiver.clone()).await?;
                        let had_saveobj = saved.is_some();
                        let (payload, payload_lease) = match saved {
                            Some(saved) => {
                                let (payload, lease) = saved.into_value_and_lease();
                                (payload, Some(lease))
                            }
                            None => (Value::Struct(object_properties_payload(&receiver)?), None),
                        };
                        work.push(Step::Object {
                            class_name,
                            kind,
                            had_saveobj,
                            payload_lease,
                        });
                        work.push(Step::Enter(payload, depth + 1));
                    }
                    Value::Struct(structure) => {
                        let mut names = Vec::with_capacity(structure.fields.len());
                        let mut values = Vec::with_capacity(structure.fields.len());
                        for (name, value) in structure.fields {
                            names.push(name);
                            values.push(value);
                        }
                        work.push(Step::Struct(names));
                        work.extend(
                            values
                                .into_iter()
                                .rev()
                                .map(|value| Step::Enter(value, depth + 1)),
                        );
                    }
                    Value::StructArray(array) => {
                        let (columns, shape) = array.into_field_major_parts();
                        let mut names = Vec::with_capacity(columns.len());
                        let mut values = Vec::new();
                        for (name, column) in columns {
                            names.push(name);
                            values.extend(column);
                        }
                        let count = values.len();
                        work.push(Step::StructArray(names, shape, count));
                        work.extend(
                            values
                                .into_iter()
                                .rev()
                                .map(|value| Step::Enter(value, depth + 1)),
                        );
                    }
                    Value::Cell(cell) => {
                        let count = cell.data.len();
                        work.push(Step::Cell(cell.shape, count));
                        work.extend(
                            cell.data
                                .into_iter()
                                .rev()
                                .map(|value| Step::Enter(value, depth + 1)),
                        );
                    }
                    other => results.push(root_value(other)?),
                }
            }
            Step::Struct(names) => {
                let prepared = root_struct(names, &mut results)?;
                results.push(prepared);
            }
            Step::Cell(shape, count) => {
                let prepared = root_cell(shape, count, &mut results, SerializationPhase::Save)?;
                results.push(prepared);
            }
            Step::StructArray(names, shape, count) => {
                let prepared =
                    root_struct_array(names, shape, count, &mut results, SerializationPhase::Save)?;
                results.push(prepared);
            }
            Step::Object {
                class_name,
                kind,
                had_saveobj,
                payload_lease,
            } => {
                let payload = results.pop().expect("prepared object payload");
                let envelope = payload.try_map_owned(
                    |payload| {
                        Ok(serialized_object_envelope(
                            class_name,
                            kind,
                            had_saveobj,
                            payload,
                        ))
                    },
                    |error| error.into_runtime_error(),
                )?;
                drop(payload_lease);
                results.push(envelope);
            }
        }
    }
    Ok(results.pop().expect("prepared root value"))
}

#[cfg(test)]
mod tests;
