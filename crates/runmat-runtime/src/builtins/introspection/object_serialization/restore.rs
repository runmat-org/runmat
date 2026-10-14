use super::*;

mod envelope;
use super::assembly::{root_cell, root_struct, root_struct_array, SerializationPhase};
use envelope::restore_envelope;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod pending_tests;

enum Step {
    Enter(Value, usize),
    Struct(Vec<String>),
    Cell(Vec<usize>, usize),
    StructArray(Vec<String>, Vec<usize>, usize),
    Envelope {
        class_name: String,
        kind: String,
        had_saveobj: bool,
    },
}

pub(super) async fn restore_value_after_load(value: Value) -> crate::BuiltinResult<RootedValue> {
    // The canonical Value tracer is iterative, so even an over-depth input
    // can be retained safely before the first effectful restoration step.
    let _input_root =
        RootedValueLease::for_traced(&value).map_err(|error| error.into_runtime_error())?;
    let mut work = vec![Step::Enter(value, 0)];
    let mut results = Vec::<RootedValue>::new();
    while let Some(step) = work.pop() {
        match step {
            Step::Enter(value, depth) => {
                if depth > MAX_SERIALIZATION_DEPTH {
                    return Err(serialization_error(
                        LOADOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_RECURSION,
                        "nested object deserialization exceeded the supported depth",
                    ));
                }
                let value = match into_serialized_envelope(value) {
                    Ok((class_name, kind, had_saveobj, payload)) => {
                        work.push(Step::Envelope {
                            class_name,
                            kind,
                            had_saveobj,
                        });
                        work.push(Step::Enter(payload, depth + 1));
                        continue;
                    }
                    Err(value) => value,
                };
                match value {
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
                let restored = root_struct(names, &mut results)?;
                results.push(restored);
            }
            Step::Cell(shape, count) => {
                let restored = root_cell(shape, count, &mut results, SerializationPhase::Load)?;
                results.push(restored);
            }
            Step::StructArray(names, shape, count) => {
                let restored =
                    root_struct_array(names, shape, count, &mut results, SerializationPhase::Load)?;
                results.push(restored);
            }
            Step::Envelope {
                class_name,
                kind,
                had_saveobj,
            } => {
                let payload = results.pop().expect("restored envelope payload");
                results.push(restore_envelope(payload, class_name, kind, had_saveobj).await?);
            }
        }
    }
    Ok(results.pop().expect("restored root value"))
}
