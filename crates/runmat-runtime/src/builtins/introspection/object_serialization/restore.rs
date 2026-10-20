use super::*;

use super::assembly::{
    root_cell, root_output_list, root_restored_object_array, root_struct, root_struct_array,
};

#[cfg(test)]
mod tests;

enum Step {
    Enter(Value, usize),
    EnterRetained(runmat_value::RetainedArrayValue, usize),
    Struct(Vec<String>),
    Cell(Vec<usize>, usize),
    OutputList(usize),
    StructArray(Vec<String>, Vec<usize>, usize),
    ObjectArray(runmat_types::ClassIdentity, Vec<usize>, usize),
}

#[cfg(test)]
pub(super) async fn restore_value_after_load(value: Value) -> crate::BuiltinResult<RootedValue> {
    restore_value_after_load_in_context(value, None).await
}

pub(super) async fn restore_value_after_load_in_context(
    value: Value,
    _runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<RootedValue> {
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
                match value {
                    Value::Struct(structure) => {
                        let mut names = Vec::with_capacity(structure.fields.len());
                        let mut values = Vec::with_capacity(structure.fields.len());
                        for (name, value) in structure.into_fields_unbounded() {
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
                        let count = cell.data().len();
                        work.push(Step::Cell(cell.shape.to_vec(), count));
                        match cell.into_owned_data() {
                            Ok(values) => work.extend(
                                values
                                    .into_iter()
                                    .rev()
                                    .map(|value| Step::Enter(value, depth + 1)),
                            ),
                            Err(cell) => work.extend((0..count).rev().map(|index| {
                                Step::EnterRetained(
                                    cell.data_owner()
                                        .retain_value_at(index)
                                        .expect("shared cell child"),
                                    depth + 1,
                                )
                            })),
                        }
                    }
                    Value::OutputList(values) => {
                        let count = values.len();
                        work.push(Step::OutputList(count));
                        match values.try_into_owned() {
                            Ok(values) => work.extend(
                                values
                                    .into_iter()
                                    .rev()
                                    .map(|value| Step::Enter(value, depth + 1)),
                            ),
                            Err(values) => work.extend((0..count).rev().map(|index| {
                                Step::EnterRetained(
                                    values.retain_value_at(index).expect("shared output child"),
                                    depth + 1,
                                )
                            })),
                        }
                    }
                    Value::ObjectArray(array) if !array.is_empty() => {
                        let count = array.len();
                        let class_name = array.class_name().clone();
                        let shape = array.shape().to_vec();
                        work.push(Step::ObjectArray(class_name, shape, count));
                        work.extend(
                            array
                                .data()
                                .iter()
                                .cloned()
                                .rev()
                                .map(|value| Step::Enter(value, depth + 1)),
                        );
                    }
                    other => results.push(root_value(other)?),
                }
            }
            Step::EnterRetained(value, depth) => {
                if depth > MAX_SERIALIZATION_DEPTH {
                    return Err(serialization_error(
                        LOADOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_RECURSION,
                        "nested object deserialization exceeded the supported depth",
                    ));
                }
                match value.value() {
                    Value::Cell(cell) => {
                        let count = cell.data().len();
                        work.push(Step::Cell(cell.shape.to_vec(), count));
                        work.extend((0..count).rev().map(|index| {
                            Step::EnterRetained(
                                value
                                    .project(|parent| match parent {
                                        Value::Cell(cell) => cell.data().get(index),
                                        _ => None,
                                    })
                                    .expect("retained cell child"),
                                depth + 1,
                            )
                        }));
                    }
                    Value::OutputList(values) => {
                        let count = values.len();
                        work.push(Step::OutputList(count));
                        work.extend((0..count).rev().map(|index| {
                            Step::EnterRetained(
                                value
                                    .project(|parent| match parent {
                                        Value::OutputList(values) => values.get(index),
                                        _ => None,
                                    })
                                    .expect("retained output child"),
                                depth + 1,
                            )
                        }));
                    }
                    Value::Struct(structure) => {
                        let names: Vec<String> =
                            structure.fields.keys().map(ToOwned::to_owned).collect();
                        work.push(Step::Struct(names.clone()));
                        work.extend(names.into_iter().rev().map(|name| {
                            Step::EnterRetained(
                                value
                                    .project(|parent| match parent {
                                        Value::Struct(structure) => structure.fields.get(&name),
                                        _ => None,
                                    })
                                    .expect("retained struct child"),
                                depth + 1,
                            )
                        }));
                    }
                    Value::StructArray(array) => {
                        let names: Vec<String> =
                            array.field_names().map(ToOwned::to_owned).collect();
                        let shape = array.shape().to_vec();
                        let field_len = shape.iter().product::<usize>();
                        let count = names.len() * field_len;
                        work.push(Step::StructArray(names.clone(), shape, count));
                        for name in names.into_iter().rev() {
                            for index in (0..field_len).rev() {
                                work.push(Step::EnterRetained(
                                    value
                                        .project(|parent| match parent {
                                            Value::StructArray(array) => array
                                                .field_values(&name)
                                                .and_then(|column| column.get(index)),
                                            _ => None,
                                        })
                                        .expect("retained struct-array child"),
                                    depth + 1,
                                ));
                            }
                        }
                    }
                    Value::ObjectArray(array) if !array.is_empty() => {
                        let count = array.len();
                        work.push(Step::ObjectArray(
                            array.class_name().clone(),
                            array.shape().to_vec(),
                            count,
                        ));
                        work.extend((0..count).rev().map(|index| {
                            Step::EnterRetained(
                                value
                                    .project(|parent| match parent {
                                        Value::ObjectArray(array) => array.get_linear(index),
                                        _ => None,
                                    })
                                    .expect("retained object-array child"),
                                depth + 1,
                            )
                        }));
                    }
                    _ => results.push(root_value(value.value().clone())?),
                }
            }
            Step::Struct(names) => {
                let restored = root_struct(names, &mut results)?;
                results.push(restored);
            }
            Step::Cell(shape, count) => {
                let restored = root_cell(shape, count, &mut results)?;
                results.push(restored);
            }
            Step::OutputList(count) => {
                let restored = root_output_list(count, &mut results)?;
                results.push(restored);
            }
            Step::StructArray(names, shape, count) => {
                let restored = root_struct_array(names, shape, count, &mut results)?;
                results.push(restored);
            }
            Step::ObjectArray(class_name, shape, count) => {
                let restored = root_restored_object_array(class_name, shape, count, &mut results)?;
                results.push(restored);
            }
        }
    }
    Ok(results.pop().expect("restored root value"))
}
