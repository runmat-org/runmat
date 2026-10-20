use super::*;

fn children(results: &mut Vec<RootedValue>, count: usize) -> Vec<RootedValue> {
    let start = results.len() - count;
    results.split_off(start)
}

pub(super) fn root_struct(
    names: Vec<String>,
    results: &mut Vec<RootedValue>,
) -> crate::BuiltinResult<RootedValue> {
    let mut builder = RootedValuesBuilder::with_capacity(names.len());
    for child in children(results, names.len()) {
        builder.push_owned(child);
    }
    builder
        .finish()
        .map_err(|error| error.into_runtime_error())?
        .with_owned_output_storage(|values| {
            let mut structure = runmat_value::UnboundedStructValueBuilder::new();
            let values = values.try_into_owned().map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?;
            for (name, value) in names.into_iter().zip(values) {
                structure.insert(name, value);
            }
            root_value(Value::Struct(structure.finish()))
        })
}

pub(super) fn root_cell(
    shape: Vec<usize>,
    count: usize,
    results: &mut Vec<RootedValue>,
) -> crate::BuiltinResult<RootedValue> {
    let mut builder = RootedValuesBuilder::with_capacity(count);
    for child in children(results, count) {
        builder.push_owned(child);
    }
    builder
        .finish()
        .map_err(|error| error.into_runtime_error())?
        .with_unbounded_values(|values| {
            let cell = CellArray::new_with_shape(values, shape).map_err(|error| {
                serialization_error(
                    LOADOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_ARGUMENT,
                    format!("failed to rebuild deserialized cell payload: {error}"),
                )
            })?;
            root_value(Value::Cell(cell))
        })?
}

pub(super) fn root_struct_array(
    names: Vec<String>,
    shape: Vec<usize>,
    count: usize,
    results: &mut Vec<RootedValue>,
) -> crate::BuiltinResult<RootedValue> {
    let mut builder = RootedValuesBuilder::with_capacity(count);
    for child in children(results, count) {
        builder.push_owned(child);
    }
    builder
        .finish()
        .map_err(|error| error.into_runtime_error())?
        .with_unbounded_values(|values| {
            let value = runmat_value::StructArray::normalize_field_major(names, values, shape)
                .map_err(|error| {
                    serialization_error(
                        LOADOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_ARGUMENT,
                        format!("failed to rebuild deserialized structure array: {error}"),
                    )
                })?;
            root_value(value)
        })?
}

pub(super) fn root_restored_object_array(
    class_name: runmat_types::ClassIdentity,
    shape: Vec<usize>,
    count: usize,
    results: &mut Vec<RootedValue>,
) -> crate::BuiltinResult<RootedValue> {
    let mut builder = RootedValuesBuilder::with_capacity(count);
    for child in children(results, count) {
        builder.push_owned(child);
    }
    builder
        .finish()
        .map_err(|error| error.into_runtime_error())?
        .with_unbounded_values(|values| {
            let array =
                runmat_value::ObjectArray::new(class_name, values, shape).map_err(|error| {
                    serialization_error(
                        LOADOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_ARGUMENT,
                        format!("failed to rebuild restored object array: {error}"),
                    )
                })?;
            root_value(Value::ObjectArray(array))
        })?
}

pub(super) fn root_output_list(
    count: usize,
    results: &mut Vec<RootedValue>,
) -> crate::BuiltinResult<RootedValue> {
    let mut builder = RootedValuesBuilder::with_capacity(count);
    for child in children(results, count) {
        builder.push_owned(child);
    }
    builder
        .finish()
        .map_err(|error| error.into_runtime_error())?
        .with_owned_output_storage(|storage| root_value(Value::OutputList(storage)))
}
