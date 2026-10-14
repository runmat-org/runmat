use super::*;

#[derive(Clone, Copy)]
pub(super) enum SerializationPhase {
    Save,
    Load,
}

impl SerializationPhase {
    fn method(self) -> &'static str {
        match self {
            Self::Save => SAVEOBJ_METHOD.display_name(),
            Self::Load => LOADOBJ_METHOD.display_name(),
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Save => "serialized",
            Self::Load => "deserialized",
        }
    }
}

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
        .with_owned_values(|values| {
            let mut structure = StructValue::new();
            for (name, value) in names.into_iter().zip(values) {
                structure.insert(name, value);
            }
            root_value(Value::Struct(structure))
        })
}

pub(super) fn root_cell(
    shape: Vec<usize>,
    count: usize,
    results: &mut Vec<RootedValue>,
    phase: SerializationPhase,
) -> crate::BuiltinResult<RootedValue> {
    let mut builder = RootedValuesBuilder::with_capacity(count);
    for child in children(results, count) {
        builder.push_owned(child);
    }
    builder
        .finish()
        .map_err(|error| error.into_runtime_error())?
        .with_owned_values(|values| {
            let cell = CellArray::new_with_shape(values, shape).map_err(|error| {
                serialization_error(
                    phase.method(),
                    &SERIALIZATION_ERROR_ARGUMENT,
                    format!(
                        "failed to rebuild {} cell payload: {error}",
                        phase.description()
                    ),
                )
            })?;
            root_value(Value::Cell(cell))
        })
}

pub(super) fn root_struct_array(
    names: Vec<String>,
    shape: Vec<usize>,
    count: usize,
    results: &mut Vec<RootedValue>,
    phase: SerializationPhase,
) -> crate::BuiltinResult<RootedValue> {
    let mut builder = RootedValuesBuilder::with_capacity(count);
    for child in children(results, count) {
        builder.push_owned(child);
    }
    builder
        .finish()
        .map_err(|error| error.into_runtime_error())?
        .with_owned_values(|values| {
            let value = runmat_value::StructArray::normalize_field_major(names, values, shape)
                .map_err(|error| {
                    serialization_error(
                        phase.method(),
                        &SERIALIZATION_ERROR_ARGUMENT,
                        format!(
                            "failed to rebuild {} structure array: {error}",
                            phase.description()
                        ),
                    )
                })?;
            root_value(value)
        })
}
