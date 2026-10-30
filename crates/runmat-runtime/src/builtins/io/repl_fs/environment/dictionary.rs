use runmat_value::{CellArray, ObjectInstance, Value};

const KEYS: &str = "Keys";
const VALUES: &str = "Values";

pub(super) fn from_pairs(
    runtime: Option<&crate::context::RuntimeContext>,
    pairs: Vec<(String, String)>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::builtins::table::ensure_table_class_registered();
    let count = pairs.len();
    let (keys, values) = match runtime {
        Some(runtime) => {
            let keys = crate::execution::RootedValues::try_initialize_in_context(
                count,
                runtime,
                |index| {
                    crate::call::argument_text::copy_str(&pairs[index].0, Some(runtime))
                        .map(|text| Value::String(text.into()))
                },
            )?;
            let values = crate::execution::RootedValues::try_initialize_in_context(
                count,
                runtime,
                |index| {
                    crate::call::argument_text::copy_str(&pairs[index].1, Some(runtime))
                        .map(|text| Value::String(text.into()))
                },
            )?;
            (
                crate::execution::structure::cell::from_rooted_values(runtime, &[1, count], keys)?,
                crate::execution::structure::cell::from_rooted_values(
                    runtime,
                    &[1, count],
                    values,
                )?,
            )
        }
        None => {
            let keys = pairs
                .iter()
                .map(|(key, _)| Value::string(key.clone()))
                .collect();
            let values = pairs
                .into_iter()
                .map(|(_, value)| Value::string(value))
                .collect();
            (
                crate::execution::RootedValue::new(Value::Cell(
                    CellArray::new(keys, 1, count).map_err(dictionary_error)?,
                ))
                .map_err(|source| source.into_runtime_error())?,
                crate::execution::RootedValue::new(Value::Cell(
                    CellArray::new(values, 1, count).map_err(dictionary_error)?,
                ))
                .map_err(|source| source.into_runtime_error())?,
            )
        }
    };
    let fields = [
        (KEYS, keys.value().clone()),
        (VALUES, values.value().clone()),
    ];
    crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        runmat_types::standard::DICTIONARY.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
}

fn dictionary_error(message: String) -> crate::RuntimeError {
    crate::runtime_error::semantic_error("EnvironmentDictionary", message)
}

pub(super) fn entries(object: &ObjectInstance) -> Result<Vec<(Value, Value)>, ()> {
    if !object.is_class(runmat_types::standard::DICTIONARY) {
        return Err(());
    }
    let Some(Value::Cell(keys)) = object.properties().get(KEYS) else {
        return Err(());
    };
    let Some(Value::Cell(values)) = object.properties().get(VALUES) else {
        return Err(());
    };
    if keys.data().len() != values.data().len() {
        return Err(());
    }
    Ok(keys
        .data()
        .iter()
        .cloned()
        .zip(values.data().iter().cloned())
        .collect())
}
