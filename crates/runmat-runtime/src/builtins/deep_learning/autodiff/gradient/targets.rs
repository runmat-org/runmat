//! Borrowed target admission before reverse execution or gradient allocation.
use super::*;

pub(super) fn validate(target: &Value) -> BuiltinResult<()> {
    match target {
        Value::Object(object) if object.class_name.is(DLARRAY_CLASS) => {
            validate_host_tape_dlarray_data(target)?;
            if dlarray_node_id(target).is_none() {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: target dlarray was not traced inside the active dlfeval",
                ));
            }
            Ok(())
        }
        Value::Object(object) if model::is_deep_learning_network_object(object) => {
            for row in model::parameters::view(object, "dlgradient")? {
                let value = row.value.ok_or_else(|| {
                    deep_learning_error(
                        "dlgradient",
                        "dlgradient: network target has an uninitialized learnable",
                    )
                })?;
                validate(value)?;
            }
            Ok(())
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
            if let Some(values) = parameter_table::values(object, "dlgradient")? {
                return validate_values(values.data.iter());
            }
            let variables = crate::builtins::table::table_variables_ref(object)
                .map_err(|error| deep_learning_error("dlgradient", error.to_string()))?
                .ok_or_else(|| {
                    deep_learning_error("dlgradient", "dlgradient: table is missing Variables")
                })?;
            validate_values(variables.fields.values())
        }
        Value::Struct(value) if is_learnables_struct(value) => {
            // The legacy structural form explicitly separates row metadata
            // from its derivative-bearing Value subtree.
            validate(value.fields.get("Value").expect("admitted structural Value"))
        }
        Value::Struct(value) => validate_values(value.fields.values()),
        Value::StructArray(array) => {
            for element in array.elements() {
                validate_values(element.fields.values())?;
            }
            Ok(())
        }
        Value::Cell(values) => validate_values(values.data.iter()),
        other => Err(deep_learning_error(
            "dlgradient",
            format!("dlgradient: target must be a traced dlarray, learnables tree, or dlnetwork, got {other:?}"),
        )),
    }
}

fn validate_values<'a>(values: impl Iterator<Item = &'a Value>) -> BuiltinResult<()> {
    for value in values {
        validate(value)?;
    }
    Ok(())
}
