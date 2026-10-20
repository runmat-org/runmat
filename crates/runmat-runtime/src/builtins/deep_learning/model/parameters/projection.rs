//! User-facing Learnables table derived from canonical layer values.
use super::*;

pub(in crate::builtins::deep_learning::model) fn public_value(
    value: Option<&Value>,
) -> BuiltinResult<Value> {
    let Some(value) = value else {
        return Ok(Value::Tensor(
            Tensor::new(vec![], vec![0, 0]).expect("empty parameter shape"),
        ));
    };
    if matches!(value, Value::Object(object) if object.class_name().is(super::DLARRAY_CLASS)) {
        return Ok(value.clone());
    }
    if empty_parameter(value) {
        return Ok(value.clone());
    }
    super::super::super::class_registration::ensure_dlarray_class_registered();
    super::super::super::object_with_identity(
        super::DLARRAY_CLASS,
        [
            ("Data", value.clone()),
            ("Format", Value::String("".into())),
            ("Labels", Value::String("".into())),
        ],
    )
}

pub(in crate::builtins) fn project(
    network: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Value> {
    let parameters = view(network, function)?;
    let rows = parameters.len();
    let mut names = Vec::with_capacity(rows);
    let mut roles = Vec::with_capacity(rows);
    let mut values = Vec::with_capacity(rows);
    for parameter in parameters {
        names.push(parameter.key.layer_name);
        roles.push(parameter.key.role.name().to_string());
        values.push(public_value(parameter.value)?);
    }
    let value_column = CellArray::new(values, rows, 1)
        .map(Value::Cell)
        .map_err(|error| deep_learning_error(function, error))?;
    crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![
            string_array(names, vec![rows, 1], function)?,
            string_array(roles, vec![rows, 1], function)?,
            value_column,
        ],
    )
    .map_err(|error| deep_learning_error(function, error.to_string()))
}
