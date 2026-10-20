use super::*;

pub(super) fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data().iter().any(contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(contains_explicit_gpu),
        Value::Object(value) => value.properties().values().any(contains_explicit_gpu),
        Value::Closure(value) => value.captures.iter().any(contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(contains_explicit_gpu),
        _ => false,
    }
}

pub(super) fn admit_options(
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Vec<OptionInput>> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "lasso: name-value options must be supplied in pairs",
        ));
    }
    let mut admitted = Vec::with_capacity(rest.len() / 2);
    let mut args = rest.into_iter();
    while let Some(name_value) = args.next() {
        let name = scalar_text(&name_value, "option name")?;
        let option = LassoOption::from_public_name(&name)
            .ok_or_else(|| invalid_argument(format!("lasso: unsupported option '{name}'")))?;
        let value = args.next().expect("preflight verified name-value pairs");
        if option == LassoOption::PredictorNames {
            string_list(&value, "PredictorNames")?;
        }
        admitted.push(OptionInput { option, value });
    }
    Ok(admitted)
}

pub(super) fn ensure_extensions(x: &Value, y: &Value, rest: &[OptionInput]) -> BuiltinResult<()> {
    if is_typed_integer(x) || is_typed_integer(y) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LASSO_INTEGER_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    for input in rest {
        if !is_typed_integer(&input.value) {
            continue;
        }
        let extension = match input.option.integer_role() {
            LassoOptionRole::StructuralControl => &LASSO_INTEGER_CONTROL_EXTENSION,
            LassoOptionRole::BooleanControl => &LASSO_INTEGER_BOOLEAN_EXTENSION,
            LassoOptionRole::FloatingParameter => &LASSO_INTEGER_PARAMETER_EXTENSION,
        };
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
    }
    if contains_explicit_gpu(x)
        || contains_explicit_gpu(y)
        || rest.iter().any(|input| contains_explicit_gpu(&input.value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LASSO_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

pub(super) async fn gathered(value: Value) -> BuiltinResult<Value> {
    if !crate::dispatcher::value_contains_gpu(&value) {
        return Ok(value);
    }
    gather_if_needed_async(&value).await.map_err(|mut error| {
        if error.context.builtin.is_none() {
            error.context = error.context.with_builtin(BUILTIN_NAME);
        }
        error
    })
}

pub(super) async fn gather_options(values: Vec<OptionInput>) -> BuiltinResult<Vec<OptionInput>> {
    let mut out = Vec::with_capacity(values.len());
    for input in values {
        out.push(OptionInput {
            option: input.option,
            value: gathered(input.value).await?,
        });
    }
    Ok(out)
}
