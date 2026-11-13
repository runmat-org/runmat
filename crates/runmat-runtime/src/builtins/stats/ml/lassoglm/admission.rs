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
        return Err(invalid(
            "lassoglm: name-value options must be supplied in pairs",
        ));
    }
    let mut admitted = Vec::with_capacity(rest.len() / 2);
    let mut args = rest.into_iter();
    while let Some(name_value) = args.next() {
        let name = scalar_text(&name_value, "option name")?;
        let option = LassoglmOption::from_public_name(&name)
            .ok_or_else(|| invalid(format!("lassoglm: unsupported option '{name}'")))?;
        let value = args.next().expect("preflight verified name-value pairs");
        match option {
            LassoglmOption::Link => {
                scalar_text(&value, "Link")?;
            }
            LassoglmOption::EstDisp => {
                scalar_text(&value, "EstDisp")?;
            }
            LassoglmOption::PredictorNames => {
                string_list(&value, "PredictorNames")?;
            }
            LassoglmOption::Options
                if !matches!(&value, Value::Struct(_)) && !is_empty_numeric(&value) =>
            {
                return Err(invalid("lassoglm: Options must be [] or a statset struct"));
            }
            _ => {}
        }
        admitted.push(OptionInput {
            option,
            name: name.into_owned(),
            value,
        });
    }
    Ok(admitted)
}

pub(super) fn ensure_integer_extension(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if is_typed_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)?;
    }
    Ok(())
}

pub(super) fn ensure_extensions(
    x: &Value,
    y: &Value,
    distr: &Value,
    rest: &[OptionInput],
) -> BuiltinResult<()> {
    if is_typed_integer(x) || is_typed_integer(y) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LASSOGLM_INTEGER_DATA_EXTENSION,
            NAME,
        )?;
    }
    for input in rest {
        let value = &input.value;
        match input.option.integer_role() {
            LassoglmOptionRole::StructuralControl => {
                ensure_integer_extension(value, &LASSOGLM_INTEGER_CONTROL_EXTENSION)?
            }
            LassoglmOptionRole::BooleanControl => {
                ensure_integer_extension(value, &LASSOGLM_INTEGER_BOOLEAN_EXTENSION)?
            }
            LassoglmOptionRole::FloatingParameter => {
                ensure_integer_extension(value, &LASSOGLM_INTEGER_PARAMETER_EXTENSION)?
            }
            LassoglmOptionRole::Aggregate => {
                if let Value::Struct(st) = value {
                    if let Some(value) = nonempty_field(st, StatOptionField::MaxIter) {
                        ensure_integer_extension(value, &LASSOGLM_INTEGER_CONTROL_EXTENSION)?;
                    }
                    if let Some(value) = nonempty_field(st, StatOptionField::TolX)
                        .or_else(|| nonempty_field(st, StatOptionField::TolFun))
                    {
                        ensure_integer_extension(value, &LASSOGLM_INTEGER_PARAMETER_EXTENSION)?;
                    }
                    if let Some(value) = nonempty_field(st, StatOptionField::UseParallel) {
                        ensure_integer_extension(value, &LASSOGLM_INTEGER_BOOLEAN_EXTENSION)?;
                    }
                }
            }
        }
    }
    if contains_explicit_gpu(x)
        || contains_explicit_gpu(y)
        || contains_explicit_gpu(distr)
        || rest.iter().any(|input| contains_explicit_gpu(&input.value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LASSOGLM_RESIDENT_INPUT_EXTENSION,
            NAME,
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
            error.context = error.context.with_builtin(NAME);
        }
        error
    })
}

pub(super) async fn gather_options(values: Vec<OptionInput>) -> BuiltinResult<Vec<OptionInput>> {
    let mut out = Vec::with_capacity(values.len());
    for input in values {
        out.push(OptionInput {
            option: input.option,
            name: input.name,
            value: gathered(input.value).await?,
        });
    }
    Ok(out)
}
