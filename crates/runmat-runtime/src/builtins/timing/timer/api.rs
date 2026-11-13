//! Public timer identities and session-bound lifecycle entry points.

use super::*;

#[runtime_builtin(
    name = "timer",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    ensure_timer_class_registered();
    let args = prepare_timer_property_pairs(Some(&runtime), args.view(), false).await?;
    let name = runtime
        .timer_session()
        .borrow_mut()
        .next_name()
        .ok_or_else(|| timer_error(&TIMER_ERROR_INVALID_INPUT, "timer: name identity exhausted"))?;
    let object = prepare_default_timer_object(&runtime, name, &args)?;
    let (object, object_roots) = object.into_value_and_lease();
    let Value::Object(object) = object else {
        unreachable!("timer constructor publishes an object")
    };
    let Some(Value::String(runmat_value::StringElement::Text(visibility_text))) =
        object.properties().get("ObjectVisibility")
    else {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: invalid ObjectVisibility value",
        ));
    };
    let visibility = TimerVisibility::parse(visibility_text).ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: invalid ObjectVisibility value",
        )
    })?;
    let target = runmat_gc::gc_allocate(Value::Object(object))
        .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?;
    let root = runmat_gc::gc_root(target)
        .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?;
    let handle = HandleRef {
        class_name: TIMER_CLASS.into(),
        target,
        valid: true,
    };
    runtime
        .timer_session()
        .borrow_mut()
        .register(handle.clone(), root, visibility);
    drop(object_roots);
    Ok(Value::HandleObject(handle))
}

#[runtime_builtin(
    name = "timerfind",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timerfind_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    find_timers(
        &runtime,
        prepare_timer_property_pairs(Some(&runtime), args.view(), true).await?,
        false,
    )
}

#[runtime_builtin(
    name = "timerfindall",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timerfindall_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    find_timers(
        &runtime,
        prepare_timer_property_pairs(Some(&runtime), args.view(), true).await?,
        true,
    )
}

#[runtime_builtin(
    name = "__runmat_timer_start",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_start_builtin(
    runtime: Option<RuntimeContext>,
    requested_outputs: Option<usize>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    admit_timer_method_outputs(&runtime, requested_outputs, "start")?;
    start_timer_values(&runtime, &value, None).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_startat",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_startat_builtin(
    runtime: Option<RuntimeContext>,
    requested_outputs: Option<usize>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    admit_timer_method_outputs(&runtime, requested_outputs, "startat")?;
    if matches!(rest.len(), 2 | 4 | 5) || rest.len() > 6 {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            "startat: expected one firing time or Y/M/D[/H/MI/S] components",
        ));
    }
    if !rest.is_empty() && !is_legacy_relative_scalar(rest.view()) {
        start_at_absolute(&runtime, &value, rest).await?;
        return Ok(Value::Num(0.0));
    }
    crate::compatibility::ensure_builtin_extension_enabled_in_context(
        &runtime,
        &TIMER_STARTAT_RELATIVE_EXTENSION,
        "startat",
    )?;
    let prepared = match rest.first() {
        Some(delay) => {
            let delay_argument = [Value::string("StartDelay".to_string()), delay.clone()];
            prepare_timer_property_pairs(
                Some(&runtime),
                crate::call::FunctionArgumentView::from_slice(&delay_argument),
                false,
            )
            .await?
        }
        None => Vec::new(),
    };
    let delay = prepared
        .first()
        .map(|pair| numeric_scalar(&pair.value, "startat"))
        .transpose()?
        .filter(|seconds| *seconds > 0.0)
        .map(|seconds| lifecycle::duration_property(seconds, "StartDelay"))
        .transpose()?;
    start_timer_values(&runtime, &value, delay).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_stop",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_stop_builtin(
    runtime: Option<RuntimeContext>,
    requested_outputs: Option<usize>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    admit_timer_method_outputs(&runtime, requested_outputs, "stop")?;
    stop_timer_values(&runtime, &value, true).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_wait",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_wait_builtin(
    runtime: Option<RuntimeContext>,
    requested_outputs: Option<usize>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    admit_timer_method_outputs(&runtime, requested_outputs, "wait")?;
    wait_timer_values(&runtime, &value).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_delete",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_delete_builtin(
    runtime: Option<RuntimeContext>,
    requested_outputs: Option<usize>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    admit_timer_method_outputs(&runtime, requested_outputs, "delete")?;
    delete_timer_values(&runtime, &value).await?;
    Ok(Value::Num(0.0))
}
