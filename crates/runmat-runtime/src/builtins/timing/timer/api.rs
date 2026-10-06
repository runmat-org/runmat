//! Public timer identities and session-bound lifecycle entry points.

use super::*;

#[runtime_builtin(
    name = "timer",
    category = "timing",
    summary = "Create a timer handle object for scheduled callbacks.",
    keywords = "timer,callback,start,stop,wait,timerfind",
    descriptor(crate::builtins::timing::timer::TIMER_DESCRIPTOR),
    extensions(crate::builtins::timing::timer::TIMER_EXTENSIONS),
    integer_capabilities(crate::builtins::timing::timer::TIMER_INTEGER_CAPABILITIES),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_builtin(
    runtime: Option<RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    ensure_timer_class_registered();
    let args = prepare_timer_property_pairs(&args, false).await?;
    let name = runtime
        .timer_session()
        .borrow_mut()
        .next_name()
        .ok_or_else(|| timer_error(&TIMER_ERROR_INVALID_INPUT, "timer: name identity exhausted"))?;
    let mut object = default_timer_object(name);
    apply_name_value_pairs(&mut object, &args, true)?;
    let Some(Value::String(visibility_text)) = object.properties.get("ObjectVisibility") else {
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
    runmat_gc::gc_add_root(target)
        .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?;
    let handle = HandleRef {
        class_name: TIMER_CLASS.into(),
        target,
        valid: true,
    };
    runtime
        .timer_session()
        .borrow_mut()
        .register(handle.clone(), visibility);
    Ok(Value::HandleObject(handle))
}

#[runtime_builtin(
    name = "timerfind",
    category = "timing",
    summary = "Find visible timer objects by property value.",
    keywords = "timerfind,timer,callback,handle",
    descriptor(crate::builtins::timing::timer::TIMERFIND_DESCRIPTOR),
    extensions(crate::builtins::timing::timer::TIMERFIND_EXTENSIONS),
    integer_capabilities(crate::builtins::timing::timer::TIMERFIND_INTEGER_CAPABILITIES),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timerfind_builtin(
    runtime: Option<RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    find_timers(
        &runtime,
        prepare_timer_property_pairs(&args, true).await?,
        false,
    )
}

#[runtime_builtin(
    name = "timerfindall",
    category = "timing",
    summary = "Find all timer objects by property value.",
    keywords = "timerfindall,timer,callback,handle",
    descriptor(crate::builtins::timing::timer::TIMERFINDALL_DESCRIPTOR),
    extensions(crate::builtins::timing::timer::TIMERFIND_EXTENSIONS),
    integer_capabilities(crate::builtins::timing::timer::TIMERFIND_INTEGER_CAPABILITIES),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timerfindall_builtin(
    runtime: Option<RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    find_timers(
        &runtime,
        prepare_timer_property_pairs(&args, true).await?,
        true,
    )
}

#[runtime_builtin(
    name = "__runmat_timer_start",
    execution_stack = "process",
    category = "timing",
    summary = "Start a timer object.",
    keywords = "timer,start",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_start_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    start_timer_values(&runtime, &value, None).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_startat",
    execution_stack = "process",
    category = "timing",
    summary = "Start a timer object at a requested time.",
    keywords = "timer,startat",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_startat_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    let prepared = match rest.first() {
        Some(delay) => {
            prepare_timer_property_pairs(
                &[Value::String("StartDelay".to_string()), delay.clone()],
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
        .filter(|seconds| *seconds > 0.0);
    start_timer_values(&runtime, &value, delay).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_stop",
    execution_stack = "process",
    category = "timing",
    summary = "Stop a timer object.",
    keywords = "timer,stop",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_stop_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    stop_timer_values(&runtime, &value, true).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_wait",
    execution_stack = "process",
    category = "timing",
    summary = "Wait until timer objects finish.",
    keywords = "timer,wait",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_wait_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    wait_timer_values(&runtime, &value).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_delete",
    execution_stack = "process",
    category = "timing",
    summary = "Delete a timer object.",
    keywords = "timer,delete",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_delete_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    delete_timer_values(&runtime, &value).await?;
    Ok(Value::Num(0.0))
}
