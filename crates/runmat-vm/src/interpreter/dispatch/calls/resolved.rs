use super::*;

pub(super) async fn bound(
    ctx: UserCallContext<'_>,
    args: Vec<Value>,
    refresh_vars: impl Fn(&[Value]),
) -> Result<UserCallHandling, RuntimeError> {
    let UserCallContext {
        stack,
        identity,
        fallback_policy,
        out_count,
        source_id,
        call_arg_spans,
        current_function_name,
        imports: _,
        exception,
    } = ctx;
    let current_class_context =
        runmat_runtime::class_registry::class_context_for_function(current_function_name);
    let _function_input_callsite_guard =
        runmat_runtime::callsite::push_function_input_callsite(source_id, call_arg_spans);
    let descriptor = CallableDescriptor::resolved(
        identity,
        args,
        out_count,
        fallback_policy,
        CallableCallKind::Direct,
    );
    execute(
        stack,
        descriptor,
        out_count,
        exception,
        current_class_context,
        refresh_vars,
    )
    .await
}

pub(super) async fn execute(
    stack: &mut Vec<Value>,
    descriptor: CallableDescriptor,
    out_count: usize,
    exception: ExceptionRouteContext<'_>,
    current_class_context: Option<String>,
    refresh_vars: impl Fn(&[Value]),
) -> Result<UserCallHandling, RuntimeError> {
    let ExceptionRouteContext {
        try_stack,
        vars,
        last_exception,
        pc,
    } = exception;
    let _access_guard = current_class_context
        .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)));
    match execute_callable_descriptor(descriptor).await {
        Ok(result) => {
            stack.push(normalize_requested_outputs(result, out_count));
            Ok(UserCallHandling::Completed)
        }
        Err(err) => Ok(
            match redirect_exception_to_catch(
                err,
                try_stack,
                vars,
                last_exception,
                pc,
                refresh_vars,
            ) {
                ExceptionHandling::Caught => UserCallHandling::Caught,
                ExceptionHandling::Uncaught(err) => UserCallHandling::Uncaught(err),
            },
        ),
    }
}
