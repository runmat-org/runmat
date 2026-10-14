use miette::SourceSpan;
use runmat_runtime::{CallFrame, RuntimeError};

pub fn callstack_limit() -> usize {
    runmat_runtime::context::legacy::active().map_or(
        runmat_runtime::context::DEFAULT_CALLSTACK_LIMIT,
        |context| context.callstack_limit(),
    )
}

pub fn attach_call_frames(
    current_function_name: &str,
    source_id: Option<runmat_types::SourceId>,
    source_owner: Option<runmat_runtime::CallFrameSourceOwner>,
    diagnostic_source: Option<runmat_runtime::CallFrameDiagnosticSource>,
    mut err: RuntimeError,
) -> RuntimeError {
    if !err.context.call_frames.is_empty() || !err.context.call_stack.is_empty() {
        return err;
    }
    let limit = callstack_limit();
    if limit == 0 {
        return err;
    }
    let span = err.span.as_ref().map(|span: &SourceSpan| {
        let start = span.offset();
        let end = start + span.len();
        (start, end)
    });
    if span.is_some() || !current_function_name.is_empty() {
        err.context.call_frames.push(CallFrame {
            function: current_function_name.to_string(),
            source_id: source_id.map(|id| id.0),
            span,
            source_owner,
            diagnostic_source,
        });
    }
    err
}
