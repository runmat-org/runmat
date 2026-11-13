//! `drawnow` through the selected host graphics presentation authority.

use runmat_builtins::{
    DrawNowControl, DRAWNOW_ERROR_TOO_MANY_OUTPUTS, DRAWNOW_STATUS_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::{
    GraphicsCallbackPolicy, GraphicsCallbackReceipt, GraphicsFlushDisposition, GraphicsFlushRate,
    GraphicsFlushRequest, GraphicsPresentation, RuntimeContext,
};
use crate::BuiltinResult;

#[path = "drawnow/arguments.rs"]
mod arguments;
#[path = "drawnow/errors.rs"]
mod errors;
#[cfg(test)]
#[path = "drawnow/tests.rs"]
mod tests;

use arguments::parse_controls;
use errors::{attach_disposition, cancelled_flush, drawnow_error};

#[runtime_builtin(
    name = "drawnow",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::plotting::drawnow"
)]
pub async fn drawnow_builtin(
    runtime: Option<RuntimeContext>,
    requested_outputs: Option<usize>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let selected = runtime.clone();
    let operation = drawnow_in_context(runtime.as_ref(), requested_outputs, arguments);
    match selected.as_ref() {
        Some(runtime) => runtime.scope(operation).await,
        None => operation.await,
    }
}

async fn drawnow_in_context(
    runtime: Option<&RuntimeContext>,
    requested_outputs: Option<usize>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    match requested_outputs {
        Some(count) if count > 1 => {
            return Err(drawnow_error(
                &DRAWNOW_ERROR_TOO_MANY_OUTPUTS,
                "at most one RunMat status result can be requested",
            ));
        }
        Some(1) => {
            if let Some(runtime) = runtime {
                crate::compatibility::ensure_builtin_extension_enabled_in_context(
                    runtime,
                    &DRAWNOW_STATUS_OUTPUT_EXTENSION,
                    "drawnow",
                )?;
            } else {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &DRAWNOW_STATUS_OUTPUT_EXTENSION,
                    "drawnow",
                )?;
            }
        }
        _ => {}
    }

    let control = parse_controls(arguments.view())?;
    let runtime = runtime.ok_or_else(|| {
        crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::GraphicsPresentation,
            "drawnow",
        )
        .into_runtime_error()
    })?;
    let service = runtime
        .service_ports()
        .require_graphics_presentation("drawnow")
        .map_err(crate::context::RuntimeCapabilityError::into_runtime_error)?
        .clone();
    let request = request_for(control);
    let mut cooperation = runtime.cpu_cooperation();
    cooperation
        .checkpoint(0)
        .await
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;

    let (receipt, disposition) = crate::context::await_graphics_completion(
        runtime,
        service.flush(request),
        cancelled_flush,
        || {
            drawnow_error(
                &runmat_builtins::DRAWNOW_ERROR_PRESENTATION,
                "host completion returned no receipt",
            )
        },
    )
    .await
    .map_err(|mut error| {
        error.context = error.context.with_builtin("drawnow");
        error
    })?;
    if disposition != GraphicsFlushDisposition::Published
        && (receipt.presentation == GraphicsPresentation::Presented
            || receipt.callbacks == GraphicsCallbackReceipt::Processed)
    {
        return Err(attach_disposition(
            drawnow_error(
                &runmat_builtins::DRAWNOW_ERROR_PRESENTATION,
                "host claimed presentation or callbacks without a committed effect receipt",
            ),
            disposition,
        ));
    }
    let expected_callbacks = match request.callbacks {
        GraphicsCallbackPolicy::Process => GraphicsCallbackReceipt::Processed,
        GraphicsCallbackPolicy::Defer => GraphicsCallbackReceipt::Deferred,
    };
    if receipt.callbacks != expected_callbacks {
        return Err(attach_disposition(
            drawnow_error(
                &runmat_builtins::DRAWNOW_ERROR_PRESENTATION,
                "host did not honor the requested callback policy",
            ),
            disposition,
        ));
    }
    let impossible_deferral = matches!(
        (request.rate, receipt.presentation),
        (
            GraphicsFlushRate::Immediate,
            GraphicsPresentation::DeferredByRate | GraphicsPresentation::DeferredByBusy
        ) | (
            GraphicsFlushRate::WhenIdle,
            GraphicsPresentation::DeferredByRate
        )
    );
    if impossible_deferral {
        return Err(attach_disposition(
            drawnow_error(
                &runmat_builtins::DRAWNOW_ERROR_PRESENTATION,
                "host reported a deferral not permitted by the requested mode",
            ),
            disposition,
        ));
    }
    if requested_outputs == Some(0) {
        Ok(Value::output_list(Vec::new()))
    } else {
        // The historical direct Rust entrypoint returned a boolean. Public
        // MATLAB-compatible calls request zero outputs; one is a gated RunMat
        // status extension, including a rate-limited host completion.
        Ok(Value::Bool(true))
    }
}

fn request_for(control: DrawNowControl) -> GraphicsFlushRequest {
    match control {
        DrawNowControl::Full => GraphicsFlushRequest {
            rate: GraphicsFlushRate::Immediate,
            callbacks: GraphicsCallbackPolicy::Process,
        },
        DrawNowControl::LimitRate => GraphicsFlushRequest {
            rate: GraphicsFlushRate::LimitRate,
            callbacks: GraphicsCallbackPolicy::Process,
        },
        DrawNowControl::NoCallbacks | DrawNowControl::Expose => GraphicsFlushRequest {
            rate: GraphicsFlushRate::Immediate,
            callbacks: GraphicsCallbackPolicy::Defer,
        },
        DrawNowControl::LimitRateNoCallbacks => GraphicsFlushRequest {
            rate: GraphicsFlushRate::LimitRate,
            callbacks: GraphicsCallbackPolicy::Defer,
        },
        DrawNowControl::Update => GraphicsFlushRequest {
            rate: GraphicsFlushRate::WhenIdle,
            callbacks: GraphicsCallbackPolicy::Defer,
        },
    }
}
