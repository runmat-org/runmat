//! MATLAB-compatible `timeit` builtin for RunMat.
//!
//! Measures the execution time of zero-input function handles by running them
//! repeatedly and returning the median per-invocation runtime in seconds.

use crate::builtins::common::gpu_helpers::gather_value_async;
use crate::builtins::common::tensor;
use crate::context::RuntimeContext;
use runmat_builtins::{
    BuiltinErrorDescriptor, EXPLICIT_GPU_NUM_OUTPUTS_EXTENSION, INTEGER_NUM_OUTPUTS_EXTENSION,
    TIMEIT_ERROR_CONTEXT_UNAVAILABLE, TIMEIT_ERROR_EMPTY_HANDLE,
    TIMEIT_ERROR_EXPECTS_AT_HANDLE_STRING, TIMEIT_ERROR_FIRST_ARG_KIND, TIMEIT_ERROR_HANDLE_KIND,
    TIMEIT_ERROR_NUM_OUTPUTS_FINITE, TIMEIT_ERROR_NUM_OUTPUTS_INTEGER,
    TIMEIT_ERROR_NUM_OUTPUTS_NONNEG, TIMEIT_ERROR_NUM_OUTPUTS_SCALAR, TIMEIT_ERROR_TOO_MANY_INPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

mod measurement;
use measurement::{collect_samples, compute_median, determine_loop_count};

const TARGET_BATCH_SECONDS: f64 = 0.005;
const MAX_BATCH_SECONDS: f64 = 0.25;
const LOOP_COUNT_LIMIT: usize = 1 << 20;
const MIN_SAMPLE_COUNT: usize = 7;
const MAX_SAMPLE_COUNT: usize = 21;
const BUILTIN_NAME: &str = "timeit";

fn timeit_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "timeit",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::timing::timeit"
)]
async fn timeit_builtin(
    runtime: Option<RuntimeContext>,
    func: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let runtime = runtime.ok_or_else(|| {
        timeit_error_with_message(
            TIMEIT_ERROR_CONTEXT_UNAVAILABLE.message,
            &TIMEIT_ERROR_CONTEXT_UNAVAILABLE,
        )
    })?;
    let requested_outputs = parse_num_outputs(&runtime, &func, rest.view()).await?;
    let callable = {
        let _context = runtime.enter();
        prepare_callable(func, requested_outputs)?
    };

    // Warm-up once to catch early errors and pay one-time JIT costs.
    runtime
        .cpu_cooperation()
        .await_cancellable(callable.invoke(&runtime))
        .await
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)??;

    let loop_count = determine_loop_count(&callable, &runtime).await?;
    let samples = collect_samples(&callable, &runtime, loop_count).await?;
    if samples.is_empty() {
        return Ok(Value::Num(0.0));
    }

    Ok(Value::Num(compute_median(samples)))
}

async fn parse_num_outputs(
    runtime: &RuntimeContext,
    func: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> Result<Option<usize>, crate::RuntimeError> {
    match rest.len() {
        0 => Ok(None),
        1 => {
            let value = &rest[0];
            if crate::builtins::common::validation::value_contains_explicit_gpu(value) {
                crate::compatibility::ensure_builtin_extension_enabled_in_context(
                    runtime,
                    &EXPLICIT_GPU_NUM_OUTPUTS_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if crate::builtins::common::validation::value_contains_native_integer_class(value) {
                crate::compatibility::ensure_builtin_extension_enabled_in_context(
                    runtime,
                    &INTEGER_NUM_OUTPUTS_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            // Host scalar validation keeps its historical count-before-callable
            // error order. A resident scalar must be materialized, so reject an
            // invalid first argument before touching an acceleration provider.
            if !matches!(value, Value::GpuTensor(_)) {
                return parse_non_negative_integer(value).map(Some);
            }
            {
                let _context = runtime.enter();
                prepare_callable(func.clone(), None)?;
            }
            let value = runtime
                .cpu_cooperation()
                .await_cancellable(gather_value_async(value))
                .await
                .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?
                .map_err(|error| {
                    timeit_error_with_message(error.message(), &TIMEIT_ERROR_NUM_OUTPUTS_SCALAR)
                })?;
            parse_non_negative_integer(&value).map(Some)
        }
        _ => Err(timeit_error_with_message(
            TIMEIT_ERROR_TOO_MANY_INPUTS.message,
            &TIMEIT_ERROR_TOO_MANY_INPUTS,
        )),
    }
}

fn parse_non_negative_integer(value: &Value) -> Result<usize, crate::RuntimeError> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            timeit_error_with_message(
                TIMEIT_ERROR_NUM_OUTPUTS_NONNEG.message,
                &TIMEIT_ERROR_NUM_OUTPUTS_NONNEG,
            )
        });
    }
    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(timeit_error_with_message(
                    TIMEIT_ERROR_NUM_OUTPUTS_FINITE.message,
                    &TIMEIT_ERROR_NUM_OUTPUTS_FINITE,
                ));
            }
            if *n < 0.0 {
                return Err(timeit_error_with_message(
                    TIMEIT_ERROR_NUM_OUTPUTS_NONNEG.message,
                    &TIMEIT_ERROR_NUM_OUTPUTS_NONNEG,
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(timeit_error_with_message(
                    TIMEIT_ERROR_NUM_OUTPUTS_INTEGER.message,
                    &TIMEIT_ERROR_NUM_OUTPUTS_INTEGER,
                ));
            }
            if !fits_platform_usize(rounded) {
                return Err(timeit_error_with_message(
                    TIMEIT_ERROR_NUM_OUTPUTS_NONNEG.message,
                    &TIMEIT_ERROR_NUM_OUTPUTS_NONNEG,
                ));
            }
            Ok(rounded as usize)
        }
        _ => Err(timeit_error_with_message(
            TIMEIT_ERROR_NUM_OUTPUTS_SCALAR.message,
            &TIMEIT_ERROR_NUM_OUTPUTS_SCALAR,
        )),
    }
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}

#[derive(Clone, Debug)]
struct TimeitCallable {
    handle: Value,
    num_outputs: Option<usize>,
}

impl TimeitCallable {
    async fn invoke(&self, runtime: &RuntimeContext) -> Result<Value, crate::RuntimeError> {
        let requested_outputs = self.num_outputs.unwrap_or(1);
        let value = crate::call_feval_async_with_outputs_in_context(
            runtime,
            self.handle.clone(),
            &[],
            requested_outputs,
        )
        .await?;
        drop(value);
        Ok(Value::Num(0.0))
    }
}

fn prepare_callable(
    func: Value,
    num_outputs: Option<usize>,
) -> Result<TimeitCallable, crate::RuntimeError> {
    fn normalize_name(name: &str) -> Result<String, crate::RuntimeError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            Err(timeit_error_with_message(
                TIMEIT_ERROR_EMPTY_HANDLE.message,
                &TIMEIT_ERROR_EMPTY_HANDLE,
            ))
        } else {
            Ok(trimmed.to_string())
        }
    }

    fn canonicalize_text_handle(handle: String) -> Value {
        let name = handle.strip_prefix('@').unwrap_or(handle.as_str());
        handle_for_name(name).unwrap_or(Value::string(handle))
    }

    match func {
        Value::String(runmat_value::StringElement::Text(text)) => {
            parse_handle_string(&text).map(|handle| TimeitCallable {
                handle: canonicalize_text_handle(handle),
                num_outputs,
            })
        }
        Value::CharArray(arr) => {
            if arr.rows != 1 {
                Err(timeit_error_with_message(
                    TIMEIT_ERROR_HANDLE_KIND.message,
                    &TIMEIT_ERROR_HANDLE_KIND,
                ))
            } else {
                let text =
                    crate::builtins::common::text::decode_utf16_with_error(arr.data(), || {
                        timeit_error_with_message(
                            "timeit: function text is not valid UTF-16",
                            &TIMEIT_ERROR_HANDLE_KIND,
                        )
                    })?;
                parse_handle_string(&text).map(|handle| TimeitCallable {
                    handle: canonicalize_text_handle(handle),
                    num_outputs,
                })
            }
        }
        Value::StringArray(sa) => {
            if sa.data().len() == 1 {
                parse_handle_string(sa.data()[0].as_str().ok_or_else(|| {
                    timeit_error_with_message(
                        "timeit: function must be a string scalar",
                        &TIMEIT_ERROR_HANDLE_KIND,
                    )
                })?)
                .map(|handle| TimeitCallable {
                    handle: canonicalize_text_handle(handle),
                    num_outputs,
                })
            } else {
                Err(timeit_error_with_message(
                    TIMEIT_ERROR_HANDLE_KIND.message,
                    &TIMEIT_ERROR_HANDLE_KIND,
                ))
            }
        }
        Value::ScopedFunctionHandle(handle) => Ok(TimeitCallable {
            handle: Value::ScopedFunctionHandle(handle),
            num_outputs,
        }),
        Value::FunctionHandle(name) => {
            let normalized = normalize_name(&name)?;
            Ok(TimeitCallable {
                handle: handle_for_name(&normalized)
                    .unwrap_or_else(|| Value::string(format!("@{normalized}"))),
                num_outputs,
            })
        }
        Value::ExternalFunctionHandle(name) => {
            let normalized = normalize_name(&name)?;
            Ok(TimeitCallable {
                handle: if crate::is_well_formed_qualified_name(&normalized) {
                    handle_for_name(&normalized)
                        .unwrap_or_else(|| Value::external_function_handle(normalized))
                } else {
                    Value::external_function_handle(normalized)
                },
                num_outputs,
            })
        }
        Value::BoundFunctionHandle(handle) => {
            let normalized = normalize_name(&handle.name)?;
            let mut data = handle.into_data();
            data.name = normalized.into();
            Ok(TimeitCallable {
                handle: Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                    data,
                )),
                num_outputs,
            })
        }
        Value::Closure(closure) => Ok(TimeitCallable {
            handle: {
                if closure.binding.is_none() {
                    if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(
                        &closure.function_name,
                    ) {
                        return Ok(TimeitCallable {
                            handle: Value::Closure(
                                crate::call::closure_binding::bind_selected_closure(
                                    closure, function,
                                )?,
                            ),
                            num_outputs,
                        });
                    }
                }
                Value::Closure(closure)
            },
            num_outputs,
        }),
        other => Err(timeit_error_with_message(
            format!("timeit: first argument must be a function handle, got {other:?}"),
            &TIMEIT_ERROR_FIRST_ARG_KIND,
        )),
    }
}

fn handle_for_name(name: &str) -> Option<Value> {
    let function = crate::user_functions::resolve_semantic_function_by_name(name)?;
    Some(crate::user_functions::bound_function_handle(
        name.into(),
        function,
    ))
}

fn parse_handle_string(text: &str) -> Result<String, crate::RuntimeError> {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix('@') {
        if rest.trim().is_empty() {
            Err(timeit_error_with_message(
                TIMEIT_ERROR_EMPTY_HANDLE.message,
                &TIMEIT_ERROR_EMPTY_HANDLE,
            ))
        } else {
            Ok(format!("@{}", rest.trim()))
        }
    } else {
        Err(timeit_error_with_message(
            TIMEIT_ERROR_EXPECTS_AT_HANDLE_STRING.message,
            &TIMEIT_ERROR_EXPECTS_AT_HANDLE_STRING,
        ))
    }
}

#[cfg(test)]
#[path = "timeit/tests.rs"]
pub(crate) mod tests;
