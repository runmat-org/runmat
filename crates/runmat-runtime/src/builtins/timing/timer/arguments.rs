//! One typed timer name/value admission path for constructors and discovery.

use super::*;
use crate::builtins::common::gpu_helpers::gather_value_async;

async fn checkpoint_timer(runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    if let Some(runtime) = runtime {
        runtime
            .cpu_cooperation()
            .checkpoint(0)
            .await
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    }
    Ok(())
}

fn admit_numeric_extension(
    runtime: Option<&RuntimeContext>,
    extension: &runmat_builtins::BuiltinExtensionDescriptor,
    builtin: &str,
) -> BuiltinResult<()> {
    match runtime {
        Some(runtime) => crate::compatibility::ensure_builtin_extension_enabled_in_context(
            runtime, extension, builtin,
        ),
        None => crate::compatibility::ensure_builtin_extension_enabled(extension, builtin),
    }
}

/// Materialization uses the selected invocation's existing cancellation budget.
/// Provider errors remain intact; cancellation is checked before publication.
async fn gather_timer_value(
    runtime: Option<&RuntimeContext>,
    value: &Value,
) -> BuiltinResult<Value> {
    let Some(runtime) = runtime else {
        return gather_value_async(value).await;
    };
    let mut budget = runtime.cpu_cooperation();
    budget
        .checkpoint(0)
        .await
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    let result = budget
        .await_cancellable(gather_value_async(value))
        .await
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    budget
        .checkpoint(0)
        .await
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    result
}

pub(super) struct PreparedTimerPair {
    pub(super) property: TimerProperty,
    pub(super) value: Value,
}

/// Setter bindings already carry the catalog-owned property identity. Keep it
/// typed through resident admission and assignment instead of rebuilding a
/// name/value pair and parsing the spelling a second time.
pub(super) async fn prepare_timer_setter_value(
    runtime: Option<&RuntimeContext>,
    property: TimerProperty,
    value: Value,
) -> BuiltinResult<PreparedTimerPair> {
    checkpoint_timer(runtime).await?;
    if property.is_readonly() {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: property '{}' is read-only", property.name()),
        ));
    }
    if property.is_numeric() {
        if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
            admit_numeric_extension(
                runtime,
                &TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION,
                "timer",
            )?;
        }
        if !crate::dispatcher::value_contains_gpu(&value) {
            normalize_property_value(property, value.clone())?;
        }
        Ok(PreparedTimerPair {
            property,
            value: gather_timer_value(runtime, &value).await?,
        })
    } else {
        let value = normalize_property_value(property, value)?;
        checkpoint_timer(runtime).await?;
        Ok(PreparedTimerPair { property, value })
    }
}

pub(super) async fn prepare_timer_property_pairs(
    runtime: Option<&RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
    finding: bool,
) -> BuiltinResult<Vec<PreparedTimerPair>> {
    if !args.len().is_multiple_of(2) {
        let name = if finding { "timerfind" } else { "timer" };
        return Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            format!("{name}: property filters must appear in name-value pairs"),
        ));
    }
    // Admit every property and every resident value before initiating the
    // first provider operation. A late typo or read-only assignment must not
    // download an earlier numeric argument.
    let mut admitted = Vec::with_capacity(args.len() / 2);
    for pair in args.chunks_exact(2) {
        let property = canonical_property_name(&value_to_string(&pair[0])?)?;
        if !finding && property.is_readonly() {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: property '{}' is read-only", property.name()),
            ));
        }
        if property.is_numeric() {
            let has_gpu = crate::dispatcher::value_contains_gpu(&pair[1]);
            if crate::builtins::common::validation::value_contains_explicit_gpu(&pair[1]) {
                let extension = if finding {
                    &TIMERFIND_EXPLICIT_GPU_NUMERIC_FILTER_EXTENSION
                } else {
                    &TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION
                };
                admit_numeric_extension(
                    runtime,
                    extension,
                    if finding { "timerfind" } else { "timer" },
                )?;
            }
            if !has_gpu && !finding {
                normalize_property_value(property, pair[1].clone())?;
            }
        } else if !finding {
            normalize_property_value(property, pair[1].clone())?;
        }
        admitted.push((property, pair.get(1).expect("paired timer property value")));
    }
    let mut prepared = Vec::with_capacity(admitted.len());
    checkpoint_timer(runtime).await?;
    for (property, input) in admitted {
        let value = if property.is_numeric() {
            // Retain the provider's identifier, retry policy, source and span.
            gather_timer_value(runtime, input).await?
        } else {
            input.clone()
        };
        let value = if finding {
            value
        } else {
            normalize_property_value(property, value)?
        };
        prepared.push(PreparedTimerPair { property, value });
    }
    checkpoint_timer(runtime).await?;
    Ok(prepared)
}

pub(super) fn apply_name_value_pairs(
    runtime: Option<&RuntimeContext>,
    object: &mut ObjectInstance,
    args: &[PreparedTimerPair],
    constructor: bool,
) -> BuiltinResult<()> {
    for pair in args {
        let property = pair.property;
        if property.is_readonly() {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: property '{}' is read-only", property.name()),
            ));
        }
        if !constructor && running_is_on(object) && property.frozen_while_running() {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!(
                    "timer: property '{}' cannot be changed while Running is on",
                    property.name()
                ),
            ));
        }
    }
    crate::execution::object::replacement::replace_properties(
        runtime,
        object,
        args.iter().map(|pair| (pair.property.name(), &pair.value)),
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:timer:Allocation",
                message,
                source,
            )
        },
        |message| timer_error(&TIMER_ERROR_INVALID_PROPERTY, message),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_value_admission_keeps_typed_identity_through_assignment() {
        let pairs = futures::executor::block_on(prepare_timer_property_pairs(
            None,
            crate::call::FunctionArgumentView::from_slice(&[
                Value::String(" eXeCuTiOnMoDe ".into()),
                Value::String("FIXEDSPACING".into()),
                Value::String("period".into()),
                Value::Num(0.25),
            ]),
            false,
        ))
        .expect("typed timer options");
        assert_eq!(pairs[0].property, TimerProperty::ExecutionMode);
        assert_eq!(pairs[1].property, TimerProperty::Period);
        let mut object = default_timer_object("timer-test".into());
        apply_name_value_pairs(None, &mut object, &pairs, true).expect("apply typed options");
        assert_eq!(
            object.properties().get("ExecutionMode"),
            Some(&Value::String("fixedSpacing".into()))
        );
        assert_eq!(object.properties().get("Period"), Some(&Value::Num(0.25)));
    }

    #[test]
    fn late_invalid_property_is_rejected_before_numeric_materialization() {
        let error = futures::executor::block_on(prepare_timer_property_pairs(
            None,
            crate::call::FunctionArgumentView::from_slice(&[
                Value::String("Period".into()),
                Value::Num(0.25),
                Value::String("unknown".into()),
                Value::Num(1.0),
            ]),
            false,
        ))
        .err()
        .expect("invalid property");
        assert_eq!(error.identifier(), Some("RunMat:timer:InvalidProperty"));

        let error = futures::executor::block_on(prepare_timer_property_pairs(
            None,
            crate::call::FunctionArgumentView::from_slice(&[
                Value::String("Period".into()),
                Value::Num(0.25),
                Value::String("Running".into()),
                Value::String("on".into()),
            ]),
            false,
        ))
        .err()
        .expect("read-only property");
        assert_eq!(error.identifier(), Some("RunMat:timer:InvalidProperty"));
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "arguments/resident_tests.rs"]
mod resident_tests;
