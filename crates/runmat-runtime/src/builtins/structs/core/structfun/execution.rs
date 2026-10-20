use super::{
    callback::Callable,
    error_context,
    fields::Fields,
    options::Options,
    output::{self, Collector, Slots},
};
use crate::context::RuntimeContext;
use crate::execution::{
    RootedValue, RootedValueChildren, RootedValueLease, RootedValueMember, RootedValues,
    RootedValuesBuilder,
};
use crate::{current_requested_outputs, gather_if_needed_async, BuiltinResult};
use runmat_value::Value;

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    function: Value,
    structure: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _inputs_root = match runtime {
        Some(runtime) => RootedValueLease::for_value_slices_in_context(
            &[
                std::slice::from_ref(&function),
                std::slice::from_ref(&structure),
            ],
            runtime,
        )?,
        None => {
            output::ordinary_policy()?;
            RootedValueLease::for_values([&function, &structure].into_iter())
                .map_err(|e| e.into_runtime_error())?
        }
    };
    let callable = Callable::parse(function)?;
    let fields = Fields::parse(structure)?;
    // Option parsing consumes projections, not their original registration.
    // Keep that exact source forest through every callback and publication.
    let _option_sources = rest.fork();
    let options = Options::parse(rest)?;
    let requested = current_requested_outputs();
    let mut collectors = Slots::new(requested, runtime, || {
        Collector::new(
            options.uniform,
            fields.iter().map(|(name, _)| name),
            runtime,
        )
    })?;
    for (index, (field, value)) in fields.iter().enumerate() {
        let value = output::root(gather_if_needed_async(value).await?, runtime)?;
        let result = match callable
            .call(runtime, std::slice::from_ref(value.value()), requested)
            .await
        {
            Ok(result) => result,
            Err(callback_error) => {
                if let super::callback::CallbackFailure::Runtime(error) = &callback_error {
                    if super::callback::is_control_failure(error) {
                        return Err(callback_error.into_runtime_error());
                    }
                }
                let Some(handler) = &options.handler else {
                    return Err(callback_error.into_runtime_error());
                };
                let context = error_context::value(&callback_error, index, field, runtime)?;
                // Both sources remain rooted through semantic dispatch, fallback,
                // and any provider retry. Only the fixed stack argument slots copy.
                let arguments = [context.value().clone(), value.value().clone()];
                handler
                    .call(runtime, &arguments, requested)
                    .await
                    .map_err(super::callback::CallbackFailure::into_runtime_error)?
            }
        };
        if requested == 0 {
            continue;
        } // Preserve callback/error-handler effects.
        let results = match result.value() {
            Value::OutputList(values) if values.len() == requested => values.as_slice(),
            Value::OutputList(values) => {
                return Err(super::error::function(format!(
                    "structfun: callback returned {} outputs but {requested} were requested",
                    values.len()
                )))
            }
            value if requested == 1 => std::slice::from_ref(value),
            _ => {
                return Err(super::error::function(
                    "structfun: callback did not return the requested number of outputs",
                ))
            }
        };
        // Borrow even shared callback output storage. The original result lease
        // stays live until each gathered successor is independently registered.
        for (collector, result) in collectors.values_mut()?.iter_mut().zip(results) {
            let gathered = output::root(gather_if_needed_async(result).await?, runtime)?;
            collector
                .as_mut()
                .expect("active collector")
                .push(field, gathered)?;
        }
    }
    if let Some(runtime) = runtime {
        let mut children = RootedValueChildren::in_context(runtime)
            .map_err(|e| output::allocation(Some(runtime), e))?;
        let outputs = RootedValues::try_initialize_in_context(requested, runtime, |index| {
            let collector = collectors.values_mut()?[index]
                .take()
                .expect("active collector");
            let value = collector.finish(fields.len(), Some(runtime))?;
            RootedValueMember::from_rooted(value)
                .into_child(&mut children)
                .map_err(|e| output::allocation(Some(runtime), e))
        })?;
        // children owns every moved source throughout final list registration.
        let result = publish(outputs, requested, Some(runtime));
        drop(children);
        result
    } else {
        output::ordinary_policy()?;
        let mut outputs = RootedValuesBuilder::with_capacity(requested);
        for collector in collectors.values_mut()? {
            outputs.push_owned(
                collector
                    .take()
                    .expect("active collector")
                    .finish(fields.len(), None)?,
            );
        }
        publish(
            outputs.finish().map_err(|e| e.into_runtime_error())?,
            requested,
            None,
        )
    }
}
fn publish(
    outputs: RootedValues,
    requested: usize,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<RootedValue> {
    outputs.with_owned_output_storage(|values| {
        let value = if requested == 1 {
            values
                .try_into_first()
                .map_err(|_| super::error::internal("structfun: final output owner was aliased"))?
                .expect("one requested output")
        } else {
            Value::OutputList(values)
        };
        output::root(value, runtime)
    })
}
