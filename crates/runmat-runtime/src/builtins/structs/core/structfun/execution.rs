use crate::execution::{RootedValue, RootedValueLease, RootedValuesBuilder};
use crate::{current_requested_outputs, gather_if_needed_async, BuiltinResult};
use runmat_value::Value;

use super::callback::Callable;
use super::error_context;
use super::fields::Fields;
use super::options::Options;
use super::output::{self, Collector};

pub(super) async fn execute(
    function: Value,
    structure: Value,
    rest: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    let _inputs_root =
        RootedValueLease::for_values([&function, &structure].into_iter().chain(rest.iter()))
            .map_err(|error| error.into_runtime_error())?;
    let callable = Callable::parse(function)?;
    let fields = Fields::parse(structure)?;
    let options = Options::parse(rest)?;
    let requested = current_requested_outputs();
    let mut collectors = (0..requested)
        .map(|_| Collector::new(options.uniform))
        .collect::<Vec<_>>();
    for (index, (field, value)) in fields.names.iter().zip(&fields.values).enumerate() {
        let value = gather_if_needed_async(value).await?;
        let result = match callable.call(std::slice::from_ref(&value), requested).await {
            Ok(result) => result,
            Err(callback_error) => {
                let Some(handler) = &options.handler else {
                    return Err(callback_error.into_runtime_error());
                };
                handler
                    .call(
                        &[
                            error_context::value(&callback_error, index, field),
                            value.clone(),
                        ],
                        requested,
                    )
                    .await
                    .map_err(super::callback::CallbackFailure::into_runtime_error)?
            }
        };
        if requested == 0 {
            // Discarding results must not discard callback or handler effects.
            continue;
        }
        for (collector, result) in collectors
            .iter_mut()
            .zip(output::normalize(result, requested)?)
        {
            let gathered = gather_if_needed_async(result.value()).await?;
            collector.push(
                field,
                RootedValue::new(gathered).map_err(|error| error.into_runtime_error())?,
            )?;
        }
    }
    let mut outputs = RootedValuesBuilder::with_capacity(collectors.len());
    for collector in collectors {
        collector
            .finish(fields.names.len())?
            .with_owned(|value| outputs.push(value))
            .map_err(|error| error.into_runtime_error())?;
    }
    outputs
        .finish()
        .map_err(|error| error.into_runtime_error())?
        .with_owned_output_storage(|values| {
            let value = if requested == 1 {
                values
                    .try_into_first()
                    .map_err(|_| {
                        crate::sequence::sequence_error_to_runtime(
                            runmat_value::ValueSequenceError::SharedOutputStorage,
                        )
                    })?
                    .expect("one requested output")
            } else {
                Value::OutputList(values)
            };
            RootedValue::new(value).map_err(|error| error.into_runtime_error())
        })
}
