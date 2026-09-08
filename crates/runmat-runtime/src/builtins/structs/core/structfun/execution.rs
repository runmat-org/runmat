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
) -> BuiltinResult<Value> {
    let callable = Callable::parse(function)?;
    let fields = Fields::parse(structure)?;
    let options = Options::parse(rest)?;
    let requested = current_requested_outputs();
    if requested == 0 {
        return Ok(Value::OutputList(Vec::new()));
    }
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
        for (collector, result) in collectors
            .iter_mut()
            .zip(output::normalize(result, requested)?)
        {
            collector.push(field, gather_if_needed_async(&result).await?)?;
        }
    }
    let outputs = collectors
        .into_iter()
        .map(|collector| collector.finish(fields.names.len()))
        .collect::<BuiltinResult<Vec<_>>>()?;
    if requested == 1 {
        Ok(outputs.into_iter().next().expect("one requested output"))
    } else {
        Ok(Value::OutputList(outputs))
    }
}
