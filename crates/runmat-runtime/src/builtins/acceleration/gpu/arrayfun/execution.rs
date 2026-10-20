use crate::execution::{RootedValue, RootedValuesBuilder};
use crate::{gather_if_needed_async, make_cell_with_shape, BuiltinResult};
use runmat_value::Value;

use super::error::{arrayfun_flow, arrayfun_flow_with_source, format_handler_error};
use super::error_context::make_error_struct;
use super::gpu;
use super::options::Invocation;
use super::output::{empty_uniform, maybe_upload_uniform, OutputContract, UniformCollector};
use super::plan::InputPlan;

pub(super) async fn execute(
    context: Option<&crate::context::RuntimeContext>,
    func: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _function_root =
        crate::builtins::common::mapped_callable::root_callable_value(&func, context)?;
    let invocation = Invocation::parse(func, rest)?;
    let Invocation {
        callable,
        inputs,
        uniform_output,
        error_handler,
    } = invocation;
    let resident_inputs: Vec<_> = inputs
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let has_gpu_input = !resident_inputs.is_empty();
    if uniform_output {
        if let Some(gpu_result) =
            gpu::try_fast_path(context, &callable, &inputs, error_handler.as_ref()).await?
        {
            return Ok(gpu_result);
        }
    }

    let plan = InputPlan::prepare(context, &inputs, has_gpu_input).await?;
    let base_shape = plan.output_shape;
    let prepared_inputs = plan.inputs;

    let total_len = base_shape.iter().product();

    if total_len == 0 {
        if uniform_output {
            let contract = OutputContract::infer(&callable, &prepared_inputs);
            return RootedValue::new(maybe_upload_uniform(
                empty_uniform(&base_shape, contract)?,
                &resident_inputs,
            )?)
            .map_err(|error| error.into_runtime_error());
        } else {
            return RootedValue::new(
                make_cell_with_shape(Vec::new(), base_shape)
                    .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?,
            )
            .map_err(|error| error.into_runtime_error());
        }
    }

    let mut collector = if uniform_output {
        Some(UniformCollector::Pending)
    } else {
        None
    };

    let mut cell_outputs = RootedValuesBuilder::with_capacity(total_len);

    for idx in 0..total_len {
        let args = inputs
            .try_map_preserving_roots_async(prepared_inputs.len(), context, async |slot| {
                prepared_inputs[slot].value_at(idx, &base_shape)
            })
            .await?;

        let result = match callable.call_retained(context, args.fork()).await {
            Ok(value) => value,
            Err(err) => {
                let handler = match error_handler.as_ref() {
                    Some(handler) => handler,
                    None => {
                        return Err(arrayfun_flow_with_source(
                            format!("arrayfun: {}", err.message()),
                            err,
                        ))
                    }
                };
                let err_message = format_handler_error(&err);
                let prefix = make_error_struct(context, &err_message, idx)?;
                let handler_args =
                    crate::builtins::common::mapped_callable::prepend_rooted_argument(
                        &args, prefix, context,
                    )
                    .await?;
                handler.call_retained(context, handler_args).await?
            }
        };

        let host_result = gather_if_needed_async(result.value()).await?;

        if let Some(collector) = collector.as_mut() {
            collector.push(&host_result)?;
        } else {
            cell_outputs
                .push(host_result)
                .map_err(|error| error.into_runtime_error())?;
        }
    }

    if let Some(collector) = collector {
        let uniform = collector.finish(&base_shape)?;
        RootedValue::new(maybe_upload_uniform(uniform, &resident_inputs)?)
            .map_err(|error| error.into_runtime_error())
    } else {
        cell_outputs
            .finish()
            .map_err(|error| error.into_runtime_error())?
            .with_unbounded_values(|values| {
                RootedValue::new(
                    make_cell_with_shape(values, base_shape)
                        .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?,
                )
                .map_err(|error| error.into_runtime_error())
            })?
    }
}
