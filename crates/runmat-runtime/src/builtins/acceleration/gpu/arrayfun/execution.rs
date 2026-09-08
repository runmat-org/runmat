use crate::{gather_if_needed_async, make_cell_with_shape, BuiltinResult};
use runmat_value::Value;

use super::error::{arrayfun_flow, arrayfun_flow_with_source, format_handler_error};
use super::error_context::make_error_struct;
use super::gpu;
use super::options::Invocation;
use super::output::{empty_uniform, maybe_upload_uniform, OutputContract, UniformCollector};
use super::plan::InputPlan;

pub(super) async fn execute(func: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let invocation = Invocation::parse(func, rest)?;
    let Invocation {
        callable,
        inputs,
        uniform_output,
        error_handler,
    } = invocation;
    let has_gpu_input = inputs
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)));
    if uniform_output {
        if let Some(gpu_result) =
            gpu::try_fast_path(&callable, &inputs, error_handler.as_ref()).await?
        {
            return Ok(gpu_result);
        }
    }

    let plan = InputPlan::prepare(inputs, has_gpu_input).await?;
    let base_shape = plan.output_shape;
    let prepared_inputs = plan.inputs;

    let total_len = base_shape.iter().product();

    if total_len == 0 {
        if uniform_output {
            let contract = OutputContract::infer(&callable, &prepared_inputs);
            return empty_uniform(&base_shape, contract);
        } else {
            return make_cell_with_shape(Vec::new(), base_shape)
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")));
        }
    }

    let mut collector = if uniform_output {
        Some(UniformCollector::Pending)
    } else {
        None
    };

    let mut cell_outputs: Vec<Value> = Vec::new();
    let mut args: Vec<Value> = Vec::with_capacity(prepared_inputs.len());

    for idx in 0..total_len {
        args.clear();
        for input in &prepared_inputs {
            args.push(input.value_at(idx, &base_shape)?);
        }

        let result = match callable.call(&args).await {
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
                let err_value = make_error_struct(&err_message, idx);
                let mut handler_args = Vec::with_capacity(1 + args.len());
                handler_args.push(err_value);
                handler_args.extend(args.clone());
                handler.call(&handler_args).await?
            }
        };

        let host_result = gather_if_needed_async(&result).await?;

        if let Some(collector) = collector.as_mut() {
            collector.push(&host_result)?;
        } else {
            cell_outputs.push(host_result);
        }
    }

    if let Some(collector) = collector {
        let uniform = collector.finish(&base_shape)?;
        maybe_upload_uniform(uniform, has_gpu_input)
    } else {
        make_cell_with_shape(cell_outputs, base_shape)
            .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))
    }
}
