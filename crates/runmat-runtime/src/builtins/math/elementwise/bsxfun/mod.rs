//! Binary callback execution with singleton expansion.

mod broadcast;
mod callback;
mod error;
mod input;
mod output;
mod spec;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[cfg(test)]
mod tests;

use runmat_builtins::BSXFUN_EXTENSIONS;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{call_feval_async_with_outputs, gather_if_needed_async, BuiltinResult};

#[cfg(test)]
use output::{classify_value, ClassifiedValue, ComplexClassedValue};

pub(super) const BUILTIN_NAME: &str = "bsxfun";

#[runtime_builtin(
    name = "bsxfun",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::bsxfun"
)]
async fn bsxfun_builtin(function: Value, left: Value, right: Value) -> BuiltinResult<Value> {
    callback::validate(&function)?;
    if matches!(
        function,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BSXFUN_EXTENSIONS[0],
            BUILTIN_NAME,
        )?;
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&left, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&right, BUILTIN_NAME)?;
    let left = gather_if_needed_async(&left)
        .await
        .map_err(|flow| error::internal(flow.to_string()))?;
    let right = gather_if_needed_async(&right)
        .await
        .map_err(|flow| error::internal(flow.to_string()))?;
    let left = input::ArrayInput::from_value(left)?;
    let right = input::ArrayInput::from_value(right)?;
    let plan = broadcast::BroadcastPlan::new(left.shape(), right.shape())?;
    let output_contract = callback::OutputContract::infer(&function, &left, &right);

    let mut collector = output::UniformCollector::default();
    for (_, left_index, right_index) in plan.iter() {
        let arguments = [left.value_at(left_index)?, right.value_at(right_index)?];
        let value = call_feval_async_with_outputs(function.clone(), &arguments, 1)
            .await
            .map_err(|cause| error::callback(cause.message()))?;
        let value = gather_if_needed_async(&value)
            .await
            .map_err(|flow| error::internal(flow.to_string()))?;
        collector.push(output_contract.normalize(value)?)?;
    }

    collector.finish(plan.output_shape(), output_contract)
}
