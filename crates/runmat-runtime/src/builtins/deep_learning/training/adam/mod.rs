//! Adam update admission, provider execution, and numeric evaluation.

use super::super::{deep_learning_error, gather_args, object, positive_usize};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::BuiltinResult;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderAdamUpdateRequest,
    ProviderAdamUpdateResult, ProviderExecutionError,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

mod admission;
mod execution;
mod provider;
#[cfg(test)]
mod tests;

use admission::*;
use execution::*;
use provider::*;

#[runtime_builtin(
    name = "adamupdate",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(in crate::builtins::deep_learning) async fn adamupdate_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args
        .iter()
        .take(4)
        .any(|value| matches!(value, Value::Int(_)))
        || args
            .iter()
            .skip(4)
            .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::ADAMUPDATE_INTEGER_EXTENSION,
            "adamupdate",
        )?;
    }
    if let Some(requested) = crate::output_count::current_output_count() {
        if requested > 3 {
            return Err(deep_learning_error(
                "adamupdate",
                format!("adamupdate: requested {requested} outputs, but at most 3 are supported"),
            ));
        }
    }
    let has_gpu_input = args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)));
    let has_explicit_gpu_input = args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu);
    if has_gpu_input {
        preflight_adam_host_operands(args.view())?;
    }
    let options = if has_gpu_input {
        match try_adamupdate_gpu(args.view())? {
            Some(eval) => return eval.output(),
            None if has_explicit_gpu_input => return Err(explicit_gpu_fallback_error()),
            None => {
                let gathered = gather_args(args, runtime.as_ref()).await?;
                AdamUpdateArgs::parse(gathered)?
            }
        }
    } else if has_explicit_gpu_input {
        return Err(explicit_gpu_fallback_error());
    } else {
        AdamUpdateArgs::parse(args)?
    };
    let eval = evaluate_adamupdate(options)?;
    match crate::output_count::current_output_count() {
        None => Ok(eval.parameters),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![eval.parameters])),
        Some(2) => Ok(Value::output_list(vec![eval.parameters, eval.average_grad])),
        Some(3) => Ok(Value::output_list(vec![
            eval.parameters,
            eval.average_grad,
            eval.average_sq_grad,
        ])),
        Some(requested) => Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: requested {requested} outputs, but at most 3 are supported"),
        )),
    }
}

fn explicit_gpu_fallback_error() -> crate::RuntimeError {
    super::super::errors::descriptor_error(
        "adamupdate",
        "adamupdate: explicit gpuArray inputs require a supported resident Adam provider path",
        &runmat_builtins::ADAMUPDATE_UNSUPPORTED,
    )
}

/// Reject known host errors before a mixed-residency fallback downloads a
/// provider-owned operand. The ordinary host parser remains the authority for
/// materialized data; this admission uses it only for host operands and its
/// existing scalar-control decoders.
fn preflight_adam_host_operands(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    if !(5..=9).contains(&args.len()) {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: expected 5 to 9 positional arguments",
        ));
    }
    let mut shapes = [None, None, None, None];
    for index in 0..4 {
        match &args[index] {
            Value::GpuTensor(handle) => {
                shapes[index] = Some(handle.shape().to_vec());
                if index == 0 && handle.shape().contains(&0) {
                    return Err(deep_learning_error(
                        "adamupdate",
                        "adamupdate: parameters must not be empty",
                    ));
                }
            }
            host => {
                let parsed = NumericPayload::parse(
                    host,
                    ["parameters", "grad", "averageGrad", "averageSqGrad"][index],
                    index < 2,
                )?;
                if !parsed.data.is_empty() {
                    shapes[index] = Some(parsed.shape);
                }
            }
        }
    }
    if let Some(expected) = &shapes[0] {
        for index in 1..4 {
            if let Some(actual) = &shapes[index] {
                if expected != actual {
                    return Err(deep_learning_error(
                        "adamupdate",
                        format!(
                            "adamupdate: {} must match parameter shape {:?}, got {:?}",
                            ["parameters", "grad", "averageGrad", "averageSqGrad"][index],
                            expected,
                            actual
                        ),
                    ));
                }
            }
        }
    }
    if !matches!(args[4], Value::GpuTensor(_)) {
        positive_iteration(&args[4])?;
    }
    for (index, default, label) in [(5, 0.001, "learnRate"), (8, 1.0e-8, "epsilon")] {
        if args
            .get(index)
            .is_some_and(|value| !matches!(value, Value::GpuTensor(_)))
        {
            optional_positive_scalar(args, index, default, label)?;
        }
    }
    for (index, default, label) in [(6, 0.9, "gradDecay"), (7, 0.999, "sqGradDecay")] {
        if args
            .get(index)
            .is_some_and(|value| !matches!(value, Value::GpuTensor(_)))
        {
            optional_unit_scalar(args, index, default, label)?;
        }
    }
    Ok(())
}
