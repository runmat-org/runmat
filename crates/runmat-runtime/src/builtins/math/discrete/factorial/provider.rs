use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, IntegerElementType, ProviderPrecision,
};
use runmat_value::Value;

use super::{evaluation::factorial_tensor, factorial_error, FactorialError, BUILTIN_NAME};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::builtins::math::resident_real_unary;
use crate::BuiltinResult;

pub(super) async fn factorial_gpu(input: GpuTensorHandle) -> BuiltinResult<Value> {
    let integer = runmat_accelerate_api::handle_integer_type(&input);
    if matches!(
        integer,
        Some(IntegerElementType::I64 | IntegerElementType::U64)
    ) {
        return Err(factorial_error(
            FactorialError::InvalidInput,
            "64-bit integer GPU inputs are not supported",
        ));
    }
    if runmat_accelerate_api::handle_storage(&input) == GpuTensorStorage::ComplexInterleaved {
        return Err(factorial_error(
            FactorialError::InvalidInput,
            "complex inputs are not supported; use gamma(z + 1) instead",
        ));
    }
    if runmat_accelerate_api::handle_is_logical(&input) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::FACTORIAL_LOGICAL_EXTENSION,
            BUILTIN_NAME,
        )?;
        return evaluate_and_restore(&input, OutputRepresentation::Double).await;
    }
    if let Some(integer) = integer {
        return evaluate_and_restore(&input, OutputRepresentation::Integer(integer)).await;
    }
    evaluate_floating(input).await
}

async fn evaluate_floating(input: GpuTensorHandle) -> BuiltinResult<Value> {
    resident_real_unary::validate_input(&input)
        .map_err(|reason| factorial_error(FactorialError::InvalidInput, reason.detail()))?;
    let provider = resident_real_unary::exact_owner(&input).ok_or_else(|| {
        factorial_error(FactorialError::Internal, "GPU input has no owning provider")
    })?;
    match provider.unary_factorial(&input).await {
        Ok(mut output) => {
            if !resident_real_unary::output_matches(&output, &input, provider) {
                resident_real_unary::reject_output(&output, &input, provider);
                return Err(factorial_error(
                    FactorialError::Internal,
                    "provider unary_factorial returned malformed output",
                ));
            }
            resident_real_unary::preserve_residency_intent(&mut output, &input);
            Ok(gpu_helpers::resident_gpu_value(output))
        }
        Err(error) if resident_real_unary::hook_is_unsupported(&error) => {
            evaluate_and_restore(&input, OutputRepresentation::Floating).await
        }
        Err(error) => Err(factorial_error(FactorialError::Internal, error)),
    }
}

#[derive(Clone, Copy)]
enum OutputRepresentation {
    Floating,
    Double,
    Integer(IntegerElementType),
}

async fn evaluate_and_restore(
    input: &GpuTensorHandle,
    representation: OutputRepresentation,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(input).ok_or_else(|| {
        factorial_error(FactorialError::Internal, "GPU input has no owning provider")
    })?;
    let gathered = gpu_helpers::gather_tensor_async(input)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let output = factorial_tensor(gathered)?;
    let mut restored = gpu_helpers::upload_tensor(provider, &output)
        .map_err(|error| factorial_error(FactorialError::Internal, error))?;
    if !output_matches(&restored, input, provider, representation) {
        gpu_helpers::free_rejected_provider_output(&restored, &[input], provider);
        return Err(factorial_error(
            FactorialError::Internal,
            "provider upload returned malformed factorial output",
        ));
    }
    resident_real_unary::preserve_residency_intent(&mut restored, input);
    Ok(gpu_helpers::resident_gpu_value(restored))
}

fn output_matches(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    representation: OutputRepresentation,
) -> bool {
    let (precision, integer) = match representation {
        OutputRepresentation::Floating => (runmat_accelerate_api::handle_precision(input), None),
        OutputRepresentation::Double => (Some(ProviderPrecision::F64), None),
        OutputRepresentation::Integer(integer) => (None, Some(integer)),
    };
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision,
            integer,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}
