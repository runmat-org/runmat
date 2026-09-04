use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, IntegerElementType, ProviderPrecision,
};
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;

use super::{errors, operation, BUILTIN_NAME};

pub(super) async fn evaluate(input: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&input) == GpuTensorStorage::ComplexInterleaved {
        return Err(errors::invalid("complex gpuArray input is not supported"));
    }
    let provider = gpu_helpers::exact_provider_for_handle(&input)
        .ok_or_else(|| errors::internal("GPU input has no owning provider"))?;
    if output_representation(&input).integer.is_some()
        || runmat_accelerate_api::handle_is_logical(&input)
    {
        return evaluate_and_restore(&input, provider).await;
    }

    match provider.unary_nextpow2(&input).await {
        Ok(mut output)
            if output_matches(&output, &input, provider, output_representation(&input)) =>
        {
            preserve_residency(&mut output, &input);
            Ok(gpu_helpers::resident_gpu_value(output))
        }
        Ok(output) => {
            gpu_helpers::free_rejected_provider_output(&output, &[&input], provider);
            Err(errors::internal(
                "provider unary_nextpow2 returned malformed output",
            ))
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
            evaluate_and_restore(&input, provider).await
        }
        Err(error) => Err(errors::internal(format!(
            "provider unary_nextpow2 failed: {error}"
        ))),
    }
}

#[derive(Clone, Copy)]
struct OutputRepresentation {
    precision: Option<ProviderPrecision>,
    integer: Option<IntegerElementType>,
}

fn output_representation(input: &GpuTensorHandle) -> OutputRepresentation {
    let integer = runmat_accelerate_api::handle_integer_type(input);
    let precision = if integer.is_some() {
        None
    } else if runmat_accelerate_api::handle_is_logical(input) {
        Some(ProviderPrecision::F64)
    } else {
        runmat_accelerate_api::handle_precision(input)
    };
    OutputRepresentation { precision, integer }
}

async fn evaluate_and_restore(
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Value> {
    let gathered = gpu_helpers::gather_tensor_async(input)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let result = operation::transform(gathered)?;
    let mut output = gpu_helpers::upload_tensor(provider, &result).map_err(errors::internal)?;
    if !output_matches(&output, input, provider, output_representation(input)) {
        gpu_helpers::free_rejected_provider_output(&output, &[input], provider);
        return Err(errors::internal(
            "provider upload returned malformed fallback output",
        ));
    }
    preserve_residency(&mut output, input);
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn output_matches(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    representation: OutputRepresentation,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: representation.precision,
            integer: representation.integer,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}

fn preserve_residency(output: &mut GpuTensorHandle, input: &GpuTensorHandle) {
    let provenance = runmat_accelerate_api::handle_provenance(input)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(output, provenance);
}
