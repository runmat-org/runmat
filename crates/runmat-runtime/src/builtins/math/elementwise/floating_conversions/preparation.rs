//! Unary data preparation after the owning conversion admits its output template.
use crate::builtins::common::gpu_helpers;
use crate::context::preparation::{
    execute_request_host_numerical, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
    PreparedHostNumericalOutput,
};
use crate::context::RuntimeContext;
use crate::context::VmFusionHostInstruction;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_value::Value;
use std::future::Future;

#[cfg(test)]
mod tests;

pub(super) async fn execute<F, Fut>(
    runtime: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    value: Value,
    signature: usize,
    input_count: usize,
    form: BuiltinPreparationFormId,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value, Option<PreparedHostNumericalOutput>) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
{
    // Text, sparse, complex and symbolic conversions retain their specialized
    // runtime-owned representation routes. Unsupported values reach the same
    // domain error without calling a preparation service.
    let admitted = match &value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => {
            gpu_helpers::exact_provider_for_handle(handle).is_some()
                && gpu_helpers::expected_handle_numeric_element_type(handle).is_ok()
                && handle.descriptor().storage
                    == Some(runmat_accelerate_api::GpuTensorStorage::Real)
        }
        _ => false,
    };
    if !admitted {
        return body(value, None).await;
    }
    let binding = BuiltinBindingIdentity {
        builtin: identity,
        variant: "default",
    };
    if input_count == 1 {
        if let Some(runtime) = runtime {
            if let Some(route) = crate::builtins::math::host_route::canonical(
                Some(runtime),
                VmFusionHostInstruction::ElementwiseCall,
                std::iter::once(&value),
            ) {
                let numerical_output = route.take_numerical_output(
                    runtime,
                    VmFusionHostInstruction::ElementwiseCall,
                    binding,
                )?;
                let output = body(value, numerical_output).await?;
                if crate::builtins::math::host_route::real_floating_host(output.value()) {
                    route.confirm_host_branch(runtime, VmFusionHostInstruction::ElementwiseCall);
                }
                return Ok(output);
            }
        }
    }
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding,
            form,
            signature,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
    )?;
    execute_request_host_numerical(runtime, request, |mut operands, numerical_output| {
        let value = operands.pop().expect("one admitted conversion operand");
        let route = crate::builtins::math::host_route::canonical(
            runtime,
            VmFusionHostInstruction::ElementwiseCall,
            std::iter::once(&value),
        );
        async move {
            let output = body(value, numerical_output).await?;
            if let (Some(route), Some(runtime)) = (route, runtime) {
                if crate::builtins::math::host_route::real_floating_host(output.value()) {
                    route.confirm_host_branch(runtime, VmFusionHostInstruction::ElementwiseCall);
                }
            }
            Ok(output)
        }
    })
    .await
}
