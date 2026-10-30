//! Domain admission for shape-preserving floating trigonometric data operands.
//! Controls and output prototypes remain in the identity's parsed state.
use crate::builtins::common::gpu_helpers;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::preparation::{
    execute_request_host_numerical, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_value::{NumericDType, Value};
use std::future::Future;

struct RealKernels<F32, F64> {
    f32: F32,
    f64: F64,
}

#[cfg(test)]
mod tests;

fn admits(value: &Value, identity: BuiltinCatalogIdentity) -> BuiltinResult<bool> {
    Ok(match value {
        Value::Num(_) => true,
        Value::Tensor(tensor) => matches!(
            tensor.numeric_dtype(),
            NumericDType::F32 | NumericDType::F64
        ),
        Value::GpuTensor(handle) => {
            if !gpu_helpers::expected_handle_numeric_element_type(handle).is_ok_and(|element| {
                gpu_helpers::numeric_descriptor_matches(
                    handle,
                    element,
                    runmat_accelerate_api::handle_storage(handle),
                )
            }) {
                return Err(build_runtime_error(format!(
                    "{}: malformed or unowned GPU input",
                    identity.name
                ))
                .with_builtin(identity.name)
                .with_identifier(format!("RunMat:{}:Internal", identity.name))
                .build());
            }
            gpu_helpers::require_resident_compute_owner(handle, || {
                build_runtime_error(format!("{}: malformed or unowned GPU input", identity.name))
                    .with_builtin(identity.name)
                    .with_identifier(format!("RunMat:{}:Internal", identity.name))
                    .build()
            })?;
            runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
        }
        _ => false,
    })
}

pub(crate) async fn unary_real<F, Fut, F32, F64>(
    runtime: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    value: Value,
    f32_kernel: F32,
    f64_kernel: F64,
    execute: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
    F32: FnMut(f32) -> BuiltinResult<f32>,
    F64: FnMut(f64) -> BuiltinResult<f64>,
{
    unary_form_inner(
        runtime,
        identity,
        value,
        0,
        1,
        RealKernels {
            f32: f32_kernel,
            f64: f64_kernel,
        },
        execute,
    )
    .await
}

pub(crate) async fn unary_form_real<F, Fut, F32, F64>(
    runtime: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    value: Value,
    signature: usize,
    input_count: usize,
    f32_kernel: F32,
    f64_kernel: F64,
    execute: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
    F32: FnMut(f32) -> BuiltinResult<f32>,
    F64: FnMut(f64) -> BuiltinResult<f64>,
{
    unary_form_inner(
        runtime,
        identity,
        value,
        signature,
        input_count,
        RealKernels {
            f32: f32_kernel,
            f64: f64_kernel,
        },
        execute,
    )
    .await
}

async fn unary_form_inner<F, Fut, F32, F64>(
    runtime: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    value: Value,
    signature: usize,
    input_count: usize,
    kernels: RealKernels<F32, F64>,
    execute: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
    F32: FnMut(f32) -> BuiltinResult<f32>,
    F64: FnMut(f64) -> BuiltinResult<f64>,
{
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    let managed = match runtime {
        Some(context) => {
            HostAllocationReceipt::policy(Some(context)).map_err(|error| {
                crate::builtin::builtin_control_error(
                    Some(&context.cancellation_context()),
                    identity.name,
                    error,
                )
            })? == HostAllocationPolicy::Managed
        }
        None => false,
    };
    if let Some(route) = crate::builtins::math::host_route::canonical(
        runtime,
        crate::context::VmFusionHostInstruction::ElementwiseCall,
        std::iter::once(&value),
    ) {
        let context = runtime.ok_or_else(|| invalid_selected(identity))?;
        if managed {
            if signature != 0 || input_count != 1 {
                return Err(invalid_selected(identity));
            }
            let output = route
                .take_numerical_output(
                    context,
                    crate::context::VmFusionHostInstruction::ElementwiseCall,
                    BuiltinBindingIdentity {
                        builtin: identity,
                        variant: "default",
                    },
                )?
                .ok_or_else(|| {
                    crate::builtin::builtin_control_error(
                        Some(&context.cancellation_context()),
                        identity.name,
                        HostAllocationError::MissingManagedOwner,
                    )
                })?;
            let mut kernels = kernels;
            let result = crate::builtins::common::host_unary_output::real(
                context,
                output,
                runmat_builtins::BuiltinPreparationOperation::Unary,
                &value,
                &mut kernels.f32,
                &mut kernels.f64,
            )?;
            let result = RootedValue::new(result).map_err(|cause| cause.into_runtime_error())?;
            route.confirm_host_branch(
                context,
                crate::context::VmFusionHostInstruction::ElementwiseCall,
            );
            return Ok(result);
        } else if signature == 0 && input_count == 1 {
            let output = execute(value).await?;
            route.confirm_host_branch(
                context,
                crate::context::VmFusionHostInstruction::ElementwiseCall,
            );
            return Ok(output);
        }
    }
    if !admits(&value, identity)? {
        // These representations use the identity's exact host/complex or
        // owner-restoring path; they are not inputs to a real floating hook.
        return execute(value).await;
    }
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
    )?;
    execute_request_host_numerical(
        runtime,
        request,
        |mut operands, numerical_output| async move {
            let value = operands.pop().expect("one admitted trigonometric operand");
            if !admits(&value, identity)? {
                return Err(build_runtime_error(format!(
                    "{}: preparation changed admitted floating input representation",
                    identity.name
                ))
                .with_builtin(identity.name)
                .with_identifier(format!("RunMat:{}:Internal", identity.name))
                .build());
            }
            let route = crate::builtins::math::host_route::canonical(
                runtime,
                crate::context::VmFusionHostInstruction::ElementwiseCall,
                std::iter::once(&value),
            );
            let output = if let Some(output) = numerical_output {
                if signature != 0 || input_count != 1 {
                    return Err(invalid_selected(identity));
                }
                let Some(context) = runtime else {
                    return Err(invalid_selected(identity));
                };
                let mut kernels = kernels;
                let output = crate::builtins::common::host_unary_output::real(
                    context,
                    output,
                    runmat_builtins::BuiltinPreparationOperation::Unary,
                    &value,
                    &mut kernels.f32,
                    &mut kernels.f64,
                )?;
                RootedValue::new(output).map_err(|cause| cause.into_runtime_error())?
            } else {
                if managed && route.is_some() && matches!(&value, Value::Num(_) | Value::Tensor(_))
                {
                    let context = runtime.ok_or_else(|| invalid_selected(identity))?;
                    return Err(crate::builtin::builtin_control_error(
                        Some(&context.cancellation_context()),
                        identity.name,
                        HostAllocationError::MissingManagedOwner,
                    ));
                }
                execute(value).await?
            };
            if let Some(route) = route {
                route.confirm_host_branch(
                    runtime.unwrap(),
                    crate::context::VmFusionHostInstruction::ElementwiseCall,
                );
            }
            Ok(output)
        },
    )
    .await
}

fn invalid_selected(identity: BuiltinCatalogIdentity) -> crate::RuntimeError {
    build_runtime_error(format!(
        "{}: selected unary output lacks its admitted consumer",
        identity.name
    ))
    .with_builtin(identity.name)
    .with_identifier("RunMat:Placement:InvalidDecision")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}
