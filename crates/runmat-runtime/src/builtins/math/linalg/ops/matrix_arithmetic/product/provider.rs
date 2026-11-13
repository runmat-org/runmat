use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::{Tensor, Value};

use super::ProductResult;
use crate::builtins::common::gpu_helpers::{BinaryGpuOutputContract, GpuOutputAliasPolicy};
use crate::builtins::common::{binary, gpu_helpers, tensor};
use crate::BuiltinResult;

#[cfg(test)]
#[path = "provider_owning_tests.rs"]
mod owning_tests;

pub(super) async fn try_product(
    lhs: &Value,
    rhs: &Value,
) -> BuiltinResult<ProductResult<Option<Value>>> {
    gpu_helpers::require_resident_compute_values(&[lhs, rhs], false, || {
        crate::RuntimeError::new("mtimes: resident input has no exact provider owner")
    })?;
    if contains_complex(lhs) || contains_complex(rhs) {
        return Ok(Ok(None));
    }
    let Some(source) = gpu_helpers::select_resident_output_source(
        [lhs, rhs].into_iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        super::NAME,
    )?
    else {
        return Ok(Ok(None));
    };
    let provider = gpu_helpers::require_resident_compute_owner(&source, || {
        crate::RuntimeError::new("mtimes: resident input has no exact provider owner")
    })?;

    if let Some(result) = product_try!(try_scalar_product(provider, lhs, rhs).await?) {
        return Ok(Ok(Some(result)));
    }
    let mut protected = resident_inputs(lhs, rhs);
    let Some(lhs) = product_try!(PreparedOperand::new(lhs, provider, &protected)?) else {
        return Ok(Ok(None));
    };
    protected.push(lhs.handle());
    let Some(rhs) = product_try!(PreparedOperand::new(rhs, provider, &protected)?) else {
        return Ok(Ok(None));
    };
    let expected_shape = product_try!(product_shape(lhs.handle(), rhs.handle()));
    let output = match provider.matmul(lhs.handle(), rhs.handle()).await {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
            return Ok(Ok(None))
        }
        Err(error) => return Err(provider_failure("matrix multiplication", error)),
    };
    finish_binary_output(output, provider, lhs.handle(), rhs.handle(), expected_shape)
        .map(|value| Ok(Some(value)))
}

async fn try_scalar_product(
    provider: &'static dyn AccelProvider,
    lhs: &Value,
    rhs: &Value,
) -> BuiltinResult<ProductResult<Option<Value>>> {
    let protected = resident_inputs(lhs, rhs);
    if let Some(scalar) = real_scalar_value(provider, lhs).await? {
        if let Some(operand) = product_try!(PreparedOperand::new(rhs, provider, &protected)?) {
            return invoke_scalar_product(provider, &operand, scalar).map(Ok);
        }
    }
    if let Some(scalar) = real_scalar_value(provider, rhs).await? {
        if let Some(operand) = product_try!(PreparedOperand::new(lhs, provider, &protected)?) {
            return invoke_scalar_product(provider, &operand, scalar).map(Ok);
        }
    }
    Ok(Ok(None))
}

fn invoke_scalar_product(
    provider: &'static dyn AccelProvider,
    operand: &PreparedOperand,
    scalar: f64,
) -> BuiltinResult<Option<Value>> {
    let output = match provider.scalar_mul(operand.handle(), scalar) {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
            return Ok(None);
        }
        Err(error) => return Err(provider_failure("scalar multiplication", error)),
    };
    let contract = output_contract(operand.handle().shape().to_vec(), provider);
    if gpu_helpers::unary_gpu_output_matches(
        &output,
        operand.handle(),
        provider,
        crate::builtins::common::gpu_helpers::UnaryGpuOutputContract {
            storage: contract.storage,
            precision: contract.precision,
            integer: contract.integer,
            logical: contract.logical,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) {
        let output = gpu_helpers::propagate_owned_output_provenance(output, [operand.handle()])?;
        return gpu_helpers::resident_gpu_value(output).map(Some);
    }
    let mut error = crate::RuntimeError::new(
        "mtimes: provider returned invalid scalar-product output metadata",
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    Err(gpu_helpers::reject_gpu_output(output, error))
}

fn finish_binary_output(
    output: GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    let contract = output_contract(shape, provider);
    binary::validate_resident_output(provider, lhs, rhs, output, &contract)
}

fn output_contract(
    shape: Vec<usize>,
    provider: &'static dyn AccelProvider,
) -> BinaryGpuOutputContract {
    BinaryGpuOutputContract {
        shape,
        storage: GpuTensorStorage::Real,
        precision: Some(provider.precision()),
        integer: None,
        logical: false,
        alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
    }
}

async fn real_scalar_value(
    provider: &'static dyn AccelProvider,
    value: &Value,
) -> BuiltinResult<Option<f64>> {
    match value {
        Value::Num(value) => Ok(Some(*value)),
        Value::Bool(value) => Ok(Some(if *value { 1.0 } else { 0.0 })),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            Ok(Some(tensor::tensor_value_f64(value, 0)))
        }
        Value::LogicalArray(value) if value.data.len() == 1 => {
            Ok(Some(if value.data[0] != 0 { 1.0 } else { 0.0 }))
        }
        Value::GpuTensor(handle) if is_scalar(handle) => {
            let host = gpu_helpers::download_floating_projection_async(provider, handle).await?;
            Ok(host.data.first().copied())
        }
        _ => Ok(None),
    }
}

fn product_shape(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> ProductResult<Vec<usize>> {
    if lhs.shape().len() != 2 || rhs.shape().len() != 2 || lhs.shape()[1] != rhs.shape()[0] {
        return Err(super::errors::invalid_input(format!(
            "mtimes: incompatible matrix dimensions {:?} and {:?}",
            lhs.shape(),
            rhs.shape()
        )));
    }
    Ok(vec![lhs.shape()[0], rhs.shape()[1]])
}

fn contains_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn is_scalar(handle: &GpuTensorHandle) -> bool {
    crate::builtins::common::shape::is_scalar_shape(handle.shape())
}

struct PreparedOperand {
    handle: GpuTensorHandle,
}

impl PreparedOperand {
    fn new(
        value: &Value,
        provider: &'static dyn AccelProvider,
        protected: &[&GpuTensorHandle],
    ) -> BuiltinResult<ProductResult<Option<Self>>> {
        match value {
            Value::GpuTensor(handle) if compatible(handle, provider) && !is_scalar(handle) => {
                Ok(Ok(Some(Self {
                    handle: handle.clone(),
                })))
            }
            Value::GpuTensor(_) => Ok(Ok(None)),
            Value::Tensor(value) if !tensor::is_scalar_tensor(value) => {
                Self::upload(provider, value, protected).map(|value| Ok(Some(value)))
            }
            Value::LogicalArray(value) if value.data.len() != 1 => {
                let value = product_try!(
                    tensor::logical_to_tensor(value).map_err(super::errors::invalid_input)
                );
                Self::upload(provider, &value, protected).map(|value| Ok(Some(value)))
            }
            _ => Ok(Ok(None)),
        }
    }

    fn upload(
        provider: &'static dyn AccelProvider,
        value: &Tensor,
        protected: &[&GpuTensorHandle],
    ) -> BuiltinResult<Self> {
        let handle = gpu_helpers::upload_tensor_protecting(provider, value, protected)?;
        Ok(Self { handle })
    }

    fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
}

fn resident_inputs<'a>(lhs: &'a Value, rhs: &'a Value) -> Vec<&'a GpuTensorHandle> {
    [lhs, rhs]
        .into_iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect()
}

fn provider_failure(operation: &str, source: anyhow::Error) -> crate::RuntimeError {
    let mut error =
        crate::RuntimeError::new(format!("mtimes: provider {operation} failed: {source}"));
    error = crate::runtime_error::with_provider_execution_source(error, source.into());
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

fn compatible(handle: &GpuTensorHandle, provider: &'static dyn AccelProvider) -> bool {
    gpu_helpers::exact_provider_for_handle(handle)
        .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle) == Some(provider.precision())
}
