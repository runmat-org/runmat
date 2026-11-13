//! One callback-result boundary: lease/shape admission, owner-preserving
//! materialization, and cooperative numeric decoding. No callback is invoked here.

use std::borrow::Cow;

use runmat_builtins::INTEGRAL_ERROR_INVALID_INPUT;
use runmat_value::Value;

use super::{invalid, resource_error, COMPONENT_CHUNK};
use crate::builtins::common::gpu_helpers;
use crate::builtins::math::optim::integral::{integral_map_error, waypoints, NAME};
use crate::builtins::math::optim::quadrature::{QuadratureCooperation, ScaledComplex};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

pub(in super::super) fn has_complex_class(value: &Value) -> bool {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => true,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        }
        _ => false,
    }
}

pub(super) async fn materialize(value: &Value) -> BuiltinResult<Cow<'_, Value>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(Cow::Borrowed(value));
    };
    let readback =
        runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
            let error = build_runtime_error("integral: resident value has no owning provider")
                .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
                .with_builtin(NAME)
                .with_gpu_gather_retry(GpuGatherRetry::Never)
                .build();
            crate::runtime_error::with_provider_execution_source(error, source.into())
        })?;
    let runtime = crate::context::legacy::active();
    let downloaded = gpu_helpers::download_readback_value_in_context(runtime.as_ref(), &readback)
        .await
        .map_err(|source| {
            let mut error = integral_map_error(source, &INTEGRAL_ERROR_INVALID_INPUT);
            error.gpu_gather_retry = GpuGatherRetry::Never;
            error
        })?;
    Ok(Cow::Owned(downloaded))
}

pub(super) fn shape(value: &Value) -> BuiltinResult<&[usize]> {
    use runmat_accelerate_api::{handle_storage, GpuTensorStorage};
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(&[1, 1]),
        Value::Tensor(array) => Ok(array.shape()),
        Value::ComplexTensor(array) => Ok(array.shape()),
        Value::LogicalArray(array) => Ok(&array.shape),
        Value::GpuTensor(handle) => {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|source| {
                    build_runtime_error("integral: callback returned an invalid resident lease")
                        .with_builtin(NAME)
                        .with_source(source)
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build()
                })?;
            if !matches!(
                handle_storage(handle),
                GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
            ) {
                return Err(invalid(
                    "callback must return real or complex numeric values",
                ));
            }
            Ok(handle.shape())
        }
        _ => Err(invalid(
            "callback must return real or complex numeric values",
        )),
    }
}

pub(super) async fn decode(
    value: &Value,
    expected: &[usize],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Vec<ScaledComplex>> {
    if shape(value)? != expected {
        return Err(invalid("complex callback changed its output size"));
    }
    let count = expected
        .iter()
        .try_fold(1_usize, |product, &axis| product.checked_mul(axis))
        .ok_or_else(|| invalid("callback output size overflows"))?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(resource_error)?;
    for index in 0..count {
        let item = match value {
            Value::Num(real) => ScaledComplex::new(*real, 0.0),
            Value::Int(integer) => ScaledComplex::new(integer.to_f64(), 0.0),
            Value::Bool(flag) => ScaledComplex::new(f64::from(*flag as u8), 0.0),
            Value::Complex(real, imaginary) => ScaledComplex::new(*real, *imaginary),
            Value::Tensor(array) => {
                let element = array
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid("invalid real callback storage"))?;
                ScaledComplex::new(waypoints::value_f64(element), 0.0)
            }
            Value::ComplexTensor(array) => {
                let (real, imaginary) = array
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid("invalid complex callback storage"))?;
                ScaledComplex::new(waypoints::value_f64(real), waypoints::value_f64(imaginary))
            }
            Value::LogicalArray(array) => {
                let bit = array
                    .data
                    .get(index)
                    .ok_or_else(|| invalid("invalid logical callback storage"))?;
                ScaledComplex::new(f64::from((*bit != 0) as u8), 0.0)
            }
            Value::GpuTensor(_) => return Err(invalid("resident callback was not materialized")),
            _ => {
                return Err(invalid(
                    "callback must return real or complex numeric values",
                ));
            }
        };
        if item.to_pair().is_none() {
            return Err(invalid("complex callback must return finite values"));
        }
        values.push(item);
        if (index + 1).is_multiple_of(COMPONENT_CHUNK) {
            cooperation.completed(COMPONENT_CHUNK as u64).await?;
        }
    }
    cooperation
        .completed((count % COMPONENT_CHUNK) as u64)
        .await?;
    Ok(values)
}
