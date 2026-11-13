//! Admitted dense output and scalar AMOS evaluation.

use complex_bessel::{besselj, besselj_scaled};
use num_complex::{Complex32, Complex64};
use runmat_builtins::BESSELJ_ERROR_NUMERICAL;
use runmat_value::{
    ComplexTensor, ExactHostComplexTensorAllocation, NumericDType, NumericStorage, Tensor, Value,
};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::input::{Argument, Order};
use super::{error, NAME};

const COOPERATION_CHUNK: usize = 128;

pub(super) async fn evaluate(
    runtime: Option<&RuntimeContext>,
    order: &Order<'_>,
    argument: &Argument<'_>,
    shape: &[usize],
    scaled: bool,
) -> BuiltinResult<Value> {
    let len = if order.len() == 0 || argument.len() == 0 {
        0
    } else {
        order.len().max(argument.len())
    };
    let dtype = if order.is_single() || argument.is_single() {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let mut cooperation = runtime.map(RuntimeContext::cpu_cooperation);
    cooperate(&mut cooperation, runtime, 0).await?;
    let complex = needs_complex(order, argument, len, &mut cooperation, runtime).await?;

    // A double scalar is a scalar Value and has no heap-backed output. Other
    // results use the canonical exact tensor owner under managed admission.
    if len == 1
        && dtype == NumericDType::F64
        && order.shape().is_none()
        && argument.shape().is_none()
    {
        let result = calculate_f64(order.get(0), argument.get(0), scaled)?;
        cooperate(&mut cooperation, runtime, 1).await?;
        return Ok(if complex {
            Value::Complex(result.re, result.im)
        } else {
            Value::Num(result.re)
        });
    }

    let retained = if complex {
        HostAllocationPlan::retained_complex_tensor_bytes_for_dimensions(dtype, len, shape.len())
    } else {
        HostAllocationPlan::retained_tensor_bytes_for_dimensions(dtype, len, shape.len())
    }
    .map_err(|source| allocation_error(runtime, source))?;
    let plan =
        HostAllocationPlan::new(0, retained).map_err(|source| allocation_error(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(runtime, plan)
        .map_err(|source| allocation_error(runtime, source))?;

    match (dtype, complex, runtime, receipt.as_ref()) {
        (NumericDType::F64, false, Some(runtime), Some(receipt)) => {
            let mut output = receipt
                .exact_output_tensor(runtime, dtype, shape)
                .map_err(|source| allocation_error(Some(runtime), source))?;
            let values = output.as_f64_slice_mut().expect("admitted double output");
            for (index, value) in values.iter_mut().enumerate() {
                checkpoint_element(&mut cooperation, Some(runtime), index).await?;
                *value = calculate_f64(order.get(index), argument.get(index), scaled)?.re;
            }
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(Some(runtime), source))?;
            let tensor = output.into_tensor().map_err(|source| {
                allocation_error(Some(runtime), HostAllocationError::Numeric(source))
            })?;
            Ok(Value::Tensor(tensor))
        }
        (NumericDType::F32, false, Some(runtime), Some(receipt)) => {
            let mut output = receipt
                .exact_output_tensor(runtime, dtype, shape)
                .map_err(|source| allocation_error(Some(runtime), source))?;
            let values = output.as_f32_slice_mut().expect("admitted single output");
            for (index, value) in values.iter_mut().enumerate() {
                checkpoint_element(&mut cooperation, Some(runtime), index).await?;
                *value = calculate_f32(order.get(index), argument.get(index), scaled)?.re;
            }
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(Some(runtime), source))?;
            let tensor = output.into_tensor().map_err(|source| {
                allocation_error(Some(runtime), HostAllocationError::Numeric(source))
            })?;
            Ok(Value::Tensor(tensor))
        }
        (NumericDType::F64 | NumericDType::F32, true, Some(runtime), Some(receipt)) => {
            let mut output = receipt
                .with_output_allocation(runtime, |permit| {
                    ExactHostComplexTensorAllocation::new(dtype, shape, permit)
                })
                .map_err(|source| allocation_error(Some(runtime), source))?;
            for index in 0..len {
                checkpoint_element(&mut cooperation, Some(runtime), index).await?;
                let value = if dtype == NumericDType::F64 {
                    calculate_f64(order.get(index), argument.get(index), scaled)?
                } else {
                    let single = calculate_f32(order.get(index), argument.get(index), scaled)?;
                    Complex64::new(f64::from(single.re), f64::from(single.im))
                };
                output.set(index, value.re, value.im).map_err(|source| {
                    allocation_error(Some(runtime), HostAllocationError::Numeric(source))
                })?;
            }
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(Some(runtime), source))?;
            let tensor = output.into_tensor().map_err(|source| {
                allocation_error(Some(runtime), HostAllocationError::Numeric(source))
            })?;
            Ok(Value::ComplexTensor(tensor))
        }
        (NumericDType::F64, false, _, None) => {
            let mut values = exact_vec(len, 0.0f64)?;
            for (index, value) in values.iter_mut().enumerate() {
                checkpoint_element(&mut cooperation, runtime, index).await?;
                *value = calculate_f64(order.get(index), argument.get(index), scaled)?.re;
            }
            let tensor =
                Tensor::from_numeric_storage(NumericStorage::F64(values), copied_shape(shape)?)
                    .map_err(|detail| internal(detail))?;
            Ok(Value::Tensor(tensor))
        }
        (NumericDType::F32, false, _, None) => {
            let mut values = exact_vec(len, 0.0f32)?;
            for (index, value) in values.iter_mut().enumerate() {
                checkpoint_element(&mut cooperation, runtime, index).await?;
                *value = calculate_f32(order.get(index), argument.get(index), scaled)?.re;
            }
            let tensor =
                Tensor::from_numeric_storage(NumericStorage::F32(values), copied_shape(shape)?)
                    .map_err(|detail| internal(detail))?;
            Ok(Value::Tensor(tensor))
        }
        (NumericDType::F64, true, _, None) => {
            let mut values = exact_vec(len, (0.0f64, 0.0f64))?;
            for (index, value) in values.iter_mut().enumerate() {
                checkpoint_element(&mut cooperation, runtime, index).await?;
                let result = calculate_f64(order.get(index), argument.get(index), scaled)?;
                *value = (result.re, result.im);
            }
            let tensor = ComplexTensor::new(values, copied_shape(shape)?)
                .map_err(|detail| internal(detail))?;
            Ok(Value::ComplexTensor(tensor))
        }
        (NumericDType::F32, true, _, None) => {
            let mut values = exact_vec(len, (0.0f32, 0.0f32))?;
            for (index, value) in values.iter_mut().enumerate() {
                checkpoint_element(&mut cooperation, runtime, index).await?;
                let result = calculate_f32(order.get(index), argument.get(index), scaled)?;
                *value = (result.re, result.im);
            }
            let tensor = ComplexTensor::from_f32(values, copied_shape(shape)?)
                .map_err(|detail| internal(detail))?;
            Ok(Value::ComplexTensor(tensor))
        }
        _ => unreachable!("floating output class"),
    }
}

async fn needs_complex(
    order: &Order<'_>,
    argument: &Argument<'_>,
    len: usize,
    cooperation: &mut Option<CpuCooperationBudget>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<bool> {
    if argument.is_complex() {
        return Ok(true);
    }
    for index in 0..len {
        checkpoint_element(cooperation, runtime, index).await?;
        let (real, _) = argument.get(index);
        let nu = order.get(index);
        if real < 0.0 && nu.fract() != 0.0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn calculate_f64(order: f64, argument: (f64, f64), scaled: bool) -> BuiltinResult<Complex64> {
    if order.is_nan() || argument.0.is_nan() || argument.1.is_nan() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    let z = Complex64::new(argument.0, argument.1);
    let result = if scaled {
        besselj_scaled(order, z)
    } else {
        besselj(order, z)
    };
    result.map_err(|source| numerical_error(source))
}

fn calculate_f32(order: f64, argument: (f64, f64), scaled: bool) -> BuiltinResult<Complex32> {
    if order.is_nan() || argument.0.is_nan() || argument.1.is_nan() {
        return Ok(Complex32::new(f32::NAN, f32::NAN));
    }
    let z = Complex32::new(argument.0 as f32, argument.1 as f32);
    let result = if scaled {
        besselj_scaled(order as f32, z)
    } else {
        besselj(order as f32, z)
    };
    result.map_err(|source| numerical_error(source))
}

fn numerical_error(source: complex_bessel::Error) -> RuntimeError {
    build_runtime_error(format!("{}: {source}", BESSELJ_ERROR_NUMERICAL.message))
        .with_builtin(NAME)
        .with_identifier(
            BESSELJ_ERROR_NUMERICAL
                .identifier
                .expect("numerical identifier"),
        )
        .with_source(source)
        .build()
}

async fn checkpoint_element(
    cooperation: &mut Option<CpuCooperationBudget>,
    runtime: Option<&RuntimeContext>,
    index: usize,
) -> BuiltinResult<()> {
    if index.is_multiple_of(COOPERATION_CHUNK) {
        cooperate(cooperation, runtime, COOPERATION_CHUNK as u64).await?;
    }
    Ok(())
}

async fn cooperate(
    cooperation: &mut Option<CpuCooperationBudget>,
    runtime: Option<&RuntimeContext>,
    work: u64,
) -> BuiltinResult<()> {
    if let Some(cooperation) = cooperation.as_mut() {
        cooperation
            .checkpoint(work)
            .await
            .map_err(|source: ExecutionServiceError| {
                crate::builtins::common::map_control_flow_with_builtin(
                    source.into_runtime_error_in_context(runtime),
                    NAME,
                )
            })?;
    }
    Ok(())
}

fn allocation_error(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), NAME, source)
}

fn exact_vec<T: Clone>(len: usize, initial: T) -> BuiltinResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|source| internal(source.to_string()))?;
    values.resize(len, initial);
    Ok(values)
}

fn copied_shape(shape: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(shape.len())
        .map_err(|source| internal(source.to_string()))?;
    result.extend_from_slice(shape);
    Ok(result)
}

fn internal(detail: impl AsRef<str>) -> RuntimeError {
    error(&runmat_builtins::BESSELJ_ERROR_INTERNAL, detail)
}
