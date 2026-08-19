//! Exact relational comparisons for native MATLAB integer storage.

use std::cmp::Ordering;

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::{
    ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericScalar, Tensor, Value,
};

use crate::builtins::common::broadcast::{broadcast_shapes, BroadcastPlan};
use crate::builtins::common::gpu_helpers;

#[derive(Clone, Copy)]
pub enum IntegerComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug)]
pub enum IntegerComparisonError {
    SizeMismatch,
    Internal,
}

pub(crate) async fn try_gpu_equality_comparison(
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    operation: IntegerComparisonOp,
) -> Option<crate::BuiltinResult<Value>> {
    if lhs.device_id != rhs.device_id {
        return None;
    }
    let provider = resolved_actual_owner(lhs)?;
    let rhs_owner = resolved_actual_owner(rhs)?;
    if !std::ptr::eq(provider, rhs_owner)
        || runmat_accelerate_api::handle_precision(lhs)
            != runmat_accelerate_api::handle_precision(rhs)
    {
        return None;
    }
    let result = match operation {
        IntegerComparisonOp::Eq => provider.elem_eq(lhs, rhs).await,
        IntegerComparisonOp::Ne => provider.elem_ne(lhs, rhs).await,
        _ => unreachable!("equality helper only supports eq and ne"),
    };
    match result {
        Ok(mut handle) if valid_equality_output(&handle, lhs, rhs, provider) => {
            let provenance = [lhs, rhs]
                .into_iter()
                .filter_map(runmat_accelerate_api::handle_provenance)
                .find(|provenance| {
                    *provenance == runmat_accelerate_api::GpuHandleProvenance::Explicit
                })
                .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            runmat_accelerate_api::set_handle_provenance(&mut handle, provenance);
            Some(Ok(gpu_helpers::logical_gpu_value(handle)))
        }
        Ok(handle) => {
            free_rejected_gpu_handle(&handle, &[lhs, rhs]);
            None
        }
        Err(_) => None,
    }
}

pub(crate) fn select_comparison_output_source(
    lhs: &Value,
    rhs: &Value,
    builtin: &str,
) -> crate::BuiltinResult<Option<GpuTensorHandle>> {
    gpu_helpers::select_resident_output_source(
        [lhs, rhs].into_iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        builtin,
    )
}

pub(crate) fn restore_explicit_comparison_result(
    value: Value,
    source: Option<&GpuTensorHandle>,
    builtin: &str,
) -> crate::BuiltinResult<Value> {
    let Some(source) = source.filter(|handle| runmat_accelerate_api::handle_is_explicit(handle))
    else {
        return Ok(value);
    };
    let value = match value {
        Value::Bool(bit) => Value::LogicalArray(
            LogicalArray::new(vec![u8::from(bit)], vec![1, 1]).map_err(|error| {
                crate::build_runtime_error(format!(
                    "{builtin}: invalid scalar logical result: {error}"
                ))
                .with_builtin(builtin)
                .build()
            })?,
        ),
        value => value,
    };
    let restored = gpu_helpers::restore_class_preserving_value(source, value, builtin)?;
    if !matches!(restored, Value::GpuTensor(_)) {
        return Err(crate::build_runtime_error(format!(
            "{builtin}: provider cannot preserve explicit gpuArray output residency"
        ))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:GpuUploadFailed"))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    Ok(restored)
}

/// Executes a resident ordering comparison, extracting real lanes first when
/// either input is complex-interleaved. The logical result remains resident.
pub(crate) async fn try_gpu_ordering_comparison(
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    operation: IntegerComparisonOp,
) -> Option<crate::BuiltinResult<Value>> {
    if lhs.device_id != rhs.device_id {
        return None;
    }
    let provider = resolved_actual_owner(lhs)?;
    let rhs_owner = resolved_actual_owner(rhs)?;
    if !std::ptr::eq(provider, rhs_owner) {
        return None;
    }
    let mut temporary_lhs = None;
    let mut temporary_rhs = None;
    let lhs_real =
        if runmat_accelerate_api::handle_storage(lhs) == GpuTensorStorage::ComplexInterleaved {
            match provider.unary_real(lhs).await {
                Ok(handle) if valid_real_projection(&handle, lhs, provider) => {
                    temporary_lhs = Some(handle);
                    temporary_lhs.as_ref().expect("temporary lhs")
                }
                Ok(handle) => {
                    free_rejected_gpu_handle(&handle, &[lhs, rhs]);
                    return None;
                }
                Err(_) => return None,
            }
        } else {
            lhs
        };
    let rhs_real =
        if runmat_accelerate_api::handle_storage(rhs) == GpuTensorStorage::ComplexInterleaved {
            match provider.unary_real(rhs).await {
                Ok(handle) if valid_real_projection(&handle, rhs, provider) => {
                    temporary_rhs = Some(handle);
                    temporary_rhs.as_ref().expect("temporary rhs")
                }
                Ok(handle) => {
                    free_rejected_gpu_handle(&handle, &[lhs, rhs, lhs_real]);
                    if let Some(handle) = temporary_lhs.as_ref() {
                        let _ = provider.free(handle);
                    }
                    return None;
                }
                Err(_) => {
                    if let Some(handle) = temporary_lhs.as_ref() {
                        let _ = provider.free(handle);
                    }
                    return None;
                }
            }
        } else {
            rhs
        };
    if runmat_accelerate_api::handle_precision(lhs_real)
        != runmat_accelerate_api::handle_precision(rhs_real)
    {
        if let Some(handle) = temporary_lhs.as_ref() {
            let _ = provider.free(handle);
        }
        if let Some(handle) = temporary_rhs.as_ref() {
            let _ = provider.free(handle);
        }
        return None;
    }
    let result = match operation {
        IntegerComparisonOp::Lt => provider.elem_lt(lhs_real, rhs_real).await,
        IntegerComparisonOp::Le => provider.elem_le(lhs_real, rhs_real).await,
        IntegerComparisonOp::Gt => provider.elem_gt(lhs_real, rhs_real).await,
        IntegerComparisonOp::Ge => provider.elem_ge(lhs_real, rhs_real).await,
        IntegerComparisonOp::Eq | IntegerComparisonOp::Ne => {
            unreachable!("resident complex ordering helper only supports lt/le/gt/ge")
        }
    };
    let result = match result {
        Ok(mut handle) if valid_ordering_output(&handle, lhs_real, rhs_real, provider) => {
            let provenance = [lhs, rhs]
                .into_iter()
                .filter_map(runmat_accelerate_api::handle_provenance)
                .find(|provenance| {
                    *provenance == runmat_accelerate_api::GpuHandleProvenance::Explicit
                })
                .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            runmat_accelerate_api::set_handle_provenance(&mut handle, provenance);
            Some(gpu_helpers::logical_gpu_value(handle))
        }
        Ok(handle) => {
            free_rejected_gpu_handle(&handle, &[lhs, rhs, lhs_real, rhs_real]);
            None
        }
        Err(_) => None,
    };
    if let Some(handle) = temporary_lhs.as_ref() {
        let _ = provider.free(handle);
    }
    if let Some(handle) = temporary_rhs.as_ref() {
        let _ = provider.free(handle);
    }
    result.map(Ok)
}

fn resolved_actual_owner(
    handle: &GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    runmat_accelerate_api::provider_for_handle(handle)
        .filter(|owner| owner.device_id() == handle.device_id)
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn valid_real_projection(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && resolved_actual_owner(output).is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn valid_ordering_output(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let expected_shape = broadcast_shapes("comparison", &lhs.shape, &rhs.shape).ok();
    expected_shape.as_deref() == Some(output.shape.as_slice())
        && output.device_id == lhs.device_id
        && !gpu_handles_alias(output, lhs)
        && !gpu_handles_alias(output, rhs)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(lhs)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && resolved_actual_owner(output).is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn valid_equality_output(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let expected_shape = broadcast_shapes("comparison", &lhs.shape, &rhs.shape).ok();
    expected_shape.as_deref() == Some(output.shape.as_slice())
        && output.device_id == lhs.device_id
        && !gpu_handles_alias(output, lhs)
        && !gpu_handles_alias(output, rhs)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(lhs)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && resolved_actual_owner(output).is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn free_rejected_gpu_handle(handle: &GpuTensorHandle, protected: &[&GpuTensorHandle]) {
    if protected
        .iter()
        .any(|protected| gpu_handles_alias(handle, protected))
    {
        return;
    }
    if let Some(owner) = resolved_actual_owner(handle) {
        let _ = owner.free(handle);
    }
}

/// Performs a comparison when native integer storage is compared against other
/// native integer storage or real numeric storage. This keeps integer values
/// exact even when the other operand is an f64 array.
pub fn try_integer_comparison(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerComparisonOp,
) -> Result<Option<Value>, IntegerComparisonError> {
    let lhs_integer = integer_operand(lhs);
    let rhs_integer = integer_operand(rhs);
    let result = match (lhs_integer, rhs_integer) {
        (None, None) => return Ok(None),
        (Some(lhs), Some(rhs)) => compare_integer_operands(&lhs, &rhs, operation)?,
        (Some(lhs), None) => {
            let Some(rhs) = numeric_operand(rhs) else {
                return Ok(None);
            };
            compare_integer_numeric(&lhs, &rhs, true, operation)?
        }
        (None, Some(rhs)) => {
            let Some(lhs) = numeric_operand(lhs) else {
                return Ok(None);
            };
            compare_integer_numeric(&rhs, &lhs, false, operation)?
        }
    };
    Ok(Some(result))
}

/// Performs an exact host ordering comparison for real numeric/logical/character operands.
/// Native integers remain in integer storage and mixed integer/floating comparisons use the
/// lossless MATLAB relational ordering rules.
pub fn try_real_ordering_comparison(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerComparisonOp,
) -> Result<Option<Value>, IntegerComparisonError> {
    let Some(lhs) = real_operand(lhs) else {
        return Ok(None);
    };
    let Some(rhs) = real_operand(rhs) else {
        return Ok(None);
    };
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, lhs_index, rhs_index) in plan.iter() {
        let ordering = compare_real_values(lhs.value_at(lhs_index), rhs.value_at(rhs_index));
        data.push(matches_optional_relation(ordering, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec()).map(Some)
}

/// Performs exact equality/inequality when a complex operand or its real
/// counterpart carries native integer storage.
pub(crate) fn try_complex_integer_equality_comparison(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerComparisonOp,
) -> Result<Option<Value>, IntegerComparisonError> {
    debug_assert!(matches!(
        operation,
        IntegerComparisonOp::Eq | IntegerComparisonOp::Ne
    ));
    let lhs_complex = complex_operand(lhs);
    let rhs_complex = complex_operand(rhs);
    let result = match (lhs_complex, rhs_complex) {
        (Some(lhs), Some(rhs)) if lhs.has_integer_storage() || rhs.has_integer_storage() => {
            compare_complex_operands(&lhs, &rhs, operation)?
        }
        (Some(lhs), None) => {
            let Some(rhs) = real_operand(rhs) else {
                return Ok(None);
            };
            if !lhs.has_integer_storage() && !rhs.has_integer_storage() {
                return Ok(None);
            }
            compare_complex_real(&lhs, &rhs, operation)?
        }
        (None, Some(rhs)) => {
            let Some(lhs) = real_operand(lhs) else {
                return Ok(None);
            };
            if !rhs.has_integer_storage() && !lhs.has_integer_storage() {
                return Ok(None);
            }
            compare_complex_real(&rhs, &lhs, operation)?
        }
        _ => return Ok(None),
    };
    Ok(Some(result))
}

/// Performs MATLAB ordering comparisons for complex values by comparing only
/// their real components. Integer-backed components remain exact across
/// complex/complex and complex/real mixed-class comparisons.
pub fn try_complex_ordering_comparison(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerComparisonOp,
) -> Result<Option<Value>, IntegerComparisonError> {
    debug_assert!(matches!(
        operation,
        IntegerComparisonOp::Lt
            | IntegerComparisonOp::Le
            | IntegerComparisonOp::Gt
            | IntegerComparisonOp::Ge
    ));
    let lhs_complex = complex_operand(lhs);
    let rhs_complex = complex_operand(rhs);
    let result = match (lhs_complex, rhs_complex) {
        (Some(lhs), Some(rhs)) => compare_complex_real_components(&lhs, &rhs, operation)?,
        (Some(lhs), None) => {
            let Some(rhs) = real_operand(rhs) else {
                return Ok(None);
            };
            compare_complex_real_ordering(&lhs, &rhs, true, operation)?
        }
        (None, Some(rhs)) => {
            let Some(lhs) = real_operand(lhs) else {
                return Ok(None);
            };
            compare_complex_real_ordering(&rhs, &lhs, false, operation)?
        }
        (None, None) => return Ok(None),
    };
    Ok(Some(result))
}

fn compare_integer_operands(
    lhs: &IntegerOperand<'_>,
    rhs: &IntegerOperand<'_>,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, lhs_index, rhs_index) in plan.iter() {
        let ordering = compare_integer_values(lhs.value_at(lhs_index), rhs.value_at(rhs_index));
        data.push(matches_relation(ordering, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn compare_integer_numeric(
    integer: &IntegerOperand<'_>,
    numeric: &NumericOperand<'_>,
    integer_is_left: bool,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&integer.shape, numeric.shape())
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, integer_index, numeric_index) in plan.iter() {
        let ordering = integer_f64_order(
            integer.value_at(integer_index),
            numeric.value_at(numeric_index),
        );
        let ordering = if integer_is_left {
            ordering
        } else {
            ordering.map(Ordering::reverse)
        };
        data.push(matches_optional_relation(ordering, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn logical_result(data: Vec<u8>, shape: Vec<usize>) -> Result<Value, IntegerComparisonError> {
    if data.len() == 1 {
        return Ok(Value::Bool(data[0] != 0));
    }
    Ok(Value::LogicalArray(
        LogicalArray::new(data, shape).map_err(|_| IntegerComparisonError::Internal)?,
    ))
}

struct ComplexOperand<'a> {
    source: ComplexSource<'a>,
    shape: Vec<usize>,
}

impl ComplexOperand<'_> {
    fn has_integer_storage(&self) -> bool {
        match self.source {
            ComplexSource::Scalar(_, _) => false,
            ComplexSource::Dense(tensor) => tensor.integer_storage().is_some(),
        }
    }

    fn real_imag_at(&self, index: usize) -> ComplexValue {
        match self.source {
            ComplexSource::Scalar(real, imag) => ComplexValue::Float(real, imag),
            ComplexSource::Dense(tensor) => complex_value_from_scalars(
                tensor
                    .numeric_value_at(index)
                    .expect("complex tensor storage must match shape"),
            ),
        }
    }
}

enum ComplexSource<'a> {
    Scalar(f64, f64),
    Dense(&'a ComplexTensor),
}

enum ComplexValue {
    Integer(IntValue, IntValue),
    Float(f64, f64),
}

enum RealValue {
    Integer(IntValue),
    Float(f64),
}

fn real_value_from_scalar(value: NumericScalar) -> RealValue {
    match value {
        NumericScalar::F64(value) => RealValue::Float(value),
        NumericScalar::F32(value) => RealValue::Float(f64::from(value)),
        integer => RealValue::Integer(
            integer
                .into_int_value()
                .expect("non-floating numeric scalar must be integer"),
        ),
    }
}

fn complex_value_from_scalars((real, imag): (NumericScalar, NumericScalar)) -> ComplexValue {
    match (real.into_int_value(), imag.into_int_value()) {
        (Some(real), Some(imag)) => ComplexValue::Integer(real, imag),
        (None, None) => {
            ComplexValue::Float(floating_scalar_to_f64(real), floating_scalar_to_f64(imag))
        }
        _ => unreachable!("complex storage components must use the same numeric domain"),
    }
}

fn floating_scalar_to_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        _ => unreachable!("expected floating numeric scalar"),
    }
}

struct RealOperand<'a> {
    source: RealSource<'a>,
    shape: Vec<usize>,
}

impl RealOperand<'_> {
    fn has_integer_storage(&self) -> bool {
        match self.source {
            RealSource::ScalarInteger(_) => true,
            RealSource::Dense(tensor) => tensor.integer_storage().is_some(),
            RealSource::ScalarFloat(_) | RealSource::Logical { .. } | RealSource::Char(_) => false,
        }
    }

    fn value_at(&self, index: usize) -> RealValue {
        match self.source {
            RealSource::ScalarInteger(ref value) => RealValue::Integer(value.clone()),
            RealSource::ScalarFloat(value) => RealValue::Float(value),
            RealSource::Dense(tensor) => real_value_from_scalar(
                tensor
                    .numeric_value_at(index)
                    .expect("tensor storage must match shape"),
            ),
            RealSource::Logical { data } => RealValue::Float(f64::from(data[index] != 0)),
            RealSource::Char(array) => {
                let row = index % array.rows;
                let column = index / array.rows;
                RealValue::Float(f64::from(array.data[row * array.cols + column] as u32))
            }
        }
    }
}

enum RealSource<'a> {
    ScalarInteger(IntValue),
    ScalarFloat(f64),
    Dense(&'a Tensor),
    Logical { data: &'a [u8] },
    Char(&'a runmat_value::CharArray),
}

fn compare_complex_operands(
    lhs: &ComplexOperand<'_>,
    rhs: &ComplexOperand<'_>,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, lhs_index, rhs_index) in plan.iter() {
        let matches =
            complex_values_equal(lhs.real_imag_at(lhs_index), rhs.real_imag_at(rhs_index));
        data.push(matches_relation_bool(matches, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn compare_complex_real_components(
    lhs: &ComplexOperand<'_>,
    rhs: &ComplexOperand<'_>,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, lhs_index, rhs_index) in plan.iter() {
        let ordering = compare_real_values(
            complex_real_component(lhs.real_imag_at(lhs_index)),
            complex_real_component(rhs.real_imag_at(rhs_index)),
        );
        data.push(matches_optional_relation(ordering, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn compare_complex_real_ordering(
    complex: &ComplexOperand<'_>,
    real: &RealOperand<'_>,
    complex_is_left: bool,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&complex.shape, &real.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, complex_index, real_index) in plan.iter() {
        let ordering = compare_real_values(
            complex_real_component(complex.real_imag_at(complex_index)),
            real.value_at(real_index),
        );
        let ordering = if complex_is_left {
            ordering
        } else {
            ordering.map(Ordering::reverse)
        };
        data.push(matches_optional_relation(ordering, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn complex_real_component(value: ComplexValue) -> RealValue {
    match value {
        ComplexValue::Integer(real, _) => RealValue::Integer(real),
        ComplexValue::Float(real, _) => RealValue::Float(real),
    }
}

fn compare_real_values(lhs: RealValue, rhs: RealValue) -> Option<Ordering> {
    match (lhs, rhs) {
        (RealValue::Integer(lhs), RealValue::Integer(rhs)) => {
            Some(compare_integer_values(lhs, rhs))
        }
        (RealValue::Integer(lhs), RealValue::Float(rhs)) => integer_f64_order(lhs, rhs),
        (RealValue::Float(lhs), RealValue::Integer(rhs)) => {
            integer_f64_order(rhs, lhs).map(Ordering::reverse)
        }
        (RealValue::Float(lhs), RealValue::Float(rhs)) => lhs.partial_cmp(&rhs),
    }
}

/// Compares two native real numeric scalars without routing either integer
/// operand through binary64. Every `f32` value is represented exactly as
/// `f64`, so the mixed floating path retains the same ordering semantics as
/// the array relational operators.
pub(crate) fn compare_numeric_scalars_exact(
    lhs: NumericScalar,
    rhs: NumericScalar,
) -> Option<Ordering> {
    fn real(value: NumericScalar) -> RealValue {
        match value {
            NumericScalar::F64(value) => RealValue::Float(value),
            NumericScalar::F32(value) => RealValue::Float(f64::from(value)),
            value => RealValue::Integer(
                value
                    .into_int_value()
                    .expect("nonfloating NumericScalar must contain an integer"),
            ),
        }
    }
    compare_real_values(real(lhs), real(rhs))
}

fn compare_complex_real(
    complex: &ComplexOperand<'_>,
    real: &RealOperand<'_>,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&complex.shape, &real.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, complex_index, real_index) in plan.iter() {
        let matches = complex_value_equals_real(
            complex.real_imag_at(complex_index),
            real.value_at(real_index),
        );
        data.push(matches_relation_bool(matches, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn complex_values_equal(lhs: ComplexValue, rhs: ComplexValue) -> bool {
    match (lhs, rhs) {
        (ComplexValue::Integer(lhs_real, lhs_imag), ComplexValue::Integer(rhs_real, rhs_imag)) => {
            compare_integer_values(lhs_real, rhs_real) == Ordering::Equal
                && compare_integer_values(lhs_imag, rhs_imag) == Ordering::Equal
        }
        (ComplexValue::Integer(real, imag), ComplexValue::Float(rhs_real, rhs_imag)) => {
            integer_value_equals_f64(real, rhs_real) && integer_value_equals_f64(imag, rhs_imag)
        }
        (ComplexValue::Float(lhs_real, lhs_imag), ComplexValue::Integer(real, imag)) => {
            integer_value_equals_f64(real, lhs_real) && integer_value_equals_f64(imag, lhs_imag)
        }
        (ComplexValue::Float(lhs_real, lhs_imag), ComplexValue::Float(rhs_real, rhs_imag)) => {
            lhs_real == rhs_real && lhs_imag == rhs_imag
        }
    }
}

fn complex_value_equals_real(complex: ComplexValue, real: RealValue) -> bool {
    match (complex, real) {
        (ComplexValue::Integer(complex_real, complex_imag), RealValue::Integer(real)) => {
            complex_imag.is_zero() && compare_integer_values(complex_real, real) == Ordering::Equal
        }
        (ComplexValue::Integer(complex_real, complex_imag), RealValue::Float(real)) => {
            complex_imag.is_zero() && integer_value_equals_f64(complex_real, real)
        }
        (ComplexValue::Float(complex_real, complex_imag), RealValue::Integer(real)) => {
            complex_imag == 0.0 && integer_value_equals_f64(real, complex_real)
        }
        (ComplexValue::Float(complex_real, complex_imag), RealValue::Float(real)) => {
            complex_imag == 0.0 && complex_real == real
        }
    }
}

fn integer_value_equals_f64(integer: IntValue, float: f64) -> bool {
    integer_f64_order(integer, float) == Some(Ordering::Equal)
}

fn matches_relation_bool(matches: bool, operation: IntegerComparisonOp) -> bool {
    match operation {
        IntegerComparisonOp::Eq => matches,
        IntegerComparisonOp::Ne => !matches,
        IntegerComparisonOp::Lt
        | IntegerComparisonOp::Le
        | IntegerComparisonOp::Gt
        | IntegerComparisonOp::Ge => {
            unreachable!("complex integer equality helper only supports eq/ne")
        }
    }
}

fn complex_operand(value: &Value) -> Option<ComplexOperand<'_>> {
    match value {
        Value::Complex(real, imag) => Some(ComplexOperand {
            source: ComplexSource::Scalar(*real, *imag),
            shape: vec![1, 1],
        }),
        Value::ComplexTensor(tensor) => Some(complex_tensor_operand(tensor)),
        _ => None,
    }
}

fn complex_tensor_operand(tensor: &ComplexTensor) -> ComplexOperand<'_> {
    ComplexOperand {
        source: ComplexSource::Dense(tensor),
        shape: tensor.shape.clone(),
    }
}

struct IntegerOperand<'a> {
    storage: IntegerStorageRef<'a>,
    shape: Vec<usize>,
}

impl IntegerOperand<'_> {
    fn value_at(&self, index: usize) -> IntValue {
        self.storage.value_at(index)
    }
}

enum IntegerStorageRef<'a> {
    Scalar(&'a IntValue),
    Array(&'a IntegerStorage),
}

impl IntegerStorageRef<'_> {
    fn value_at(&self, index: usize) -> IntValue {
        match self {
            Self::Scalar(value) => (*value).clone(),
            Self::Array(storage) => storage_value(storage, index),
        }
    }
}

fn integer_operand(value: &Value) -> Option<IntegerOperand<'_>> {
    match value {
        Value::Int(value) => Some(IntegerOperand {
            storage: IntegerStorageRef::Scalar(value),
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor.integer_storage().map(|storage| IntegerOperand {
            storage: IntegerStorageRef::Array(storage),
            shape: tensor.shape.clone(),
        }),
        _ => None,
    }
}

fn real_operand(value: &Value) -> Option<RealOperand<'_>> {
    match value {
        Value::Int(value) => Some(RealOperand {
            source: RealSource::ScalarInteger(value.clone()),
            shape: vec![1, 1],
        }),
        Value::Num(value) => Some(RealOperand {
            source: RealSource::ScalarFloat(*value),
            shape: vec![1, 1],
        }),
        Value::Bool(value) => Some(RealOperand {
            source: RealSource::ScalarFloat(if *value { 1.0 } else { 0.0 }),
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => Some(RealOperand {
            source: RealSource::Dense(tensor),
            shape: tensor.shape.clone(),
        }),
        Value::LogicalArray(array) => Some(RealOperand {
            source: RealSource::Logical { data: &array.data },
            shape: array.shape.clone(),
        }),
        Value::CharArray(array) => Some(RealOperand {
            source: RealSource::Char(array),
            shape: vec![array.rows, array.cols],
        }),
        _ => None,
    }
}

pub(crate) fn compare_integer_values(lhs: IntValue, rhs: IntValue) -> Ordering {
    match (signed_value(&lhs), signed_value(&rhs)) {
        (Some(lhs), Some(rhs)) => lhs.cmp(&rhs),
        (None, None) => unsigned_value(&lhs).cmp(&unsigned_value(&rhs)),
        (Some(lhs), None) => {
            if lhs < 0 {
                Ordering::Less
            } else {
                (lhs as u64).cmp(&unsigned_value(&rhs))
            }
        }
        (None, Some(rhs)) => {
            if rhs < 0 {
                Ordering::Greater
            } else {
                unsigned_value(&lhs).cmp(&(rhs as u64))
            }
        }
    }
}

pub(crate) fn integer_f64_order(integer: IntValue, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if float == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }

    const MIN_I64: f64 = -9_223_372_036_854_775_808.0;
    const U64_EXCLUSIVE_UPPER: f64 = 18_446_744_073_709_551_616.0;
    if float < MIN_I64 {
        return Some(Ordering::Greater);
    }
    if float >= U64_EXCLUSIVE_UPPER {
        return Some(Ordering::Less);
    }

    let integer = integer_as_i128(&integer);
    let truncated = float as i128;
    let ordering = integer.cmp(&truncated);
    if float.fract() == 0.0 {
        return Some(ordering);
    }
    Some(if float.is_sign_positive() {
        if ordering == Ordering::Greater {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    } else if ordering == Ordering::Less {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

fn integer_as_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

enum NumericOperand<'a> {
    Scalar(f64),
    Dense(&'a Tensor),
    Logical(&'a [u8], &'a [usize]),
}

impl NumericOperand<'_> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Scalar(_) => &[1, 1],
            Self::Dense(tensor) => &tensor.shape,
            Self::Logical(_, shape) => shape,
        }
    }

    fn value_at(&self, index: usize) -> f64 {
        match self {
            Self::Scalar(value) => *value,
            Self::Dense(tensor) => match tensor
                .numeric_value_at(index)
                .expect("tensor storage must match shape")
            {
                NumericScalar::F64(value) => value,
                NumericScalar::F32(value) => f64::from(value),
                _ => unreachable!("integer tensors use the exact integer operand path"),
            },
            Self::Logical(data, _) => f64::from(data[index] != 0),
        }
    }
}

fn numeric_operand(value: &Value) -> Option<NumericOperand<'_>> {
    match value {
        Value::Num(value) => Some(NumericOperand::Scalar(*value)),
        Value::Bool(value) => Some(NumericOperand::Scalar(if *value { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
            Some(NumericOperand::Dense(tensor))
        }
        Value::LogicalArray(array) => Some(NumericOperand::Logical(&array.data, &array.shape)),
        _ => None,
    }
}

fn signed_value(value: &IntValue) -> Option<i64> {
    match value {
        IntValue::I8(value) => Some(*value as i64),
        IntValue::I16(value) => Some(*value as i64),
        IntValue::I32(value) => Some(*value as i64),
        IntValue::I64(value) => Some(*value),
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => None,
    }
}

fn unsigned_value(value: &IntValue) -> u64 {
    match value {
        IntValue::U8(value) => *value as u64,
        IntValue::U16(value) => *value as u64,
        IntValue::U32(value) => *value as u64,
        IntValue::U64(value) => *value,
        IntValue::I8(_) | IntValue::I16(_) | IntValue::I32(_) | IntValue::I64(_) => {
            unreachable!("unsigned conversion is only used for unsigned integer values")
        }
    }
}

pub(crate) fn storage_value(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

pub(crate) fn matches_relation(ordering: Ordering, operation: IntegerComparisonOp) -> bool {
    match operation {
        IntegerComparisonOp::Eq => ordering == Ordering::Equal,
        IntegerComparisonOp::Ne => ordering != Ordering::Equal,
        IntegerComparisonOp::Lt => ordering == Ordering::Less,
        IntegerComparisonOp::Le => ordering != Ordering::Greater,
        IntegerComparisonOp::Gt => ordering == Ordering::Greater,
        IntegerComparisonOp::Ge => ordering != Ordering::Less,
    }
}

pub(crate) fn matches_optional_relation(
    ordering: Option<Ordering>,
    operation: IntegerComparisonOp,
) -> bool {
    match ordering {
        Some(ordering) => matches_relation(ordering, operation),
        None => matches!(operation, IntegerComparisonOp::Ne),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_value::ComplexStorage;

    fn array(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(runmat_value::Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    #[test]
    fn compares_signed_unsigned_and_uint64_exactly() {
        let lhs = Value::Int(IntValue::U64(u64::MAX));
        let rhs = Value::Int(IntValue::I64(i64::MAX));
        assert_eq!(
            try_integer_comparison(&lhs, &rhs, IntegerComparisonOp::Gt).expect("comparison"),
            Some(Value::Bool(true))
        );
        assert_eq!(
            try_integer_comparison(
                &Value::Int(IntValue::I8(-1)),
                &Value::Int(IntValue::U8(0)),
                IntegerComparisonOp::Lt,
            )
            .expect("comparison"),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn broadcasts_exact_integer_arrays_for_all_relations() {
        let lhs = array(IntegerStorage::U64(vec![0, u64::MAX]), vec![2, 1]);
        let rhs = array(IntegerStorage::I64(vec![0, 1, i64::MAX]), vec![1, 3]);
        let result = try_integer_comparison(&lhs, &rhs, IntegerComparisonOp::Ge)
            .expect("comparison")
            .expect("integer path");
        assert_eq!(
            result,
            Value::LogicalArray(
                LogicalArray::new(vec![1, 1, 0, 1, 0, 1], vec![2, 3]).expect("logical result")
            )
        );
    }

    #[test]
    fn compares_integer_storage_to_scalar_double_without_64_bit_loss() {
        let exact = Value::Int(IntValue::U64((1_u64 << 53) + 1));
        let rounded = Value::Num((1_u64 << 53) as f64);
        assert_eq!(
            try_integer_comparison(&exact, &rounded, IntegerComparisonOp::Eq).expect("comparison"),
            Some(Value::Bool(false))
        );
        assert_eq!(
            try_integer_comparison(&exact, &rounded, IntegerComparisonOp::Gt).expect("comparison"),
            Some(Value::Bool(true))
        );

        let tensor = array(IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]), vec![1, 2]);
        assert_eq!(
            try_integer_comparison(&tensor, &rounded, IntegerComparisonOp::Ne).expect("comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![1, 1], vec![1, 2]).expect("logical result")
            ))
        );
    }

    #[test]
    fn compares_integer_storage_to_broadcast_float_arrays_without_64_bit_loss() {
        let integer = array(
            IntegerStorage::U64(vec![1_u64 << 53, (1_u64 << 53) + 1]),
            vec![2, 1],
        );
        let float = Value::Tensor(
            runmat_value::Tensor::new(
                vec![(1_u64 << 53) as f64, 0.0, (1_u64 << 53) as f64],
                vec![1, 3],
            )
            .expect("float tensor"),
        );
        let result = try_integer_comparison(&integer, &float, IntegerComparisonOp::Eq)
            .expect("comparison")
            .expect("integer path");
        assert_eq!(
            result,
            Value::LogicalArray(
                LogicalArray::new(vec![1, 0, 0, 0, 1, 0], vec![2, 3]).expect("logical result")
            )
        );

        let result = try_integer_comparison(&float, &integer, IntegerComparisonOp::Lt)
            .expect("comparison")
            .expect("integer path");
        assert_eq!(
            result,
            Value::LogicalArray(
                LogicalArray::new(vec![0, 1, 1, 1, 0, 1], vec![2, 3]).expect("logical result")
            )
        );
    }

    #[test]
    fn compares_integer_storage_to_logical_arrays() {
        let integer = array(IntegerStorage::I8(vec![0, 1]), vec![1, 2]);
        let logical =
            Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical array"));
        assert_eq!(
            try_integer_comparison(&integer, &logical, IntegerComparisonOp::Eq)
                .expect("comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![1, 1], vec![1, 2]).expect("logical result")
            ))
        );
    }

    #[test]
    fn compares_all_integer_storage_classes_to_complex_tensors_exactly() {
        let cases = [
            (
                IntegerStorage::I8(vec![-7, 5]),
                vec![(-7.0, 0.0), (0.0, 0.0)],
                vec![1, 0],
            ),
            (
                IntegerStorage::I16(vec![-300, 5]),
                vec![(-300.0, 0.0), (0.0, 0.0)],
                vec![1, 0],
            ),
            (
                IntegerStorage::I32(vec![-70_000, 5]),
                vec![(-70_000.0, 0.0), (0.0, 0.0)],
                vec![1, 0],
            ),
            (
                IntegerStorage::I64(vec![i64::MAX, -9_007_199_254_740_991]),
                vec![(i64::MAX as f64, 0.0), (-9_007_199_254_740_991.0, 0.0)],
                vec![0, 1],
            ),
            (
                IntegerStorage::U8(vec![7, 5]),
                vec![(7.0, 0.0), (0.0, 0.0)],
                vec![1, 0],
            ),
            (
                IntegerStorage::U16(vec![300, 5]),
                vec![(300.0, 0.0), (0.0, 0.0)],
                vec![1, 0],
            ),
            (
                IntegerStorage::U32(vec![70_000, 5]),
                vec![(70_000.0, 0.0), (0.0, 0.0)],
                vec![1, 0],
            ),
            (
                IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
                vec![((1_u64 << 53) as f64, 0.0), (u64::MAX as f64, 0.0)],
                vec![0, 0],
            ),
        ];

        for (storage, complex_data, expected_eq) in cases {
            let integer =
                runmat_value::Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
            let complex = Value::ComplexTensor(
                ComplexTensor::new(complex_data, vec![1, 2]).expect("complex tensor"),
            );
            let integer = Value::Tensor(integer);

            assert_eq!(
                try_complex_integer_equality_comparison(
                    &integer,
                    &complex,
                    IntegerComparisonOp::Eq,
                )
                .expect("comparison"),
                Some(Value::LogicalArray(
                    LogicalArray::new(expected_eq.clone(), vec![1, 2]).expect("logical result")
                ))
            );
            assert_eq!(
                try_complex_integer_equality_comparison(
                    &complex,
                    &integer,
                    IntegerComparisonOp::Ne,
                )
                .expect("comparison"),
                Some(Value::LogicalArray(
                    LogicalArray::new(
                        expected_eq.iter().map(|value| *value ^ 1).collect(),
                        vec![1, 2],
                    )
                    .expect("logical result")
                ))
            );
        }
    }

    #[test]
    fn complex_ordering_uses_only_real_components_for_all_relations() {
        let lhs = Value::Complex(2.0, f64::NAN);
        let rhs = Value::Complex(2.0, f64::INFINITY);
        for (operation, expected) in [
            (IntegerComparisonOp::Lt, false),
            (IntegerComparisonOp::Le, true),
            (IntegerComparisonOp::Gt, false),
            (IntegerComparisonOp::Ge, true),
        ] {
            assert_eq!(
                try_complex_ordering_comparison(&lhs, &rhs, operation).expect("comparison"),
                Some(Value::Bool(expected))
            );
        }

        let nan_real = Value::Complex(f64::NAN, 0.0);
        for operation in [
            IntegerComparisonOp::Lt,
            IntegerComparisonOp::Le,
            IntegerComparisonOp::Gt,
            IntegerComparisonOp::Ge,
        ] {
            assert_eq!(
                try_complex_ordering_comparison(&nan_real, &Value::Num(0.0), operation)
                    .expect("NaN comparison"),
                Some(Value::Bool(false))
            );
        }
        assert_eq!(
            try_complex_ordering_comparison(
                &Value::Complex(1.0, 0.0),
                &Value::Num(2.0),
                IntegerComparisonOp::Lt,
            )
            .expect("structurally complex zero-imaginary comparison"),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn complex_integer_ordering_preserves_wide_real_components_and_broadcasts() {
        let storage = runmat_value::IntegerComplexStorage::new(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 0]),
        )
        .expect("complex integer storage");
        let complex = Value::ComplexTensor(
            ComplexTensor::new_integer(storage, vec![2, 1]).expect("complex integer tensor"),
        );
        let rounded = Value::Tensor(
            Tensor::new(vec![(1_u64 << 53) as f64, u64::MAX as f64], vec![1, 2])
                .expect("double tensor"),
        );

        assert_eq!(
            try_complex_ordering_comparison(&complex, &rounded, IntegerComparisonOp::Gt)
                .expect("complex/double comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![1, 1, 0, 0], vec![2, 2]).expect("logical result")
            ))
        );
        assert_eq!(
            try_complex_ordering_comparison(&rounded, &complex, IntegerComparisonOp::Lt)
                .expect("double/complex comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![1, 1, 0, 0], vec![2, 2]).expect("logical result")
            ))
        );

        let rounded_complex = Value::ComplexTensor(
            ComplexTensor::new(
                vec![((1_u64 << 53) as f64, -1.0), (u64::MAX as f64, 1.0)],
                vec![1, 2],
            )
            .expect("floating complex tensor"),
        );
        assert_eq!(
            try_complex_ordering_comparison(&complex, &rounded_complex, IntegerComparisonOp::Gt,)
                .expect("integer-complex/floating-complex comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![1, 1, 0, 0], vec![2, 2]).expect("logical result")
            ))
        );
    }

    #[test]
    fn complex_single_and_double_ordering_broadcasts_without_using_imaginary_components() {
        let lhs = Value::ComplexTensor(
            ComplexTensor::from_complex_storage(
                ComplexStorage::F32(vec![(1.0, f32::NAN), (3.0, f32::INFINITY)]),
                vec![2, 1],
            )
            .expect("complex single"),
        );
        let rhs = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("double tensor"));
        let expected = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).expect("logical result");
        assert_eq!(
            try_complex_ordering_comparison(&lhs, &rhs, IntegerComparisonOp::Le)
                .expect("complex-single/double comparison"),
            Some(Value::LogicalArray(expected.clone()))
        );
        assert_eq!(
            try_complex_ordering_comparison(&rhs, &lhs, IntegerComparisonOp::Ge)
                .expect("double/complex-single comparison"),
            Some(Value::LogicalArray(expected))
        );
    }

    #[test]
    fn complex_ordering_accepts_logical_and_character_numeric_operands() {
        let logical =
            Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical"));
        assert_eq!(
            try_complex_ordering_comparison(
                &Value::Complex(0.5, 99.0),
                &logical,
                IntegerComparisonOp::Gt,
            )
            .expect("complex/logical comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical result")
            ))
        );

        let chars = Value::CharArray(
            runmat_value::CharArray::new(vec!['A', 'C', 'B', 'D'], 2, 2).expect("character array"),
        );
        assert_eq!(
            try_complex_ordering_comparison(
                &Value::Complex(66.0, -99.0),
                &chars,
                IntegerComparisonOp::Lt,
            )
            .expect("complex/character comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![0, 0, 1, 1], vec![2, 2]).expect("logical result")
            ))
        );
    }

    fn assert_resident_complex_ordering(provider: &dyn runmat_accelerate_api::AccelProvider) {
        let complex =
            ComplexTensor::new(vec![(1.0, 99.0), (3.0, -99.0)], vec![1, 2]).expect("complex");
        let complex_rhs =
            ComplexTensor::new(vec![(2.0, -7.0), (2.0, 7.0)], vec![1, 2]).expect("complex rhs");
        let real = Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("real");
        let complex_handle = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
        let complex_rhs_handle =
            gpu_helpers::upload_complex_tensor(provider, &complex_rhs).unwrap();
        let real_handle = gpu_helpers::upload_tensor(provider, &real).unwrap();
        for (builtin, expected, reverse_expected) in [
            ("lt", vec![1.0, 0.0], vec![0.0, 1.0]),
            ("le", vec![1.0, 0.0], vec![0.0, 1.0]),
            ("gt", vec![0.0, 1.0], vec![1.0, 0.0]),
            ("ge", vec![0.0, 1.0], vec![1.0, 0.0]),
        ] {
            for rhs in [&real_handle, &complex_rhs_handle] {
                let result = crate::call_builtin(
                    builtin,
                    &[
                        Value::GpuTensor(complex_handle.clone()),
                        Value::GpuTensor(rhs.clone()),
                    ],
                )
                .expect("resident complex ordering");
                assert!(
                    matches!(&result, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
                );
                let gathered = test_support::gather(result).expect("gather logical result");
                assert_eq!(gathered.shape, vec![1, 2]);
                assert_eq!(gathered.materialize_f64(), expected);
            }
            let result = crate::call_builtin(
                builtin,
                &[
                    Value::GpuTensor(real_handle.clone()),
                    Value::GpuTensor(complex_handle.clone()),
                ],
            )
            .expect("reverse resident complex ordering");
            let gathered = test_support::gather(result).expect("gather reverse logical result");
            assert_eq!(gathered.materialize_f64(), reverse_expected);
        }
        let _ = provider.free(&complex_handle);
        let _ = provider.free(&complex_rhs_handle);
        let _ = provider.free(&real_handle);
    }

    #[test]
    fn resident_complex_ordering_uses_provider_real_component_paths() {
        test_support::with_test_provider(assert_resident_complex_ordering);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_complex_ordering_uses_provider_real_component_paths() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("WGPU provider");
        assert_resident_complex_ordering(provider);
    }
}
