//! Checked, cooperative host materialization for admitted IIR operands.

use super::*;
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::{ComplexTensor, LogicalArray};

pub(super) const PREPARATION_CHUNK: usize = 4096;

pub(super) fn provider_resource_error(source: std::collections::TryReserveError) -> RuntimeError {
    let mut error = filter_error(&FILTER_ERROR_INTERNAL);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    crate::runtime_error::with_provider_execution_source(
        error,
        runmat_accelerate_api::ProviderExecutionError::from_source(source),
    )
}

fn rounded_source(value: f64, precision: ProviderPrecision) -> Option<f64> {
    if !value.is_finite() {
        return None;
    }
    let rounded = match precision {
        ProviderPrecision::F32 => f64::from(value as f32),
        ProviderPrecision::F64 => value,
    };
    rounded.is_finite().then_some(rounded)
}

/// Match the old source-upload, physical-precision download, normalize and
/// re-upload path without its redundant provider traffic.
pub(super) async fn prepared_real_coefficients(
    source_b: &[Complex<f64>],
    source_a: &[Complex<f64>],
    order: usize,
    precision: ProviderPrecision,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<(Vec<f64>, Vec<f64>)>> {
    let Some(leading) = source_a.first() else {
        return Ok(None);
    };
    if leading.im != 0.0 {
        return Ok(None);
    }
    let Some(a0) = rounded_source(leading.re, precision) else {
        return Ok(None);
    };
    if a0 == 0.0 {
        return Ok(None);
    }
    let mut numerator = Vec::new();
    let mut denominator = Vec::new();
    numerator
        .try_reserve_exact(order)
        .map_err(provider_resource_error)?;
    denominator
        .try_reserve_exact(order)
        .map_err(provider_resource_error)?;
    for start in (0..order).step_by(PREPARATION_CHUNK) {
        let end = start.saturating_add(PREPARATION_CHUNK).min(order);
        for index in start..end {
            let b = source_b
                .get(index)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            let a = source_a
                .get(index)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            if b.im != 0.0 || a.im != 0.0 {
                return Ok(None);
            }
            let (Some(b), Some(a)) = (
                rounded_source(b.re, precision),
                rounded_source(a.re, precision),
            ) else {
                return Ok(None);
            };
            let b = b / a0;
            let a = if index == 0 { 1.0 } else { a / a0 };
            if !b.is_finite()
                || !a.is_finite()
                || (precision == ProviderPrecision::F32
                    && (!(b as f32).is_finite() || !(a as f32).is_finite()))
            {
                return Ok(None);
            }
            numerator.push(b);
            denominator.push(a);
        }
        cooperation.completed((end - start) as u64).await?;
    }
    Ok(Some((numerator, denominator)))
}

pub(super) async fn to_real_vec(
    data: &[Complex<f64>],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let mut out = Vec::new();
    out.try_reserve_exact(data.len())
        .map_err(provider_resource_error)?;
    for chunk in data.chunks(PREPARATION_CHUNK) {
        for value in chunk {
            if value.im != 0.0 {
                return Ok(None);
            }
            out.push(value.re);
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(Some(out))
}

fn reservation_error(source: std::collections::TryReserveError) -> RuntimeError {
    let mut builder = build_runtime_error("filter: operand storage reservation failed")
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = FILTER_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

async fn materialize(
    len: usize,
    cooperation: &mut SignalCooperation,
    mut at: impl FnMut(usize) -> Option<Complex<f64>>,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len).map_err(reservation_error)?;
    while data.len() < len {
        let start = data.len();
        let end = start.saturating_add(PREPARATION_CHUNK).min(len);
        for index in start..end {
            data.push(at(index).ok_or_else(|| filter_error(&FILTER_ERROR_INTERNAL))?);
        }
        cooperation.completed((end - start) as u64).await?;
    }
    Ok(data)
}

pub(super) async fn real_tensor(
    tensor: &Tensor,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    materialize(tensor.len(), cooperation, |index| {
        tensor
            .numeric_value_at(index)
            .map(|value| Complex::new(value.materialize_f64(), 0.0))
    })
    .await
}

pub(super) async fn complex_tensor(
    tensor: &ComplexTensor,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    materialize(tensor.len(), cooperation, |index| {
        tensor.numeric_value_at(index).map(|(real, imaginary)| {
            Complex::new(real.materialize_f64(), imaginary.materialize_f64())
        })
    })
    .await
}

pub(super) async fn logical_array(
    logical: &LogicalArray,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    materialize(logical.data.len(), cooperation, |index| {
        logical
            .data
            .get(index)
            .map(|value| Complex::new(if *value == 0 { 0.0 } else { 1.0 }, 0.0))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_value::LogicalArray;
    use std::num::NonZeroU64;

    #[test]
    fn logical_materialization_preserves_shape_order_and_boolean_values() {
        let logical = LogicalArray::new(vec![2, 0, 1, 0], vec![2, 2]).unwrap();
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        let values = block_on(logical_array(&logical, &mut cooperation)).unwrap();
        assert_eq!(
            values,
            [2.0, 0.0, 1.0, 0.0]
                .map(|value| Complex::new(if value == 0.0 { 0.0 } else { 1.0 }, 0.0))
        );
    }

    #[test]
    fn large_operand_preparation_retains_service_cause_and_never_retries_gather() {
        let tensor = Tensor::new(
            vec![1.0; PREPARATION_CHUNK + 1],
            vec![PREPARATION_CHUNK + 1, 1],
        )
        .unwrap();
        let cancellation = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            cancellation.token(),
            || Box::pin(async { Err(ExecutionServiceError::Infrastructure("turn failed".into())) }),
        );
        let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
        let Err(error) = block_on(real_tensor(&tensor, &mut cooperation)) else {
            panic!("failed cooperation turn must stop preparation")
        };
        assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(std::error::Error::source(&error)
            .is_some_and(|cause| cause.is::<ExecutionServiceError>()));
    }
}
