//! Whole-route resident zero-phase admission and publication.
use num_complex::Complex;
use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_storage,
    provider_iir_zero_phase_query, GpuTensorHandle, GpuTensorStorage, ProviderIirZeroPhaseFault,
    ProviderIirZeroPhaseIndexBuilder, ProviderIirZeroPhaseIndexPlan, ProviderIirZeroPhaseRequest,
    ProviderPrecision, ProviderPreparedIirCoefficients,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_value::{NumericDType, Value};

use super::{
    filtfilt_error, filtfilt_error_with_detail, lfilter_zi, resident_output,
    NormalizedCoefficients, OutputContext, SignalCooperation, BUILTIN_NAME,
    FILTFILT_ERROR_INTERNAL, FILTFILT_ERROR_INVALID_SIGNAL, FILTFILT_ERROR_PASS_FAILED,
    FILTFILT_ERROR_PROVIDER_OUTPUT,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

async fn real_coefficients(
    values: &[Complex<f64>],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut real = Vec::new();
    real.try_reserve_exact(values.len()).map_err(|source| {
        let mut builder =
            build_runtime_error("filtfilt: coefficient preparation allocation failed")
                .with_builtin(BUILTIN_NAME)
                .with_source(source);
        if let Some(identifier) = FILTFILT_ERROR_INTERNAL.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.with_gpu_gather_retry(GpuGatherRetry::Never).build()
    })?;
    for chunk in values.chunks(4096) {
        real.extend(chunk.iter().map(|value| value.re));
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(real)
}

async fn prepare_indices(
    shape: [usize; 2],
    coefficient_len: usize,
    axis: usize,
    reflection_len: usize,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<ProviderIirZeroPhaseIndexPlan> {
    let preparation_error = |source: anyhow::Error| {
        crate::runtime_error::with_provider_execution_source(
            filtfilt_error_with_detail(
                &FILTFILT_ERROR_INTERNAL,
                "resident reflection index preparation failed",
            ),
            source.into(),
        )
    };
    cooperation.completed(0).await?;
    let mut builder =
        ProviderIirZeroPhaseIndexBuilder::new(shape, coefficient_len, axis, reflection_len)
            .map_err(preparation_error)?;
    while let Some(work) = builder.advance_chunk() {
        cooperation.completed(work as u64).await?;
    }
    builder.finish().map_err(preparation_error)
}

/// Query the complete resident geometry before coefficient or signal
/// materialization. The later submission rechecks the same typed route after
/// numerical normalization and binds it to the current capability revision.
pub(super) fn preflight_route(
    input: &GpuTensorHandle,
    coefficient_len: usize,
    output_dtype: NumericDType,
) -> BuiltinResult<bool> {
    if handle_integer_type(input).is_some()
        || handle_is_logical(input)
        || handle_storage(input) != GpuTensorStorage::Real
    {
        return Ok(false);
    }
    let provider =
        resident_output::owner(input, || filtfilt_error(&FILTFILT_ERROR_INVALID_SIGNAL))?;
    let Some(precision) = handle_precision(input) else {
        return Ok(false);
    };
    if (precision == ProviderPrecision::F32) != (output_dtype == NumericDType::F32)
        || precision != provider.precision()
    {
        return Ok(false);
    }
    let reflection_len = coefficient_len
        .saturating_sub(1)
        .checked_mul(3)
        .ok_or_else(|| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    let axis = if super::vector_is_row(input.shape()) {
        1
    } else {
        0
    };
    let query = provider_iir_zero_phase_query(input, coefficient_len, axis, reflection_len)
        .map_err(|source| {
            crate::runtime_error::with_provider_execution_source(
                filtfilt_error_with_detail(
                    &FILTFILT_ERROR_INTERNAL,
                    "resident route geometry failed",
                ),
                source.into(),
            )
        })?;
    Ok(provider.query_feasibility(&query).is_supported())
}

pub(super) async fn try_execute(
    coeffs: &NormalizedCoefficients,
    input: &GpuTensorHandle,
    output_dtype: NumericDType,
    borrowed: &[&GpuTensorHandle],
    runtime: Option<&RuntimeContext>,
    cooperation: &mut SignalCooperation,
    preflight_supported: bool,
) -> BuiltinResult<Option<Value>> {
    if !preflight_supported {
        return Ok(None);
    }
    if handle_integer_type(input).is_some()
        || handle_is_logical(input)
        || handle_storage(input) != GpuTensorStorage::Real
        || !coeffs.is_real()
    {
        return Ok(None);
    }
    let provider =
        resident_output::owner(input, || filtfilt_error(&FILTFILT_ERROR_INVALID_SIGNAL))?;
    let Some(precision) = handle_precision(input) else {
        return Ok(None);
    };
    if (output_dtype == NumericDType::F32) != (precision == ProviderPrecision::F32)
        || provider.precision() != precision
    {
        return Ok(None);
    }
    let Some(reflection_len) = coeffs.state_len.checked_mul(3) else {
        return Err(filtfilt_error_with_detail(
            &FILTFILT_ERROR_INTERNAL,
            "reflection extent overflow",
        ));
    };
    let axis = if super::vector_is_row(input.shape()) {
        1
    } else {
        0
    };
    let query = provider_iir_zero_phase_query(input, coeffs.order, axis, reflection_len).map_err(
        |source| {
            crate::runtime_error::with_provider_execution_source(
                filtfilt_error_with_detail(
                    &FILTFILT_ERROR_INTERNAL,
                    "resident route geometry failed",
                ),
                source.into(),
            )
        },
    )?;
    if !provider.query_feasibility(&query).is_supported() {
        return Ok(None);
    }
    let capability_revision = provider.capability_snapshot().revision;
    cooperation.completed(0).await?;
    let zi = lfilter_zi(coeffs, cooperation).await?;
    for chunk in zi.chunks(4096) {
        if chunk.iter().any(|value| value.im != 0.0) {
            return Ok(None);
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    let numerator = real_coefficients(&coeffs.b, cooperation).await?;
    let denominator = real_coefficients(&coeffs.a, cooperation).await?;
    let zi = real_coefficients(&zi, cooperation).await?;
    let shape: [usize; 2] = input.shape().try_into().map_err(|_| {
        filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, "resident vector rank mismatch")
    })?;
    let indices = prepare_indices(shape, coeffs.order, axis, reflection_len, cooperation).await?;
    let mut protected = Vec::new();
    protected
        .try_reserve_exact(borrowed.len().saturating_add(1))
        .map_err(|source| {
            let mut error =
                build_runtime_error("filtfilt: borrowed-input admission allocation failed")
                    .with_builtin(BUILTIN_NAME)
                    .with_source(source)
                    .build();
            error.gpu_gather_retry = GpuGatherRetry::Never;
            error
        })?;
    protected.push(input);
    protected.extend_from_slice(borrowed);
    let prepared = ProviderPreparedIirCoefficients::upload(
        provider,
        &numerator,
        &denominator,
        precision,
        &protected,
    )
    .map_err(|source| {
        crate::runtime_error::with_provider_execution_source(
            filtfilt_error_with_detail(
                &FILTFILT_ERROR_PASS_FAILED,
                "resident coefficient preparation failed",
            ),
            runmat_accelerate_api::ProviderExecutionError::from_source(source),
        )
    })?;
    let fallback_cancellation = CancellationSource::new();
    let cancellation = runtime
        .map(RuntimeContext::cancellation)
        .unwrap_or_else(|| fallback_cancellation.token());
    let request = ProviderIirZeroPhaseRequest {
        input,
        coefficients: &prepared,
        indices: &indices,
        borrowed,
        initial_state: &zi,
        axis,
        reflection_len,
        capability_revision,
        cancellation,
    };
    let output = provider.iir_zero_phase(&request).await.map_err(|source| {
        let mut fault: Option<ProviderIirZeroPhaseFault> = None;
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&source);
        while let Some(error) = cause {
            if let Some(found) = error.downcast_ref::<ProviderIirZeroPhaseFault>() {
                fault = Some(*found);
                break;
            }
            cause = error.source();
        }
        let outer = match fault {
            Some(ProviderIirZeroPhaseFault::NonFiniteInput) => filtfilt_error_with_detail(
                &FILTFILT_ERROR_INVALID_SIGNAL,
                "signal values must be finite",
            ),
            Some(ProviderIirZeroPhaseFault::UnrepresentableReflection) => {
                filtfilt_error_with_detail(
                    &FILTFILT_ERROR_INVALID_SIGNAL,
                    "endpoint reflection is outside the floating-point range",
                )
            }
            None => filtfilt_error_with_detail(
                &FILTFILT_ERROR_PASS_FAILED,
                "resident zero-phase pass failed",
            ),
        };
        crate::runtime_error::with_provider_execution_source(outer, source.after_effect())
    })?;
    let context = OutputContext {
        input,
        provider,
        caller_inputs: borrowed,
    };
    let contract =
        resident_output::floating_contract(input, input.shape().to_vec(), GpuTensorStorage::Real);
    let output = context.admit(
        output,
        &contract,
        &[prepared.numerator(), prepared.denominator()],
        || filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT),
    )?;
    cooperation.completed(0).await?;
    Ok(Some(Value::GpuTensor(output)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use std::cell::Cell;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};

    struct PendingTurn(Rc<Cell<usize>>);
    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn dropped_pending_index_preparation_retains_no_unfinished_work() {
        let cancelled = CancellationSource::new();
        let dropped = Rc::new(Cell::new(0));
        let dropped_for_turn = dropped.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            cancelled.token(),
            move || {
                let marker = PendingTurn(dropped_for_turn.clone());
                Box::pin(async move {
                    std::future::pending::<()>().await;
                    drop(marker);
                    Ok(())
                })
            },
        );
        let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
        let mut future = Box::pin(prepare_indices([5000, 1], 2, 0, 3, &mut cooperation));
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(dropped.get(), 0);
        drop(future);
        assert_eq!(dropped.get(), 1);
    }
}
