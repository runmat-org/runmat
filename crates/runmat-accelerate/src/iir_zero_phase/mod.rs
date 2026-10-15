//! Provider-owned reflected two-pass IIR route over resident primitives.
mod stages;
#[cfg(test)]
mod tests;

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_storage, provider_for_handle,
    same_provider_instance, upload_numeric_protecting, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderExecutionError,
    ProviderExecutionResult, ProviderIirCoefficientSource, ProviderIirFilterOptions,
    ProviderIirZeroPhaseFault, ProviderIirZeroPhaseIndexPlan, ProviderIirZeroPhaseRequest,
    ProviderPrecision,
};
use runmat_execution::CancellationReason;
use stages::{StageClass, Stages};

#[derive(Debug, thiserror::Error)]
#[error("iir_zero_phase: invocation cancelled ({reason:?})")]
struct RouteCancelled {
    reason: Option<CancellationReason>,
}

fn check_cancel(request: &ProviderIirZeroPhaseRequest<'_>) -> ProviderExecutionResult<()> {
    if request.cancellation.is_cancelled() {
        Err(ProviderExecutionError::from_source(RouteCancelled {
            reason: request.cancellation.reason(),
        }))
    } else {
        Ok(())
    }
}

fn check_owner(provider: &dyn AccelProvider, handle: &GpuTensorHandle) -> anyhow::Result<()> {
    handle.allocation_lease().validate_live()?;
    ensure!(
        handle.device_id() == provider.device_id()
            && provider_for_handle(handle)
                .is_some_and(|owner| same_provider_instance(owner, provider)),
        "iir_zero_phase: resident operand allocation owner mismatch"
    );
    ensure!(
        handle_precision(handle) == Some(provider.precision())
            && handle_storage(handle) == GpuTensorStorage::Real
            && handle_integer_type(handle).is_none()
            && !handle_is_logical(handle)
            && handle.view().layout() == runmat_accelerate_api::GpuTensorLayout::Contiguous,
        "iir_zero_phase: real floating operand class mismatch"
    );
    Ok(())
}

fn prepared_state(
    request: &ProviderIirZeroPhaseRequest<'_>,
    precision: ProviderPrecision,
) -> ProviderExecutionResult<Option<Vec<f32>>> {
    check_cancel(request)?;
    let mut single = if precision == ProviderPrecision::F32 {
        let mut values = Vec::new();
        values
            .try_reserve_exact(request.initial_state.len())
            .map_err(ProviderExecutionError::from_source)?;
        Some(values)
    } else {
        None
    };
    for chunk in request.initial_state.chunks(4096) {
        for &value in chunk {
            if !value.is_finite() {
                return Err(ProviderExecutionError::from(anyhow!(
                    "iir_zero_phase: nonfinite initial state"
                )));
            }
            if let Some(single) = single.as_mut() {
                let narrowed = value as f32;
                if !narrowed.is_finite() {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "iir_zero_phase: initial state exceeds single range"
                    )));
                }
                single.push(narrowed);
            }
        }
        check_cancel(request)?;
    }
    Ok(single)
}

pub(crate) async fn execute(
    provider: &dyn AccelProvider,
    request: &ProviderIirZeroPhaseRequest<'_>,
) -> ProviderExecutionResult<GpuTensorHandle> {
    check_cancel(request)?;
    let query = request
        .feasibility_query()
        .map_err(ProviderExecutionError::from)?;
    if request.capability_revision != provider.capability_snapshot().revision
        || !provider.query_feasibility(&query).is_supported()
    {
        return Err(ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation(
                "iir_zero_phase: whole-route feasibility changed before execution",
            ),
        ));
    }
    for handle in [
        request.input,
        request.coefficients.numerator(),
        request.coefficients.denominator(),
    ] {
        check_owner(provider, handle).map_err(ProviderExecutionError::from)?;
    }
    ensure_distinct(request).map_err(ProviderExecutionError::from)?;
    let state_len = request
        .coefficients
        .numerator()
        .shape()
        .first()
        .copied()
        .ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("iir_zero_phase: empty coefficient geometry"))
        })?
        .saturating_sub(1);
    if request.initial_state.len() != state_len {
        return Err(ProviderExecutionError::from(anyhow!(
            "iir_zero_phase: initial state length mismatch"
        )));
    }
    let indices = request.indices;
    let single_state = prepared_state(request, provider.precision())?;
    let mut stages = Stages::new(
        provider,
        [
            request.input,
            request.coefficients.numerator(),
            request.coefficients.denominator(),
        ],
        request.borrowed,
    )?;
    check_cancel(request)?;
    // From the first primitive call onward, even an Unsupported cause is an
    // operational result: the provider may already have staged a command.
    let result = execute_staged(
        provider,
        request,
        indices,
        single_state.as_deref(),
        &mut stages,
    )
    .await;
    stages
        .finish(result)
        .map_err(ProviderExecutionError::after_effect)
}

fn ensure_distinct(request: &ProviderIirZeroPhaseRequest<'_>) -> anyhow::Result<()> {
    let b = request.coefficients.numerator();
    let a = request.coefficients.denominator();
    ensure!(
        !request.input.shares_allocation_with(b)
            && !request.input.shares_allocation_with(a)
            && !b.shares_allocation_with(a)
            && request.borrowed.iter().all(|borrowed| {
                !b.shares_allocation_with(borrowed) && !a.shares_allocation_with(borrowed)
            }),
        "iir_zero_phase: coefficients and signal must have distinct allocations"
    );
    Ok(())
}

async fn execute_staged(
    provider: &dyn AccelProvider,
    request: &ProviderIirZeroPhaseRequest<'_>,
    indices: &ProviderIirZeroPhaseIndexPlan,
    single_state: Option<&[f32]>,
    stages: &mut Stages<'_>,
) -> ProviderExecutionResult<usize> {
    let shape = indices.extended_shape();
    verify_finite(
        provider,
        stages,
        request.input,
        request.input.shape(),
        ProviderIirZeroPhaseFault::NonFiniteInput,
    )
    .await?;
    check_cancel(request)?;
    let edge = stages.admit(
        provider
            .gather_linear(request.input, indices.edge(), shape)
            .map_err(ProviderExecutionError::from)?,
        shape,
        StageClass::Real,
    )?;
    check_cancel(request)?;
    let neighbor = stages.admit(
        provider
            .gather_linear(request.input, indices.neighbor(), shape)
            .map_err(ProviderExecutionError::from)?,
        shape,
        StageClass::Real,
    )?;
    let difference = provider
        .elem_sub(stages.get(edge), stages.get(neighbor))
        .await?;
    let difference = stages.admit(difference, shape, StageClass::Real)?;
    let reflected = provider
        .elem_add(stages.get(edge), stages.get(difference))
        .await?;
    let reflected = stages.admit(reflected, shape, StageClass::Real)?;
    check_cancel(request)?;
    let reflected_handle = stages.get(reflected).clone();
    verify_finite(
        provider,
        stages,
        &reflected_handle,
        shape,
        ProviderIirZeroPhaseFault::UnrepresentableReflection,
    )
    .await?;
    check_cancel(request)?;

    let state = if request.initial_state.is_empty() {
        None
    } else {
        let data = match single_state {
            Some(single) => HostNumericDataView::F32(single),
            None => HostNumericDataView::F64(request.initial_state),
        };
        let state = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data,
                shape: indices.state_shape(),
                storage: GpuTensorStorage::Real,
            },
            stages.protected(),
        )
        .map_err(ProviderExecutionError::from_source)?;
        Some(stages.admit(state, indices.state_shape(), StageClass::Real)?)
    };
    let first_zi = scale_endpoint(
        provider,
        request,
        stages,
        reflected,
        state,
        indices.first_endpoint(),
        indices.state_shape(),
    )
    .await?;
    let first = pass(
        provider,
        request,
        stages,
        reflected,
        first_zi,
        shape,
        indices.state_shape(),
    )
    .await?;
    check_cancel(request)?;
    let second_zi = scale_endpoint(
        provider,
        request,
        stages,
        first,
        state,
        indices.second_endpoint(),
        indices.state_shape(),
    )
    .await?;
    let flipped = provider.flip(stages.get(first), &[request.axis])?;
    let reversed = stages.admit_flip(flipped, first, request.axis, shape)?;
    let second = pass(
        provider,
        request,
        stages,
        reversed,
        second_zi,
        shape,
        indices.state_shape(),
    )
    .await?;
    check_cancel(request)?;
    let flipped = provider.flip(stages.get(second), &[request.axis])?;
    let restored = stages.admit_flip(flipped, second, request.axis, shape)?;
    let output = provider
        .gather_linear(stages.get(restored), indices.trim(), request.input.shape())
        .map_err(ProviderExecutionError::from)?;
    let output = stages.admit(output, request.input.shape(), StageClass::Real)?;
    check_cancel(request)?;
    Ok(output)
}

async fn verify_finite(
    provider: &dyn AccelProvider,
    stages: &mut Stages<'_>,
    signal: &GpuTensorHandle,
    shape: &[usize],
    fault: ProviderIirZeroPhaseFault,
) -> ProviderExecutionResult<()> {
    let finite = stages.admit(
        provider
            .logical_isfinite(signal)
            .map_err(ProviderExecutionError::from)?,
        shape,
        StageClass::Logical,
    )?;
    let all_finite = provider
        .reduce_all(stages.get(finite), false)
        .await
        .map_err(ProviderExecutionError::from)?;
    let all_finite = stages.admit(all_finite, &[1, 1], StageClass::Predicate)?;
    let predicate = provider
        .download(stages.get(all_finite))
        .await
        .map_err(ProviderExecutionError::from)?;
    if predicate.shape != [1, 1] {
        return Err(ProviderExecutionError::from(anyhow!(
            "iir_zero_phase: malformed finite predicate output"
        )));
    }
    match predicate.data.as_slice() {
        [1.0] => Ok(()),
        [0.0] => Err(ProviderExecutionError::from_source(fault)),
        _ => Err(ProviderExecutionError::from(anyhow!(
            "iir_zero_phase: malformed finite predicate value"
        ))),
    }
}

async fn scale_endpoint(
    provider: &dyn AccelProvider,
    request: &ProviderIirZeroPhaseRequest<'_>,
    stages: &mut Stages<'_>,
    signal: usize,
    state: Option<usize>,
    indices: &[u32],
    shape: &[usize; 2],
) -> ProviderExecutionResult<Option<usize>> {
    let Some(state) = state else { return Ok(None) };
    let endpoint = stages.admit(
        provider
            .gather_linear(stages.get(signal), indices, shape)
            .map_err(ProviderExecutionError::from)?,
        shape,
        StageClass::Real,
    )?;
    let scaled = provider
        .elem_mul(stages.get(state), stages.get(endpoint))
        .await?;
    let scaled = stages.admit(scaled, shape, StageClass::Real)?;
    check_cancel(request)?;
    Ok(Some(scaled))
}

async fn pass(
    provider: &dyn AccelProvider,
    request: &ProviderIirZeroPhaseRequest<'_>,
    stages: &mut Stages<'_>,
    signal: usize,
    zi: Option<usize>,
    signal_shape: &[usize; 2],
    state_shape: &[usize; 2],
) -> ProviderExecutionResult<usize> {
    let result = provider
        .iir_filter(
            ProviderIirCoefficientSource::Prepared(request.coefficients),
            stages.get(signal),
            ProviderIirFilterOptions {
                dim: request.axis,
                zi: zi.map(|index| stages.get(index).clone()),
            },
        )
        .await
        .map_err(ProviderExecutionError::from)?;
    let output = stages.admit_filter_result(
        result,
        signal_shape,
        state_shape,
        !request.initial_state.is_empty(),
    )?;
    check_cancel(request)?;
    Ok(output)
}
