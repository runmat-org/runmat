//! Host IIR numerical engine and result publication.
use super::*;
use runmat_builtins::FILTER_ERROR_DENOMINATOR_A1_ZERO;
use runmat_value::ComplexTensor;

const HOST_WORK_CHUNK: usize = 4096;

fn resource_error(source: std::collections::TryReserveError) -> RuntimeError {
    let mut builder = build_runtime_error("filter: host storage reservation failed")
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = FILTER_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

async fn checked_copy<T: Copy>(
    source: &[T],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(source.len())
        .map_err(resource_error)?;
    for chunk in source.chunks(HOST_WORK_CHUNK) {
        result.extend_from_slice(chunk);
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(result)
}

async fn checked_map<T: Copy, U>(
    source: &[T],
    cooperation: &mut SignalCooperation,
    mut map: impl FnMut(T) -> U,
) -> BuiltinResult<Vec<U>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(source.len())
        .map_err(resource_error)?;
    for chunk in source.chunks(HOST_WORK_CHUNK) {
        result.extend(chunk.iter().copied().map(&mut map));
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(result)
}

async fn checked_zeros<T: Copy + Default>(
    len: usize,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<T>> {
    let mut result = Vec::new();
    result.try_reserve_exact(len).map_err(resource_error)?;
    while result.len() < len {
        let next = result.len().saturating_add(HOST_WORK_CHUNK).min(len);
        let added = next - result.len();
        result.resize(next, T::default());
        cooperation.completed(added as u64).await?;
    }
    Ok(result)
}

async fn extend_zeros(
    values: &mut Vec<Complex<f64>>,
    target: usize,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<()> {
    if target <= values.len() {
        return Ok(());
    }
    values
        .try_reserve_exact(target - values.len())
        .map_err(resource_error)?;
    while values.len() < target {
        let next = values.len().saturating_add(HOST_WORK_CHUNK).min(target);
        let added = next - values.len();
        values.resize(next, Complex::new(0.0, 0.0));
        cooperation.completed(added as u64).await?;
    }
    Ok(())
}

pub(super) async fn filter_host(
    args: &FilterArgs,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<FilterEvaluation> {
    cooperation.completed(0).await?;
    let signal_data = args.signal.host_data()?;
    let mut b_norm = checked_copy(&args.coeffs_b.data, cooperation).await?;
    let mut a_norm = checked_copy(&args.coeffs_a.data, cooperation).await?;

    let a0 = a_norm[0];
    if a0 == Complex::new(0.0, 0.0) {
        return Err(filter_error(&FILTER_ERROR_DENOMINATOR_A1_ZERO));
    }

    for chunk in b_norm.chunks_mut(HOST_WORK_CHUNK) {
        for coeff in chunk.iter_mut() {
            *coeff /= a0;
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    for (chunk_idx, chunk) in a_norm.chunks_mut(HOST_WORK_CHUNK).enumerate() {
        for (index, coeff) in chunk.iter_mut().enumerate() {
            let idx = chunk_idx * HOST_WORK_CHUNK + index;
            if idx == 0 {
                *coeff = Complex::new(1.0, 0.0);
            } else if idx < args.coeffs_a.len {
                *coeff /= a0;
            } else {
                *coeff = Complex::new(0.0, 0.0);
            }
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    extend_zeros(&mut b_norm, args.order, cooperation).await?;
    extend_zeros(&mut a_norm, args.order, cooperation).await?;

    let expected_states = args
        .state_len
        .checked_mul(args.channel_count)
        .ok_or_else(|| runtime_error_for("filter: state extent overflow"))?;

    let mut states = if args.state_len == 0 {
        Vec::<Complex<f64>>::new()
    } else if args.initial.provided {
        if args.initial.column_major.len() != expected_states {
            return Err(filter_error_with_detail(
                &FILTER_ERROR_INVALID_INITIAL_STATE,
                format!(
                    "initial conditions have {} elements but {} were expected",
                    args.initial.column_major.len(),
                    expected_states
                ),
            ));
        }
        checked_copy(&args.initial.column_major, cooperation).await?
    } else {
        let mut states = Vec::new();
        extend_zeros(&mut states, expected_states, cooperation).await?;
        states
    };

    let mut output = Vec::new();
    extend_zeros(&mut output, signal_data.len(), cooperation).await?;

    if args.state_len == 0 {
        let gain = b_norm[0];
        for (destination, source) in output
            .chunks_mut(HOST_WORK_CHUNK)
            .zip(signal_data.chunks(HOST_WORK_CHUNK))
        {
            for (dst, &src) in destination.iter_mut().zip(source) {
                *dst = gain * src;
            }
            cooperation.completed(destination.len() as u64).await?;
        }
    } else if args.channel_count == 0 || args.shape_ext[args.dim_idx] == 0 {
        // nothing to do, states already contain initial values.
    } else {
        let dim_len = args.shape_ext[args.dim_idx];
        let per_sample_work = u64::try_from(args.order)
            .map_err(|_| runtime_error_for("filter: work extent overflow"))?;
        let mut pending_work = 0u64;
        for t in 0..args.trailing {
            let base = t
                .checked_mul(dim_len)
                .and_then(|v| v.checked_mul(args.leading))
                .ok_or_else(|| runtime_error_for("filter: index overflow during evaluation"))?;
            for l in 0..args.leading {
                let channel_idx = t
                    .checked_mul(args.leading)
                    .and_then(|v| v.checked_add(l))
                    .ok_or_else(|| runtime_error_for("filter: index overflow during evaluation"))?;
                if channel_idx >= args.channel_count {
                    continue;
                }
                let state_base = channel_idx
                    .checked_mul(args.state_len)
                    .ok_or_else(|| runtime_error_for("filter: state index overflow"))?;
                for step in 0..dim_len {
                    let idx = base
                        .checked_add(l)
                        .and_then(|v| v.checked_add(step.saturating_mul(args.leading)))
                        .ok_or_else(|| runtime_error_for("filter: signal index overflow"))?;
                    if idx >= signal_data.len() {
                        break;
                    }
                    let x_n = signal_data[idx];
                    let y = b_norm[0] * x_n + states[state_base];
                    output[idx] = y;
                    for i in 1..args.order {
                        let next_state = if i < args.state_len {
                            states[state_base + i]
                        } else {
                            Complex::new(0.0, 0.0)
                        };
                        let new_state = b_norm[i] * x_n + next_state - a_norm[i] * y;
                        states[state_base + i - 1] = new_state;
                    }
                    pending_work = pending_work
                        .checked_add(per_sample_work)
                        .ok_or_else(|| runtime_error_for("filter: work extent overflow"))?;
                    if pending_work >= HOST_WORK_CHUNK as u64 {
                        cooperation.completed(pending_work).await?;
                        pending_work = 0;
                    }
                }
            }
        }
        cooperation.completed(pending_work).await?;
    }

    let final_states_column = if args.state_len == 0 {
        Vec::<Complex<f64>>::new()
    } else {
        states
    };

    let output_value = if args.is_complex {
        if output.len() == 1 {
            if args.signal.output_single {
                let data =
                    checked_map(&output, cooperation, |c| (c.re as f32, c.im as f32)).await?;
                let tensor = ComplexTensor::from_f32(data, vec![1, 1])
                    .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
                Value::ComplexTensor(tensor)
            } else {
                Value::Complex(output[0].re, output[0].im)
            }
        } else {
            let tensor = if args.signal.output_single {
                ComplexTensor::from_f32(
                    checked_map(&output, cooperation, |c| (c.re as f32, c.im as f32)).await?,
                    args.signal.shape.clone(),
                )
            } else {
                ComplexTensor::new(
                    checked_map(&output, cooperation, |c| (c.re, c.im)).await?,
                    args.signal.shape.clone(),
                )
            }
            .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
            Value::ComplexTensor(tensor)
        }
    } else {
        let tensor = if args.signal.output_single {
            Tensor::from_f32(
                checked_map(&output, cooperation, |c| c.re as f32).await?,
                args.signal.shape.clone(),
            )
        } else {
            Tensor::new(
                checked_map(&output, cooperation, |c| c.re).await?,
                args.signal.shape.clone(),
            )
        }
        .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
        tensor::tensor_into_value(tensor)
    };

    let final_state_value = if args.state_len == 0 {
        let count = args
            .state_shape
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| runtime_error_for("filter: state extent overflow"))?;
        let tensor = if args.signal.output_single {
            Tensor::from_f32(
                checked_zeros::<f32>(count, cooperation).await?,
                args.state_shape.clone(),
            )
        } else {
            Tensor::new(
                checked_zeros::<f64>(count, cooperation).await?,
                args.state_shape.clone(),
            )
        }
        .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
        tensor::tensor_into_value(tensor)
    } else if args.is_complex || args.initial.is_complex {
        if final_states_column.len() == 1 && !args.signal.output_single {
            Value::Complex(final_states_column[0].re, final_states_column[0].im)
        } else {
            let tensor = if args.signal.output_single {
                ComplexTensor::from_f32(
                    checked_map(&final_states_column, cooperation, |c| {
                        (c.re as f32, c.im as f32)
                    })
                    .await?,
                    args.state_shape.clone(),
                )
            } else {
                ComplexTensor::new(
                    checked_map(&final_states_column, cooperation, |c| (c.re, c.im)).await?,
                    args.state_shape.clone(),
                )
            }
            .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
            Value::ComplexTensor(tensor)
        }
    } else {
        let tensor = if args.signal.output_single {
            Tensor::from_f32(
                checked_map(&final_states_column, cooperation, |c| c.re as f32).await?,
                args.state_shape.clone(),
            )
        } else {
            Tensor::new(
                checked_map(&final_states_column, cooperation, |c| c.re).await?,
                args.state_shape.clone(),
            )
        }
        .map_err(|e| runtime_error_for(format!("filter: {e}")))?;
        tensor::tensor_into_value(tensor)
    };

    Ok(FilterEvaluation {
        output: output_value,
        final_state: final_state_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use runmat_execution::cancellation::CancellationSource;
    use std::cell::Cell;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};

    fn long_host_args() -> FilterArgs {
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        futures::executor::block_on(FilterArgs::parse(
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0; 8192], vec![8192, 1]).unwrap()),
            crate::call::FunctionArgumentView::from_slice(&[]),
            &mut cooperation,
        ))
        .unwrap()
    }

    #[test]
    fn host_filter_keeps_execution_service_failure_and_publishes_nothing() {
        let args = long_host_args();
        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(10_000).unwrap()),
            source.token(),
            || Box::pin(async { Err(ExecutionServiceError::Infrastructure("turn failed".into())) }),
        );
        let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
        let Err(error) = futures::executor::block_on(filter_host(&args, &mut cooperation)) else {
            panic!("failed service turn must stop the host solver before publication")
        };
        assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(std::error::Error::source(&error)
            .is_some_and(|cause| cause.is::<ExecutionServiceError>()));
    }

    struct PendingTurn(Rc<Cell<usize>>);

    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn dropping_pending_host_filter_drops_its_turn_without_publishing() {
        let args = long_host_args();
        let source = CancellationSource::new();
        let dropped = Rc::new(Cell::new(0));
        let observed = dropped.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(10_000).unwrap()),
            source.token(),
            move || {
                let marker = PendingTurn(observed.clone());
                Box::pin(async move {
                    std::future::pending::<()>().await;
                    drop(marker);
                    Ok(())
                })
            },
        );
        let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
        let mut future = Box::pin(filter_host(&args, &mut cooperation));
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(dropped.get(), 0);
        drop(future);
        assert_eq!(dropped.get(), 1);
    }
}
