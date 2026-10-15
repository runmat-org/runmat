//! Solver-owned binary64 traversal of authoritative host storage.

use runmat_value::{IntegerStorage, Tensor};

use super::ScalarCooperation;
use crate::BuiltinResult;

// Charge one unit for each source value visited; this is not an elapsed-time estimate.
const VALUES_PER_CHECKPOINT: usize = 4096;

pub(crate) async fn tensor_values(
    tensor: &Tensor,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    if let Some(values) = tensor.as_f64_slice() {
        return convert(values, |value| value, cooperation).await;
    }
    if let Some(values) = tensor.as_f32_slice() {
        return convert(values, f64::from, cooperation).await;
    }
    match tensor
        .integer_storage()
        .expect("real numeric tensor storage")
    {
        IntegerStorage::I8(values) => convert(values, f64::from, cooperation).await,
        IntegerStorage::I16(values) => convert(values, f64::from, cooperation).await,
        IntegerStorage::I32(values) => convert(values, f64::from, cooperation).await,
        IntegerStorage::I64(values) => convert(values, |value| value as f64, cooperation).await,
        IntegerStorage::U8(values) => convert(values, f64::from, cooperation).await,
        IntegerStorage::U16(values) => convert(values, f64::from, cooperation).await,
        IntegerStorage::U32(values) => convert(values, f64::from, cooperation).await,
        IntegerStorage::U64(values) => convert(values, |value| value as f64, cooperation).await,
    }
}

pub(crate) async fn logical_values(
    values: &[u8],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    convert(
        values,
        |value| if value == 0 { 0.0 } else { 1.0 },
        cooperation,
    )
    .await
}

async fn convert<T: Copy>(
    values: &[T],
    convert: impl Fn(T) -> f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    cooperation.completed(0).await?;
    let mut output = Vec::with_capacity(values.len());
    for chunk in values.chunks(VALUES_PER_CHECKPOINT) {
        output.extend(chunk.iter().copied().map(&convert));
        runmat_value::record_host_copy(
            runmat_value::HostCopyReason::OwnedMaterialization,
            chunk.len() * std::mem::size_of::<f64>(),
        );
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(output)
}

pub(crate) async fn first_nonfinite(
    values: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<f64>> {
    cooperation.completed(0).await?;
    for chunk in values.chunks(VALUES_PER_CHECKPOINT) {
        if let Some(value) = chunk.iter().find(|value| !value.is_finite()) {
            return Ok(Some(*value));
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use futures::{executor::block_on, task::noop_waker};
    use runmat_builtins::OptimSolver;
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::{cell::Cell, future::Future, num::NonZeroU64, rc::Rc, task::Context};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn borrowed_native_conversion_preserves_order_and_existing_binary64_rounding() {
        for storage in [
            runmat_value::NumericStorage::F32(vec![-0.0, 1.5]),
            runmat_value::NumericStorage::I64(vec![-7, 9]),
            runmat_value::NumericStorage::U64(vec![(1_u64 << 53) + 1, 1_u64 << 54]),
        ] {
            let expected = storage.materialize_f64();
            let tensor = Tensor::from_numeric_storage(storage, vec![1, 1, 2]).unwrap();
            let mut cooperation = ScalarCooperation::new(OptimSolver::Lsqnonlin, None);
            let values = block_on(tensor_values(&tensor, &mut cooperation)).unwrap();
            assert_eq!(
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(tensor.shape(), &[1, 1, 2]);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn chunked_conversion_cancellation_and_drop_do_not_replay_or_mutate_source() {
        let tensor = Tensor::new(
            vec![3.0; VALUES_PER_CHECKPOINT + 1],
            vec![VALUES_PER_CHECKPOINT + 1, 1],
        )
        .unwrap();
        for cancel in [false, true] {
            let source = Rc::new(CancellationSource::new());
            let turns = Rc::new(Cell::new(0));
            let count = turns.clone();
            let cancellation = source.clone();
            let budget = CpuCooperationBudget::with_test_turn(
                CpuCooperationPolicy::WorkQuantum(
                    NonZeroU64::new(VALUES_PER_CHECKPOINT as u64).unwrap(),
                ),
                source.token(),
                move || {
                    count.set(count.get() + 1);
                    if cancel {
                        cancellation.cancel(CancellationReason::User);
                    }
                    Box::pin(async { futures::future::pending().await })
                },
            );
            let mut cooperation = ScalarCooperation::with_budget(OptimSolver::Lsqnonlin, budget);
            let mut future = Box::pin(tensor_values(&tensor, &mut cooperation));
            let waker = noop_waker();
            let result = future.as_mut().poll(&mut Context::from_waker(&waker));
            if cancel {
                let std::task::Poll::Ready(Err(error)) = result else {
                    panic!("cancellation must terminate the conversion at its turn")
                };
                assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
                assert!(error.source.is_some());
            } else {
                assert!(result.is_pending());
            }
            drop(future);
            assert_eq!(turns.get(), 1);
            assert!(tensor
                .as_f64_slice()
                .unwrap()
                .iter()
                .all(|value| *value == 3.0));
        }
    }
}
