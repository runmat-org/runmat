//! Move host values through gather while retaining the canonical provider download path.

use std::future::Future;
use std::pin::Pin;

use runmat_value::{CellArray, Value};

use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::RuntimeError;

use super::gather_if_needed_async_impl;

/// Gather resident descendants without cloning host cell payloads. Opaque
/// aggregates still use the established borrowed gather until their owned
/// traversal is available; provider and payload validation remain single-owned.
pub async fn gather_if_needed_owned_async(
    value: Value,
    budget: Option<&mut CpuCooperationBudget>,
) -> Result<Value, RuntimeError> {
    gather_owned(value, budget).await
}

fn gather_owned<'a>(
    value: Value,
    budget: Option<&'a mut CpuCooperationBudget>,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        let mut budget = budget;
        if let Some(budget) = budget.as_deref_mut() {
            budget.checkpoint(0).await.map_err(execution_error)?;
        }
        match value {
            Value::OutputList(_) => Err(crate::runtime_error::semantic_error(
                "TransientSequenceBelowDispatch",
                "legacy output-list carrier reached scalar gather dispatch",
            )),
            Value::Cell(cell) => {
                let mut gathered = Vec::with_capacity(cell.data.len());
                for element in cell.data {
                    if let Some(budget) = budget.as_deref_mut() {
                        budget.checkpoint(1).await.map_err(execution_error)?;
                    }
                    gathered.push(gather_owned(element, budget.as_deref_mut()).await?);
                }
                CellArray::new_with_shape(gathered, cell.shape)
                    .map(Value::Cell)
                    .map_err(|error| {
                        crate::build_runtime_error(format!("gather: {error}"))
                            .with_identifier("RunMat:gather:CellShapeError")
                            .build()
                    })
            }
            // These aggregate families can carry transient or resident
            // descendants. The borrowed canonical walk owns their semantics
            // until an exhaustive owned traversal is available.
            aggregate @ (Value::GpuTensor(_)
            | Value::Struct(_)
            | Value::StructArray(_)
            | Value::Object(_)
            | Value::ObjectArray(_)
            | Value::Closure(_)) => {
                let gather = gather_if_needed_async_impl(&aggregate);
                match budget.as_deref() {
                    Some(budget) => budget
                        .await_cancellable(gather)
                        .await
                        .map_err(execution_error)?,
                    None => gather.await,
                }
            }
            other => Ok(other),
        }
    })
}

fn execution_error(source: ExecutionServiceError) -> RuntimeError {
    let mut error = source.clone().into_runtime_error();
    error.source = Some(Box::new(source));
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::CpuCooperationPolicy;
    use futures::executor::block_on;
    use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use runmat_value::{CharArray, Closure, IntValue, ObjectInstance, StructValue};
    use std::cell::Cell;
    use std::error::Error;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    #[test]
    fn owned_cell_gather_moves_host_payload_and_preserves_wide_integer_and_shape() {
        let chars = CharArray::new_row("2024-04-09");
        let original = chars.data.as_ptr();
        let input = Value::Cell(
            CellArray::new_with_shape(
                vec![Value::CharArray(chars), Value::Int(IntValue::U64(u64::MAX))],
                vec![1, 2, 1],
            )
            .unwrap(),
        );
        let Value::Cell(output) = block_on(gather_if_needed_owned_async(input, None)).unwrap()
        else {
            panic!("expected cell");
        };
        assert_eq!(output.shape, [1, 2, 1]);
        assert!(
            matches!(&output.data[0], Value::CharArray(value) if value.data.as_ptr() == original)
        );
        assert!(matches!(&output.data[1], Value::Int(IntValue::U64(value)) if *value == u64::MAX));
    }

    #[test]
    fn nested_resident_failure_keeps_provider_boundary_and_owner() {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let input = Value::Cell(
            CellArray::new(vec![Value::GpuTensor(allocation.handle.clone())], 1, 1).unwrap(),
        );
        let error = block_on(gather_if_needed_owned_async(input, None))
            .expect_err("unregistered provider must reject");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gather:ProviderUnavailable")
        );
        assert_eq!(allocation.probe.retirements(), 0);
        allocation.handle.release().unwrap();
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn transient_output_list_is_rejected_at_top_level_and_inside_cell() {
        for input in [
            Value::OutputList(vec![Value::Num(1.0)]),
            Value::Cell(
                CellArray::new(vec![Value::OutputList(vec![Value::Num(1.0)])], 1, 1).unwrap(),
            ),
        ] {
            let error = block_on(gather_if_needed_owned_async(input, None))
                .expect_err("transient list may not pass scalar gather");
            assert_eq!(
                error.identifier(),
                Some("RunMat:TransientSequenceBelowDispatch")
            );
        }
    }

    #[test]
    fn host_aggregates_preserve_canonical_transient_descendant_rejection() {
        let transient = || Value::OutputList(vec![Value::Num(1.0)]);
        let mut structure = StructValue::new();
        structure.insert("payload", transient());
        let mut object = ObjectInstance::new("GatherFixture");
        object.properties.insert("Payload".into(), transient());
        let closure = Closure {
            function_name: "f".into(),
            bound_function: None,
            captures: vec![transient()],
        };
        for input in [
            Value::Struct(structure),
            Value::Object(object),
            Value::Closure(closure),
        ] {
            let error = block_on(gather_if_needed_owned_async(input, None))
                .expect_err("aggregate child is transient");
            assert_eq!(
                error.identifier(),
                Some("RunMat:TransientSequenceBelowDispatch")
            );
        }
    }

    #[test]
    fn owned_cell_gather_drops_pending_turn_and_cancels_without_output() {
        for cancel in [false, true] {
            let source = CancellationSource::new();
            let dropped = Rc::new(Cell::new(false));
            let dropped_for_turn = dropped.clone();
            let mut budget = CpuCooperationBudget::with_test_turn(
                CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
                source.token(),
                move || {
                    let flag = dropped_for_turn.clone();
                    Box::pin(async move {
                        struct PendingTurn(Rc<Cell<bool>>);
                        impl Drop for PendingTurn {
                            fn drop(&mut self) {
                                self.0.set(true);
                            }
                        }
                        impl Future for PendingTurn {
                            type Output = Result<(), ExecutionServiceError>;
                            fn poll(
                                self: Pin<&mut Self>,
                                _: &mut Context<'_>,
                            ) -> Poll<Self::Output> {
                                Poll::Pending
                            }
                        }
                        PendingTurn(flag).await
                    })
                },
            );
            let input = Value::Cell(
                CellArray::new(vec![Value::CharArray(CharArray::new_row("x"))], 1, 1).unwrap(),
            );
            let mut future = Box::pin(gather_if_needed_owned_async(input, Some(&mut budget)));
            let waker = futures::task::noop_waker();
            assert!(matches!(
                future.as_mut().poll(&mut Context::from_waker(&waker)),
                Poll::Pending
            ));
            if cancel {
                source.cancel(CancellationReason::User);
                let error = match future.as_mut().poll(&mut Context::from_waker(&waker)) {
                    Poll::Ready(Err(error)) => error,
                    _ => panic!("cancel must finish the gather"),
                };
                assert!(matches!(
                    error
                        .source()
                        .and_then(|source| source.downcast_ref::<ExecutionServiceError>()),
                    Some(ExecutionServiceError::Cancelled)
                ));
            }
            drop(future);
            assert!(dropped.get());
        }
    }
}
