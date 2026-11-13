//! Move host values through gather while retaining the canonical provider download path.

use std::future::Future;
use std::pin::Pin;

use runmat_value::Value;

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
    let runtime = crate::context::legacy::active();
    if super::managed_gather::is_managed(runtime.as_ref())? {
        return super::managed_gather::owned(
            value,
            runtime.as_ref().expect("managed gather context"),
            budget,
        )
        .await;
    }
    gather_owned(value, budget).await
}

fn gather_owned<'a>(
    mut value: Value,
    budget: Option<&'a mut CpuCooperationBudget>,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        runmat_value::validate_no_transient_sequence(&value).map_err(|_| {
            crate::runtime_error::semantic_error(
                "TransientSequenceBelowDispatch",
                "legacy output-list carrier reached scalar gather dispatch",
            )
        })?;
        let context = crate::context::legacy::active();
        let owner =
            crate::context::host_allocation::HostAllocationReceipt::capture_owner(context.as_ref())
                .map_err(|source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        context.as_ref(),
                        "RunMat:gather:Ownership",
                        "gather ownership admission failed",
                        source,
                    )
                })?;
        let _roots = super::structure_alias::lease(&value, context.as_ref())?;
        gather_in_place(&mut value, budget).await?;
        if let Some(owner) = owner {
            owner
                .check(context.as_ref().expect("managed owner has context"))
                .map_err(|source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        context.as_ref(),
                        "RunMat:gather:Ownership",
                        "gather authority changed",
                        source,
                    )
                })?;
        }
        Ok(value)
    })
}

fn gather_in_place<'a>(
    value: &'a mut Value,
    budget: Option<&'a mut CpuCooperationBudget>,
) -> Pin<Box<dyn Future<Output = Result<(), RuntimeError>> + 'a>> {
    Box::pin(async move {
        let mut budget = budget;
        if let Some(budget) = budget.as_deref_mut() {
            budget.checkpoint(0).await.map_err(execution_error)?;
        }
        match value {
            Value::GpuTensor(handle) if !unbounded_shared_cell_policy()? => {
                use crate::builtins::common::gpu_helpers;
                use crate::context::host_allocation::{
                    host_allocation_runtime_error, HostAllocationError, HostAllocationReceipt,
                };

                let context = crate::context::legacy::active().ok_or_else(|| {
                    crate::runtime_error::semantic_error(
                        "ManagedOwnerUnavailable",
                        "resident gather has no active allocation owner",
                    )
                })?;
                let readback = runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(
                    |source| {
                        crate::runtime_error::with_provider_execution_source(
                            crate::build_runtime_error(
                                "gather: exact source provider is unavailable or not authorized",
                            )
                            .with_identifier("RunMat:gather:ProviderUnavailable")
                            .build(),
                            source.into(),
                        )
                    },
                )?;
                let quote = gpu_helpers::quote_exact_native_readback(&readback)?;
                let receipt = HostAllocationReceipt::admit(
                    Some(&context),
                    quote.single_source_plan().map_err(|source| {
                        host_allocation_runtime_error(
                            Some(&context),
                            "RunMat:gather:Readback",
                            "resident gather extent failed",
                            source,
                        )
                    })?,
                )
                .map_err(|source| {
                    host_allocation_runtime_error(
                        Some(&context),
                        "RunMat:gather:Readback",
                        "resident gather admission failed",
                        source,
                    )
                })?
                .ok_or_else(|| {
                    host_allocation_runtime_error(
                        Some(&context),
                        "RunMat:gather:Readback",
                        "resident gather lost its managed owner",
                        HostAllocationError::MissingManagedOwner,
                    )
                })?;
                let download = gpu_helpers::download_exact_native_readback_with_diagnostics(
                    &context,
                    &readback,
                    &receipt,
                    quote,
                    gpu_helpers::NumericDownloadDiagnostic::Gather,
                );
                let gathered = match budget.as_deref() {
                    Some(budget) => budget
                        .await_cancellable(download)
                        .await
                        .map_err(execution_error)?,
                    None => download.await,
                }
                .map_err(|failure| failure.error)?;
                receipt.checkpoint(&context).map_err(|source| {
                    host_allocation_runtime_error(
                        Some(&context),
                        "RunMat:gather:Readback",
                        "resident gather authority changed",
                        source,
                    )
                })?;
                *value = gathered;
                Ok(())
            }
            Value::Cell(cell) => {
                if let Some(elements) = cell.try_data_mut_unique() {
                    for element in elements {
                        if let Some(budget) = budget.as_deref_mut() {
                            budget.checkpoint(1).await.map_err(execution_error)?;
                        }
                        if !is_host_leaf(element) {
                            gather_in_place(element, budget.as_deref_mut()).await?;
                        }
                    }
                    Ok(())
                } else if cell
                    .data()
                    .iter()
                    .any(super::owned_gpu::has_transformable_gpu)
                {
                    if unbounded_shared_cell_policy()? {
                        let gather = gather_if_needed_async_impl(value);
                        *value = match budget.as_deref() {
                            Some(budget) => budget
                                .await_cancellable(gather)
                                .await
                                .map_err(execution_error)?,
                            None => gather.await,
                        }?;
                        Ok(())
                    } else {
                        let context = crate::context::legacy::active().ok_or_else(|| {
                            crate::runtime_error::semantic_error(
                                "ManagedOwnerUnavailable",
                                "shared Cell gather has no active allocation owner",
                            )
                        })?;
                        let (replacement, receipt) =
                            super::cell_alias::prepare_shared_cell(cell, &context)?;
                        *value = Value::Cell(replacement);
                        gather_in_place(value, budget.as_deref_mut()).await?;
                        receipt.checkpoint(&context).map_err(|source| {
                            crate::context::host_allocation::host_allocation_runtime_error(
                                Some(&context),
                                "RunMat:gather:SharedCell",
                                "shared Cell gather authority changed",
                                source,
                            )
                        })?;
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            }
            Value::Struct(record) => {
                if !record
                    .fields
                    .values()
                    .any(super::owned_gpu::has_transformable_gpu)
                {
                    return Ok(());
                }
                let context = crate::context::legacy::active();
                let mut _replacement_roots = None;
                if !record.fields.is_unique() {
                    let (replacement, roots) =
                        super::structure_alias::record(record, context.as_ref())?;
                    *record = replacement;
                    _replacement_roots = Some(roots);
                }
                for child in record
                    .fields
                    .try_values_mut_unique()
                    .expect("explicit unique map owner")
                {
                    if let Some(budget) = budget.as_deref_mut() {
                        budget.checkpoint(1).await.map_err(execution_error)?;
                    }
                    if !is_host_leaf(child) {
                        gather_in_place(child, budget.as_deref_mut()).await?;
                    }
                }
                Ok(())
            }
            // These aggregate families can carry transient or resident
            // descendants. The borrowed canonical walk owns their semantics
            // until an exhaustive owned traversal is available.
            aggregate if !is_host_leaf(aggregate) => {
                let gather = gather_if_needed_async_impl(aggregate);
                *aggregate = match budget.as_deref() {
                    Some(budget) => budget
                        .await_cancellable(gather)
                        .await
                        .map_err(execution_error)?,
                    None => gather.await,
                }?;
                Ok(())
            }
            _ => Ok(()),
        }
    })
}

pub(super) fn unbounded_shared_cell_policy() -> Result<bool, RuntimeError> {
    use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};

    let runtime = crate::context::legacy::active();
    HostAllocationReceipt::policy(runtime.as_ref())
        .map(|policy| policy == HostAllocationPolicy::Unbounded)
        .map_err(|source| {
            crate::build_runtime_error(format!("gather: host policy: {source}"))
                .with_identifier(format!("RunMat:{}", source.runtime_identifier_suffix()))
                .with_source(source)
                .build()
        })
}

fn is_host_leaf(value: &Value) -> bool {
    match value {
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::ScopedFunctionHandle(_)
        | Value::ClassRef(_)
        | Value::MException(_) => true,
        Value::Cell(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::GpuTensor(_)
        | Value::Object(_)
        | Value::ObjectArray(_)
        | Value::Closure(_)
        | Value::OutputList(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_) => false,
    }
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
    use runmat_value::{CellArray, CharArray, Closure, IntValue, ObjectInstance};
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
        let original = chars.data().as_ptr();
        let input = Value::Cell(
            CellArray::new_with_shape(
                vec![Value::CharArray(chars), Value::Int(IntValue::U64(u64::MAX))],
                vec![1, 2, 1],
            )
            .unwrap(),
        );
        let Value::Cell(source) = &input else {
            unreachable!("constructed cell")
        };
        let original_collection = source.data().as_ptr();
        let Value::Cell(output) = block_on(gather_if_needed_owned_async(input, None)).unwrap()
        else {
            panic!("expected cell");
        };
        assert_eq!(output.shape, [1, 2, 1]);
        assert_eq!(output.data().as_ptr(), original_collection);
        assert!(
            matches!(&output.data()[0], Value::CharArray(value) if value.data().as_ptr() == original)
        );
        assert!(
            matches!(&output.data()[1], Value::Int(IntValue::U64(value)) if *value == u64::MAX)
        );
    }

    #[test]
    fn nested_resident_failure_keeps_provider_boundary_and_owner() {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let input = Value::Cell(
            CellArray::new(vec![Value::GpuTensor(allocation.handle.clone())], 1, 1).unwrap(),
        );
        let retained_alias = input.clone();
        let error = block_on(gather_if_needed_owned_async(input, None))
            .expect_err("unregistered provider must reject");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gather:ProviderUnavailable")
        );
        drop(retained_alias);
        assert_eq!(allocation.probe.retirements(), 0);
        allocation.handle.release().unwrap();
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn transient_output_list_is_rejected_at_top_level_and_inside_cell() {
        for input in [
            Value::output_list(vec![Value::Num(1.0)]),
            Value::Cell(
                CellArray::new(vec![Value::output_list(vec![Value::Num(1.0)])], 1, 1).unwrap(),
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
    fn multilevel_cell_moves_host_leaf_but_rejects_nested_transient() {
        let chars = CharArray::new_row("2024-04-09");
        let pointer = chars.data().as_ptr();
        let nested = Value::Cell(
            CellArray::new(
                vec![Value::Cell(
                    CellArray::new(vec![Value::CharArray(chars)], 1, 1).unwrap(),
                )],
                1,
                1,
            )
            .unwrap(),
        );
        let Value::Cell(outer) = block_on(gather_if_needed_owned_async(nested, None)).unwrap()
        else {
            panic!("outer cell");
        };
        let Value::Cell(inner) = &outer.data()[0] else {
            panic!("inner cell");
        };
        assert!(
            matches!(&inner.data()[0], Value::CharArray(value) if value.data().as_ptr() == pointer)
        );

        let nested_transient = Value::Cell(
            CellArray::new(
                vec![Value::Cell(
                    CellArray::new(vec![Value::output_list(vec![Value::Num(1.0)])], 1, 1).unwrap(),
                )],
                1,
                1,
            )
            .unwrap(),
        );
        let error = block_on(gather_if_needed_owned_async(nested_transient, None))
            .expect_err("nested transient must not bypass scalar gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:TransientSequenceBelowDispatch")
        );
    }

    #[test]
    fn host_aggregates_preserve_canonical_transient_descendant_rejection() {
        let transient = || Value::output_list(vec![Value::Num(1.0)]);
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("payload", transient());
        let mut object = ObjectInstance::new("GatherFixture");
        object.replace_property_unbounded("Payload".into(), transient());
        let closure = Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "f".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![transient()].into(),
        });
        for input in [
            Value::Struct(structure.finish()),
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
