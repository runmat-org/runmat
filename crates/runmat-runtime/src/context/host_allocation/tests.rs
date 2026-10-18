use std::rc::Rc;

use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use runmat_value::{
    ExactHostAllocationPermit, ExactHostTensor2dAllocation, NumericDType, NumericScalar, Value,
};

use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};

struct ManagedParallel(ManagedHostEffectLedger);

struct StaleReportedParallel(ManagedHostEffectLedger, u64);

impl RuntimeParallelService for StaleReportedParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: Some(self.0.snapshot().free_bytes()),
            epoch: self.1,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

impl RuntimeParallelService for ManagedParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

struct UnmanagedFinite;

impl RuntimeParallelService for UnmanagedFinite {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: Some(64),
            ..RuntimeParallelResources::default()
        }
    }
}

fn runtime(ports: RuntimeServicePorts) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(ports)
}

#[test]
fn trace_scratch_growth_claims_exact_live_bytes_and_revokes_new_growth() {
    let ledger = ManagedHostEffectLedger::new_with_epoch(4_096, 7);
    let cancellation = CancellationSource::new();
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    let scratch = HostAllocationReceipt::dynamic_trace_scratch(&runtime)
        .unwrap()
        .expect("finite trace issuer");
    let claim = scratch.reserve_exact_claim(128).expect("admitted growth");
    let charged = 128 + crate::context::host_effect::owned_output_claim_metadata_bytes();
    assert_eq!(ledger.snapshot().live_reserved_bytes, charged as u64);
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged as u64);

    cancellation.cancel(CancellationReason::User);
    assert!(scratch.reserve_exact_claim(1).is_err());
    assert_eq!(
        scratch.source_failure(runmat_value::ExactHostNumericAllocationError::ExceedsPermit),
        HostAllocationError::Cancelled
    );
    drop(claim);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn admitted_collection_blocks_survive_retirement_without_new_growth() {
    use allocator_api2::alloc::Allocator;
    use std::alloc::Layout;

    for layout in [
        Layout::from_size_align(37, 64).unwrap(),
        Layout::from_size_align(0, 64).unwrap(),
    ] {
        let ledger = ManagedHostEffectLedger::new(65_536);
        let context = runtime(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
        let physical = AdmittedCollectionAllocator::requested_bytes(layout).unwrap();
        let charged = HostAllocationPlan::retained_exact_output_bytes(
            physical as u64,
            AdmittedCollectionAllocator::retained_claim_count(),
        )
        .unwrap();
        let receipt = HostAllocationReceipt::admit(
            Some(&context),
            HostAllocationPlan::new(0, charged).unwrap(),
        )
        .unwrap()
        .unwrap();
        let allocator = AdmittedCollectionAllocator::new(receipt.output_permit(&context).unwrap());
        let block = allocator.allocate(layout).unwrap();
        let pointer = block.cast::<u8>();
        assert_eq!(pointer.as_ptr().align_offset(layout.align()), 0);
        // SAFETY: the returned block has exactly the requested writable size.
        unsafe { pointer.as_ptr().write_bytes(0x5a, layout.size()) };
        drop(receipt);
        assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
        assert_eq!(ledger.snapshot().live_reserved_bytes, charged);
        let replacement = Layout::from_size_align(layout.size() + 1, layout.align()).unwrap();
        // SAFETY: pointer/layout identify the live block. Failed growth must
        // leave it owned by the original allocation, even after revocation.
        assert!(unsafe { allocator.clone().grow(pointer, layout, replacement) }.is_err());
        for index in 0..layout.size() {
            // SAFETY: the original block remains live after failed growth.
            assert_eq!(unsafe { pointer.as_ptr().add(index).read() }, 0x5a);
        }
        // SAFETY: consume the sole block with its original exact layout.
        unsafe { allocator.deallocate(pointer, layout) };
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn operation_collection_growth_uses_only_the_installed_receipt() {
    use allocator_api2::alloc::Allocator;
    use std::alloc::Layout;

    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let layout = Layout::from_size_align(41, 16).unwrap();
    let block = HostAllocationPlan::retained_exact_output_bytes(
        AdmittedCollectionAllocator::requested_bytes(layout).unwrap() as u64,
        AdmittedCollectionAllocator::retained_claim_count(),
    )
    .unwrap();
    let first = HostAllocationReceipt::admit(
        Some(&context),
        HostAllocationPlan::new(0, OperationPermitGate::retained_bytes().unwrap() + block).unwrap(),
    )
    .unwrap()
    .unwrap();
    let gate = OperationPermitGate::prepare(&first.output_permit(&context).unwrap()).unwrap();
    let allocator = OperationCollectionAllocator::new(gate.clone());
    assert!(allocator.allocate(layout).is_err());
    let installed = gate
        .install(first.output_permit(&context).unwrap())
        .unwrap();
    assert!(matches!(
        gate.install(first.output_permit(&context).unwrap()),
        Err(OperationPermitGateError::AlreadyInstalled)
    ));
    let first_block = allocator.allocate(layout).unwrap().cast::<u8>();
    drop(installed);
    drop(first);
    assert!(allocator.allocate(layout).is_err());

    let second =
        HostAllocationReceipt::admit(Some(&context), HostAllocationPlan::new(0, block).unwrap())
            .unwrap()
            .unwrap();
    let installed = gate
        .install(second.output_permit(&context).unwrap())
        .unwrap();
    let second_block = allocator.allocate(layout).unwrap().cast::<u8>();
    drop(installed);
    drop(second);
    unsafe {
        allocator.deallocate(first_block, layout);
        allocator.deallocate(second_block, layout);
    }
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        OperationPermitGate::retained_bytes().unwrap()
    );
    drop(allocator);
    drop(gate);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn session_random_state_claim_survives_snapshot_and_rehomes_before_cow() {
    use crate::builtins::common::random::RandomState;

    let ledger = ManagedHostEffectLedger::new_with_epoch(16_384, 19);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let mut state = RandomState::seeded(41);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    state.prepare_mutation(Some(&runtime)).unwrap();
    let one_state = ledger.snapshot().live_claimed_bytes;
    assert!(one_state > 624 * 4);
    let saved = state.clone();
    state.advance_uniforms(1, Some(&runtime)).unwrap();
    assert_eq!(ledger.snapshot().live_claimed_bytes, 2 * one_state);

    let next_ledger = ManagedHostEffectLedger::new_with_epoch(16_384, 20);
    let rebound = runtime.clone().with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(next_ledger.clone()))),
    );
    state.advance_uniforms(1, Some(&rebound)).unwrap();
    assert_eq!(ledger.snapshot().live_claimed_bytes, one_state);
    assert_eq!(next_ledger.snapshot().live_claimed_bytes, one_state);
    drop(saved);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    drop(state);
    assert_eq!(next_ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn session_random_state_rejects_missing_owner_and_cancel_before_claim() {
    use crate::builtins::common::random::RandomState;

    let unmanaged = runtime(RuntimeServicePorts::default().with_parallel(Rc::new(UnmanagedFinite)));
    let mut state = RandomState::seeded(41);
    assert!(state.prepare_mutation(Some(&unmanaged)).is_err());
    assert_eq!(state, RandomState::seeded(41));

    let ledger = ManagedHostEffectLedger::new_with_epoch(16_384, 21);
    let cancellation = CancellationSource::new();
    let cancelled = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    cancellation.cancel(CancellationReason::User);
    assert!(state.prepare_mutation(Some(&cancelled)).is_err());
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn retained_state_rechecks_ledgers_epoch_even_if_service_reports_stale_epoch() {
    use crate::builtins::common::random::RandomState;

    let ledger = ManagedHostEffectLedger::new_with_epoch(16_384, 27);
    let runtime = runtime(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(StaleReportedParallel(ledger.clone(), 27))),
    );
    let mut state = RandomState::seeded(41);
    state.prepare_mutation(Some(&runtime)).unwrap();
    let live = ledger.snapshot().live_claimed_bytes;
    ledger.set_epoch_for_test(28);
    assert!(state.prepare_mutation(Some(&runtime)).is_err());
    assert_eq!(ledger.snapshot().live_claimed_bytes, live);
    drop(state);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn ordinary_numeric_call_shares_one_ledger_and_transfers_retained_output() {
    let scratch = HostAllocationPlan::numeric_workspace_bytes(NumericDType::F64, 4).unwrap();
    let output = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[1, 4]).unwrap();
    let plan = HostAllocationPlan::new(scratch, output).unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(scratch + output, 7);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .unwrap()
        .unwrap();
    assert_eq!(ledger.snapshot().live_reserved_bytes, scratch + output);
    let mut workspace = receipt
        .temporary_numeric(&runtime, NumericDType::F64, 4)
        .unwrap();
    workspace
        .as_f64_slice_mut()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ledger.snapshot().live_claimed_bytes, scratch);
    drop(workspace);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let mut native_samples = receipt
        .temporary_numeric(&runtime, NumericDType::I16, 4)
        .unwrap();
    native_samples.set_value(0, NumericScalar::I16(17)).unwrap();
    assert!(matches!(
        native_samples.data_view(),
        runmat_accelerate_api::HostNumericDataView::I16(values) if values == [17, 0, 0, 0]
    ));
    drop(native_samples);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let tensor = receipt
        .with_output_allocation(&runtime, |permit| {
            let mut allocation = ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 4, permit)?;
            allocation
                .as_f64_slice_mut()
                .unwrap()
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            allocation.into_tensor()
        })
        .unwrap();
    assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, output);
    assert_eq!(ledger.snapshot().live_claimed_bytes, output);
    drop(tensor);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn complex_scratch_uses_one_claimed_interleaved_real_backing() {
    let scratch = HostAllocationPlan::complex_workspace_bytes(3).unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(scratch, 7);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let receipt =
        HostAllocationReceipt::admit(Some(&runtime), HostAllocationPlan::new(scratch, 0).unwrap())
            .unwrap()
            .unwrap();
    let mut workspace = receipt.temporary_complex(&runtime, 3).unwrap();
    assert_eq!(workspace.len(), 3);
    assert_eq!(
        workspace.get(2),
        Some(num_complex::Complex64::new(0.0, 0.0))
    );
    assert_eq!(
        workspace.set(2, num_complex::Complex64::new(4.0, -1.0)),
        Some(())
    );
    assert_eq!(
        workspace.get(2),
        Some(num_complex::Complex64::new(4.0, -1.0))
    );
    assert_eq!(workspace.get(3), None);
    assert_eq!(ledger.snapshot().live_claimed_bytes, scratch);
    drop(workspace);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let mut lanes = receipt
        .temporary_numeric(&runtime, NumericDType::F64, 6)
        .unwrap();
    lanes.as_complex_f64_slice_mut().unwrap()[1] = runmat_value::ComplexElement(2.0, -3.0);
    assert_eq!(lanes.as_f64_slice().unwrap()[2..4], [2.0, -3.0]);
    drop(lanes);
    let mut singles = receipt
        .temporary_numeric(&runtime, NumericDType::F32, 6)
        .unwrap();
    singles.as_complex_f32_slice_mut().unwrap()[2] = runmat_value::ComplexElement(4.0, -5.0);
    assert_eq!(singles.as_f32_slice_mut().unwrap()[4..6], [4.0, -5.0]);
    drop(singles);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn builtin_continuation_keeps_box_claim_after_ready_and_rechecks_before_poll() {
    let ledger = ManagedHostEffectLedger::new_with_epoch(1 << 20, 19);
    let cancellation = CancellationSource::new();
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    runtime.set_error_namespace("ContinuationScope");

    let polled = Rc::new(std::cell::Cell::new(false));
    let observed = polled.clone();
    let mut pending =
        crate::builtin::admitted_builtin_future(Some(&runtime), "continuation_test", async move {
            observed.set(true);
            Ok(crate::execution::RootedValueSequence::new(
                runmat_value::ValueSequence::single(Value::Num(4.0)).unwrap(),
            )
            .unwrap())
        });
    let charged = ledger.snapshot().live_reserved_bytes;
    assert!(charged > 0);
    cancellation.cancel(CancellationReason::User);
    let error = futures::executor::block_on(pending.as_mut()).unwrap_err();
    assert_eq!(error.identifier(), Some("ContinuationScope:Cancelled"));
    assert!(
        !polled.get(),
        "revoked continuation must not enter builtin body"
    );
    assert!(error
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>())
        .is_some_and(|source| matches!(
            source,
            crate::execution::ExecutionServiceError::Cancelled
        )));
    assert_eq!(ledger.snapshot().live_reserved_bytes, charged);
    drop(pending);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let fresh = runtime.with_cancellation_token(CancellationSource::new().token());
    let mut completed =
        crate::builtin::admitted_builtin_future(Some(&fresh), "continuation_test", async {
            Ok(crate::execution::RootedValueSequence::new(
                runmat_value::ValueSequence::single(Value::Num(5.0)).unwrap(),
            )
            .unwrap())
        });
    let charged = ledger.snapshot().live_reserved_bytes;
    assert!(charged > 0);
    let output = futures::executor::block_on(completed.as_mut()).unwrap();
    assert_eq!(ledger.snapshot().live_reserved_bytes, charged);
    drop(output);
    drop(completed);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    // An unbounded invocation has no allocation receipt, but still owns its
    // cancellation boundary both before entry and after a successful poll.
    for cancel_before_poll in [true, false] {
        let source = CancellationSource::new();
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_cancellation_token(source.token());
        runtime.set_error_namespace("UnboundedContinuation");
        let polled = Rc::new(std::cell::Cell::new(false));
        let observed = polled.clone();
        let inside = source.clone();
        let pending = crate::builtin::admitted_builtin_future(
            Some(&runtime),
            "continuation_test",
            async move {
                observed.set(true);
                inside.cancel(CancellationReason::User);
                Ok(crate::execution::RootedValueSequence::new(
                    runmat_value::ValueSequence::single(Value::Num(6.0)).unwrap(),
                )
                .unwrap())
            },
        );
        if cancel_before_poll {
            source.cancel(CancellationReason::User);
        }
        let error = futures::executor::block_on(pending).unwrap_err();
        assert_eq!(polled.get(), !cancel_before_poll);
        assert_eq!(error.identifier(), Some("UnboundedContinuation:Cancelled"));
    }
}

#[test]
fn owned_cpu_work_retains_scratch_after_cancelled_caller_retires_receipt() {
    let scratch = HostAllocationPlan::numeric_workspace_bytes(NumericDType::F64, 4).unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(scratch, 13);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let receipt =
        HostAllocationReceipt::admit(Some(&runtime), HostAllocationPlan::new(scratch, 0).unwrap())
            .unwrap()
            .unwrap();
    let retained = receipt.retain_scratch_for_cpu_work();
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, scratch);
    drop(retained);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn output_allocation_restores_control_cause_and_drops_unpublished_claim() {
    let output = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[1, 2]).unwrap();
    let plan = HostAllocationPlan::new(0, output).unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(output, 11);
    let cancellation = CancellationSource::new();
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .unwrap()
        .unwrap();
    let result = receipt.with_output_allocation(&runtime, |permit| {
        let mut allocation = ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 2, permit)?;
        allocation
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[3.0, 4.0]);
        cancellation.cancel(CancellationReason::User);
        allocation.into_tensor()
    });
    assert!(matches!(result, Err(HostAllocationError::Cancelled)));
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let fresh = runtime.with_cancellation_token(CancellationSource::new().token());
    let receipt = HostAllocationReceipt::admit(Some(&fresh), plan)
        .unwrap()
        .unwrap();
    let result = receipt.with_output_allocation(&fresh, |permit| {
        fresh.invalidate_host_effect_session_authority();
        ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 2, permit)
    });
    assert!(matches!(result, Err(HostAllocationError::StaleAuthority)));
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn retained_view_claim_outlives_stage_receipt_without_holding_scratch() {
    let view = 80;
    let per_view = HostAllocationPlan::retained_view_bytes(view).unwrap();
    let issuer_bytes = HostAllocationPlan::retained_view_issuer_bytes().unwrap();
    let retained = per_view + issuer_bytes;
    let ledger = ManagedHostEffectLedger::new_with_epoch(retained + per_view + 32, 11);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let receipt = HostAllocationReceipt::admit(
        Some(&runtime),
        HostAllocationPlan::new(32, retained).unwrap(),
    )
    .unwrap()
    .unwrap();
    let permit = receipt.retained_view_permit(&runtime).unwrap();
    let claim = permit.claim_view(view).unwrap();
    let issuer = claim.issuer();
    drop(receipt);
    assert_eq!(ledger.snapshot().live_claimed_bytes, retained);
    assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
    drop(permit);
    let derived = issuer.clone().reserve_view(view).unwrap();
    assert_eq!(ledger.snapshot().live_claimed_bytes, retained + per_view);
    assert!(matches!(
        issuer.clone().reserve_view(view),
        Err(runmat_execution::resource::RetainedHostViewError::ExceedsCapacity)
    ));
    drop(claim);
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        issuer_bytes + per_view
    );
    drop(derived);
    assert_eq!(ledger.snapshot().live_claimed_bytes, issuer_bytes);
    drop(issuer);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn finite_requires_its_managed_owner_and_cancellation_or_rebind_revokes_new_claims() {
    let plan = HostAllocationPlan::new(16, 16).unwrap();
    let unmanaged = runtime(RuntimeServicePorts::default().with_parallel(Rc::new(UnmanagedFinite)));
    assert!(matches!(
        HostAllocationReceipt::admit(Some(&unmanaged), plan),
        Err(HostAllocationError::MissingManagedOwner)
    ));
    let unbounded = runtime(RuntimeServicePorts::default());
    assert!(HostAllocationReceipt::admit(Some(&unbounded), plan)
        .unwrap()
        .is_none());
    assert!(HostAllocationReceipt::admit(None, plan).unwrap().is_none());
    assert!(HostAllocationPlan::new(u64::MAX, 1).is_err());

    let small_ledger = ManagedHostEffectLedger::new_with_epoch(31, 3);
    let small_runtime = runtime(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(ManagedParallel(small_ledger.clone()))),
    );
    assert!(matches!(
        HostAllocationReceipt::admit(Some(&small_runtime), plan),
        Err(HostAllocationError::Resource(_))
    ));
    assert_eq!(small_ledger.snapshot().live_reserved_bytes, 0);

    let ledger = ManagedHostEffectLedger::new_with_epoch(32, 3);
    let cancellation = CancellationSource::new();
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .unwrap()
        .unwrap();
    cancellation.cancel(CancellationReason::User);
    assert!(matches!(
        receipt.checkpoint(&runtime),
        Err(HostAllocationError::Cancelled)
    ));
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let fresh = runtime.with_cancellation_token(CancellationSource::new().token());
    let receipt = HostAllocationReceipt::admit(Some(&fresh), plan)
        .unwrap()
        .unwrap();
    fresh.invalidate_host_effect_session_authority();
    assert!(matches!(
        receipt.checkpoint(&fresh),
        Err(HostAllocationError::StaleAuthority)
    ));
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
