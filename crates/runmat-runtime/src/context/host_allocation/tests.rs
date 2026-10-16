use std::rc::Rc;

use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use runmat_value::{ExactHostTensor2dAllocation, NumericDType};

use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};

struct ManagedParallel(ManagedHostEffectLedger);

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
fn ordinary_numeric_call_shares_one_ledger_and_transfers_retained_output() {
    let scratch = HostAllocationPlan::numeric_workspace_bytes(NumericDType::F64, 4).unwrap();
    let output = ExactHostTensor2dAllocation::requested_bytes(NumericDType::F64, 1, 4).unwrap();
    let output = u64::try_from(output).unwrap();
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
    let output = ExactHostTensor2dAllocation::requested_bytes(NumericDType::F64, 1, 2).unwrap();
    let plan = HostAllocationPlan::new(0, u64::try_from(output).unwrap()).unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(u64::try_from(output).unwrap(), 11);
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
