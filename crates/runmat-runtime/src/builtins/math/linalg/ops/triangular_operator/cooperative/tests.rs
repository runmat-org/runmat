use super::super::{Diagonal, TriangularLinearOperator};
use super::*;
use crate::context::host_allocation::HostAllocationError;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use crate::execution::{
    CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError, RuntimeExecutionService,
};
use futures::{executor::block_on, task::noop_waker};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use runmat_value::{SparseTensor, Value};
use std::{cell::Cell, future::Future, num::NonZeroU64, rc::Rc, task::Context};

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let s = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(s.free_bytes()),
            epoch: s.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn context(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    )
}
fn long_lower(n: usize) -> Value {
    let mut pointers = vec![0, n];
    let mut rows: Vec<_> = (0..n).collect();
    let mut values = vec![1.0; n];
    for col in 1..n {
        rows.push(col);
        values.push(1.0);
        pointers.push(rows.len());
    }
    Value::SparseTensor(SparseTensor::new(n, n, pointers, rows, values).unwrap())
}
fn lower<'a>(matrix: &'a Value, context: &RuntimeContext) -> TriangularOperator<'a, f64> {
    let TriangularLinearOperator::RealF64(operator) =
        block_on(TriangularLinearOperator::borrow_cooperatively(
            matrix,
            Triangle::Lower,
            Diagonal::Explicit,
            context,
        ))
        .unwrap()
    else {
        panic!("class changed")
    };
    operator
}

#[test]
fn a_long_single_csc_column_obeys_the_real_execution_quantum() {
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let context = context(&ledger);
    let matrix = long_lower(128);
    let operator = lower(&matrix, &context);
    let calls = Rc::new(Cell::new(0));
    let count = calls.clone();
    let cancel = CancellationSource::new();
    let quantum = NonZeroU64::new(7).unwrap();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(quantum),
        cancel.token(),
        move || {
            count.set(count.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let mut work = OperatorWork::with_test_budget(&context, budget, quantum).unwrap();
    let mut input = vec![2.0; 128];
    input[0] = 1.0;
    let mut output = vec![0.0; 128];
    block_on(apply(&operator, &input, &mut output, &mut work)).unwrap();
    assert_eq!(output, vec![1.0; 128]);
    assert_eq!(
        calls.get(),
        (6 * 128 - 2) / 7,
        "stored edges, RHS and columns all count; no after-the-loop yield"
    );
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        0,
        "borrowed substitution allocates no payload"
    );
}

#[test]
fn issuer_rotation_during_a_turn_is_not_relabelled_as_numerical_failure() {
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let context = context(&ledger);
    let matrix = long_lower(32);
    let operator = lower(&matrix, &context);
    let revoke = ledger.clone();
    let quantum = NonZeroU64::new(3).unwrap();
    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(quantum),
        source.token(),
        move || {
            revoke.revoke();
            Box::pin(async { Ok(()) })
        },
    );
    let mut work = OperatorWork::with_test_budget(&context, budget, quantum).unwrap();
    let mut output = vec![0.0; 32];
    assert!(matches!(
        block_on(apply(&operator, &[1.0; 32], &mut output, &mut work)),
        Err(TriangularOperatorError::Operator(
            super::super::super::sparse_operator::SparseOperatorError::Admission(
                HostAllocationError::StaleAuthority
            )
        ))
    ));
}

#[test]
fn pending_turn_cancellation_preserves_execution_error_and_drops_the_turn() {
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let context = context(&ledger);
    let matrix = long_lower(32);
    let operator = lower(&matrix, &context);
    let dropped = Rc::new(Cell::new(false));
    let flag = dropped.clone();
    struct Probe(Rc<Cell<bool>>);
    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }
    let quantum = NonZeroU64::new(2).unwrap();
    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(quantum),
        source.token(),
        move || {
            let flag = flag.clone();
            Box::pin(async move {
                let _guard = Probe(flag);
                futures::future::pending().await
            })
        },
    );
    let mut work = OperatorWork::with_test_budget(&context, budget, quantum).unwrap();
    let mut output = vec![0.0; 32];
    let mut future = Box::pin(apply(&operator, &[1.0; 32], &mut output, &mut work));
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&noop_waker()))
        .is_pending());
    source.cancel(CancellationReason::User);
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker())),
        std::task::Poll::Ready(Err(TriangularOperatorError::Execution(
            ExecutionServiceError::Cancelled
        )))
    ));
    drop(future);
    assert!(dropped.get());
}
