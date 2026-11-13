use super::*;
use crate::builtins::math::linalg::ops::linear_operator::OperatorWork;
use crate::builtins::math::linalg::ops::triangular_operator::{
    Diagonal, Triangle, TriangularLinearOperator, TriangularOperatorError,
};
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use futures::task::noop_waker;
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use std::{future::Future, num::NonZeroU64, task::Context};

#[test]
fn pending_substitution_drop_or_cancel_destroys_private_output_and_shape() {
    use super::super::super::super::managed_output::ManagedSolveOutput;
    for cancel_await in [false, true] {
        let ledger = ManagedHostEffectLedger::new(2_000_000);
        let context = context(&ledger);
        let matrix = Value::SparseTensor(
            SparseTensor::new(
                32,
                32,
                (0..=32).collect::<Vec<_>>(),
                (0..32).collect::<Vec<_>>(),
                vec![1.0; 32],
            )
            .unwrap(),
        );
        let TriangularLinearOperator::RealF64(operator) =
            block_on(TriangularLinearOperator::borrow_cooperatively(
                &matrix,
                Triangle::Lower,
                Diagonal::Explicit,
                &context,
            ))
            .unwrap()
        else {
            panic!("class changed")
        };
        let source = CancellationSource::new();
        let quantum = NonZeroU64::new(3).unwrap();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(quantum),
            source.token(),
            || Box::pin(futures::future::pending()),
        );
        let mut future = Box::pin(async {
            let mut output = ManagedSolveOutput::<f64>::new(&[32, 1], &context)?;
            let mut work = OperatorWork::with_test_budget(&context, budget, quantum)?;
            operator
                .apply_with_work(&[1.0; 32], output.as_mut_slice()?, &mut work)
                .await?;
            output
                .finish(&context)
                .map_err(TriangularOperatorError::Admission)
        });
        assert!(future
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker()))
            .is_pending());
        let bytes = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[32, 1]).unwrap();
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            bytes,
            "the pending owner holds final payload and shape, not a second RHS"
        );
        if cancel_await {
            source.cancel(CancellationReason::User);
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(&noop_waker())),
                std::task::Poll::Ready(Err(TriangularOperatorError::Execution(
                    ExecutionServiceError::Cancelled
                )))
            ));
        }
        drop(future);
        empty(&ledger);
    }
}
