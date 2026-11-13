//! Task-turn-cooperative products over the same validated borrowed views.
//! Only caller-owned output is mutated; yielding never holds a GC or RefCell
//! borrow and never publishes a partial result.
mod work;
mod policy;
use super::LinearOperatorError;
use super::{scalar::LinearOperatorScalar, TypedLinearOperator};
use crate::context::{host_allocation::HostAllocationReceipt, RuntimeContext};
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
pub(in crate::builtins::math::linalg) use policy::{BorrowedWork, ProductWork, UnmanagedWork};
use runmat_value::ComplexElement;
pub(in crate::builtins::math::linalg) use work::OperatorWork;

#[derive(Debug)]
pub(crate) enum CooperativeOperatorError {
    Operator(LinearOperatorError),
    Execution(ExecutionServiceError),
}
impl From<crate::context::host_allocation::HostAllocationError> for CooperativeOperatorError {
    fn from(error: crate::context::host_allocation::HostAllocationError) -> Self {
        Self::Operator(LinearOperatorError::Admission(error))
    }
}

async fn apply<T: LinearOperatorScalar>(
    operator: &TypedLinearOperator<'_, T>,
    input: &[T],
    output: &mut [T],
    context: &RuntimeContext,
    budget: &mut CpuCooperationBudget,
) -> Result<(), CooperativeOperatorError> {
    if input.len() != operator.cols() || output.len() != operator.rows() {
        return Err(CooperativeOperatorError::Operator(
            LinearOperatorError::DimensionMismatch,
        ));
    }
    HostAllocationReceipt::policy(Some(context))?;
    for chunk in output.chunks_mut(1024) {
        budget
            .checkpoint(chunk.len() as u64)
            .await
            .map_err(CooperativeOperatorError::Execution)?;
        HostAllocationReceipt::policy(Some(context))?;
        chunk.fill(T::zero());
    }
    for (column, &x) in input.iter().enumerate() {
        // Charge column traversal too: a huge, empty CSC is still work.
        if column % 256 == 0 {
            budget
                .checkpoint((input.len() - column).min(256) as u64)
                .await
                .map_err(CooperativeOperatorError::Execution)?;
            HostAllocationReceipt::policy(Some(context))?;
        }
        match operator {
            TypedLinearOperator::Scalar(value) => T::add_product(&mut output[0], *value, x),
            TypedLinearOperator::Dense(matrix) => {
                let start = column * matrix.rows();
                for (block, coefficients) in matrix.as_slice()[start..start + matrix.rows()]
                    .chunks(1024)
                    .enumerate()
                {
                    budget
                        .checkpoint(coefficients.len() as u64)
                        .await
                        .map_err(CooperativeOperatorError::Execution)?;
                    HostAllocationReceipt::policy(Some(context))?;
                    for (offset, &coefficient) in coefficients.iter().enumerate() {
                        T::add_product(&mut output[block * 1024 + offset], coefficient, x);
                    }
                }
            }
            TypedLinearOperator::Sparse(matrix) => {
                let (rows, values) = matrix.column(column);
                for (row_block, value_block) in rows.chunks(1024).zip(values.chunks(1024)) {
                    budget
                        .checkpoint(value_block.len() as u64)
                        .await
                        .map_err(CooperativeOperatorError::Execution)?;
                    HostAllocationReceipt::policy(Some(context))?;
                    for (&row, &coefficient) in row_block.iter().zip(value_block) {
                        T::add_product(&mut output[row], coefficient, x);
                    }
                }
            }
        }
    }
    HostAllocationReceipt::policy(Some(context))?;
    Ok(())
}
macro_rules! cooperative_product {
    ($scalar:ty) => {
        impl TypedLinearOperator<'_, $scalar> {
            pub(crate) async fn apply_cooperatively(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
                budget: &mut CpuCooperationBudget,
            ) -> Result<(), CooperativeOperatorError> {
                apply(self, input, output, context, budget).await
            }
        }
    };
}
cooperative_product!(f32);
cooperative_product!(f64);
cooperative_product!(ComplexElement<f32>);
cooperative_product!(ComplexElement<f64>);

#[cfg(test)]
mod tests {
    use super::super::{DenseOperator, LinearOperator};
    use super::*;
    use crate::execution::{CpuCooperationPolicy, RuntimeExecutionService};
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::{cell::Cell, num::NonZeroU64, rc::Rc};

    #[test]
    fn long_columns_and_empty_sparse_columns_cooperate_without_output_publication() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let source = CancellationSource::new();
        let token = source.token();
        let calls = Rc::new(Cell::new(0));
        let observed = Rc::clone(&calls);
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1024).unwrap()),
            token,
            move || {
                observed.set(observed.get() + 1);
                if observed.get() == 6 {
                    source.cancel(CancellationReason::User);
                }
                Box::pin(async { Ok(()) })
            },
        );
        let coefficients = vec![2.0_f64; 4096];
        let dense = TypedLinearOperator::Dense(
            DenseOperator::from_column_major(&coefficients, 4096, 1).unwrap(),
        );
        let mut output = vec![9.0; 4096];
        let result = futures::executor::block_on(dense.apply_cooperatively(
            &[3.0],
            &mut output,
            &runtime,
            &mut budget,
        ));
        assert!(matches!(
            result,
            Err(CooperativeOperatorError::Execution(
                ExecutionServiceError::Cancelled
            ))
        ));
        assert_eq!(calls.get(), 6);
        assert_eq!(&output[..1024], &[6.0; 1024]);
        assert_eq!(&output[1024..], &[0.0; 3072]);

        let empty = runmat_value::Value::SparseTensor(
            runmat_value::SparseTensor::new(0, 4096, vec![0; 4097], vec![], vec![]).unwrap(),
        );
        let LinearOperator::RealF64(empty) =
            LinearOperator::borrow_with_context(&empty, &runtime).unwrap()
        else {
            panic!("double CSC")
        };
        let count = Rc::new(Cell::new(0));
        let observed = Rc::clone(&count);
        let source = CancellationSource::new();
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(256).unwrap()),
            source.token(),
            move || {
                observed.set(observed.get() + 1);
                Box::pin(async { Ok(()) })
            },
        );
        futures::executor::block_on(empty.apply_cooperatively(
            &vec![1.0; 4096],
            &mut [],
            &runtime,
            &mut budget,
        ))
        .unwrap();
        assert_eq!(count.get(), 16);
        // A facade cannot reset a partially consumed original quantum.
        let source = CancellationSource::new();
        let partial_runtime = runtime.clone().with_cancellation_token(source.token());
        let turns = Rc::new(Cell::new(0));
        let seen = turns.clone();
        let mut partial = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(8).unwrap()),
            source.token(),
            move || {
                seen.set(seen.get() + 1);
                Box::pin(async { Ok(()) })
            },
        );
        futures::executor::block_on(partial.checkpoint(3)).unwrap();
        assert_eq!(partial.remaining_until_turn().unwrap().get(), 5);
        let mut work = BorrowedWork::new(&partial_runtime, &mut partial).unwrap();
        assert_eq!(ProductWork::capacity(&work), 5);
        futures::executor::block_on(ProductWork::charge(&mut work, 3)).unwrap();
        assert_eq!(ProductWork::capacity(&work), 2);
        assert_eq!(
            turns.get(),
            0,
            "buffered work does not request an early turn"
        );
        futures::executor::block_on(ProductWork::charge(&mut work, 2)).unwrap();
        assert_eq!(turns.get(), 1);
        assert_eq!(ProductWork::capacity(&work), 8);
        futures::executor::block_on(ProductWork::charge(&mut work, 4)).unwrap();
        futures::executor::block_on(ProductWork::flush(&mut work)).unwrap();
        assert_eq!(turns.get(), 1);
        assert_eq!(ProductWork::capacity(&work), 4);
        drop(work);
        assert_eq!(partial.remaining_until_turn().unwrap().get(), 4);
        let mut next = BorrowedWork::new(&partial_runtime, &mut partial).unwrap();
        assert_eq!(ProductWork::capacity(&next), 4);
        futures::executor::block_on(ProductWork::charge(&mut next, 4)).unwrap();
        assert_eq!(
            turns.get(),
            2,
            "same budget crosses exactly two original turns"
        );
        drop(next);
        assert_eq!(partial.remaining_until_turn().unwrap().get(), 8);

        // The same canonical CSC cursor can borrow the actual original
        // quantum/token rather than manufacture a budget for each result.
        let source = CancellationSource::new();
        let runtime = runtime.with_cancellation_token(source.token());
        let sparse = runmat_value::Value::SparseTensor(
            runmat_value::SparseTensor::new(
                4096,
                1,
                vec![0, 4096],
                (0..4096).collect::<Vec<_>>(),
                vec![2.; 4096],
            )
            .unwrap(),
        );
        let turns = Rc::new(Cell::new(0));
        let seen = turns.clone();
        let trigger = source.clone();
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                seen.set(seen.get() + 1);
                if seen.get() == 5 {
                    trigger.cancel(CancellationReason::User);
                }
                Box::pin(async { Ok(()) })
            },
        );
        let borrowed = futures::executor::block_on(LinearOperator::borrow_with_budget(
            &sparse,
            Some(&runtime),
            Some(&mut budget),
        ));
        assert!(matches!(
            borrowed,
            Err(CooperativeOperatorError::Execution(
                ExecutionServiceError::Cancelled
            ))
        ));
        assert_eq!(turns.get(), 5);
        let runmat_value::Value::SparseTensor(matrix) = &sparse else {
            unreachable!();
        };
        assert_eq!(matrix.as_f64_slice().unwrap()[4095], 2.);
        let foreign = CancellationSource::new();
        let effects = Rc::new(Cell::new(0));
        let seen = effects.clone();
        let mut wrong = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            foreign.token(),
            move || {
                seen.set(seen.get() + 1);
                Box::pin(async { Ok(()) })
            },
        );
        let refused = futures::executor::block_on(LinearOperator::borrow_with_budget(
            &sparse,
            Some(&runtime),
            Some(&mut wrong),
        ));
        assert!(matches!(
            refused,
            Err(CooperativeOperatorError::Operator(
                LinearOperatorError::Admission(
                    crate::context::host_allocation::HostAllocationError::StaleAuthority
                )
            ))
        ));
        assert_eq!(effects.get(), 0);
    }
}
