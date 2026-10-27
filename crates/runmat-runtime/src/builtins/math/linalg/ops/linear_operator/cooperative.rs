//! Task-turn-cooperative products over the same validated borrowed views.
//! Only caller-owned output is mutated; yielding never holds a GC or RefCell
//! borrow and never publishes a partial result.
mod work;
use super::LinearOperatorError;
use super::{scalar::LinearOperatorScalar, TypedLinearOperator};
use crate::context::{host_allocation::HostAllocationReceipt, RuntimeContext};
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
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
    }
}
