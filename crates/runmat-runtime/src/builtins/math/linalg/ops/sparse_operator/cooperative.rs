//! Cooperative borrowing uses the canonical incremental CSC validator.
use super::{make_view, SparseLinearOperator, SparseOperatorError};
use crate::builtins::math::linalg::ops::linear_operator::{
    BorrowedWork, CooperativeOperatorError, OperatorWork, ProductWork, UnmanagedWork,
};
use crate::context::RuntimeContext;
use runmat_value::{SparseCscValidation, SparseTensor};
use std::num::NonZeroUsize;

impl<'a> SparseLinearOperator<'a> {
    pub(in crate::builtins::math::linalg) async fn borrow_cooperatively(
        sparse: &'a SparseTensor,
        context: &RuntimeContext,
    ) -> Result<Self, CooperativeOperatorError> {
        Self::borrow_work(sparse, &mut OperatorWork::new(context)?).await
    }
    pub(in crate::builtins::math::linalg) async fn borrow_with_budget(
        sparse: &'a SparseTensor,
        context: Option<&RuntimeContext>,
        budget: Option<&mut crate::execution::CpuCooperationBudget>,
    ) -> Result<Self, CooperativeOperatorError> {
        match (context, budget) {
            (Some(runtime), Some(budget))
                if budget
                    .cancellation_token()
                    .same_generation(&runtime.cancellation()) =>
            {
                Self::borrow_work(sparse, &mut BorrowedWork::new(runtime, budget)?).await
            }
            (None, None) => {
                crate::context::host_allocation::HostAllocationReceipt::policy(None)?;
                Self::borrow_work(sparse, &mut UnmanagedWork).await
            }
            _ => Err(crate::context::host_allocation::HostAllocationError::StaleAuthority.into()),
        }
    }
    async fn borrow_work(
        sparse: &'a SparseTensor,
        work: &mut impl ProductWork,
    ) -> Result<Self, CooperativeOperatorError> {
        work.flush().await?;
        sparse
            .cols
            .checked_add(1)
            .ok_or(CooperativeOperatorError::Operator(
                SparseOperatorError::Overflow,
            ))?;
        // Preserve the existing topology error category; the canonical cursor
        // alone decides validity, without parsing its human diagnostic text.
        let topology = |_| {
            CooperativeOperatorError::Operator(SparseOperatorError::InvalidTopology(
                "canonical CSC topology",
            ))
        };
        let mut validation = SparseCscValidation::new(
            sparse.rows,
            sparse.cols,
            sparse.col_ptrs.clone(),
            sparse.row_indices.clone(),
            sparse.nnz(),
        )
        .map_err(topology)?;
        loop {
            let count = work.capacity();
            let done = validation
                .advance(NonZeroUsize::new(count).expect("nonzero work capacity"))
                .map_err(topology)?;
            work.charge(count).await?;
            if done {
                break;
            }
        }
        work.flush().await?;
        let result = if let Some(values) = sparse.as_f32_slice() {
            Self::RealF32(make_view(sparse, values))
        } else if let Some(values) = sparse.as_f64_slice() {
            Self::RealF64(make_view(sparse, values))
        } else if let Some(values) = sparse.as_complex_f32_slice() {
            Self::ComplexF32(make_view(sparse, values))
        } else if let Some(values) = sparse.as_complex_f64_slice() {
            Self::ComplexF64(make_view(sparse, values))
        } else {
            return Err(CooperativeOperatorError::Operator(
                SparseOperatorError::UnsupportedClass,
            ));
        };
        Ok(result)
    }
}
