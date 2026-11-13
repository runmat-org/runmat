//! Untruncated public QR, sharing numerical fill/reflectors with COD without
//! allowing a solve's numerical-rank cutoff to change A*P = Q*R.
use super::{
    factor::{self, RankPolicy},
    householder, Qr, SparseCompleteQrError as Error, SparseQrScalar, Work,
};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::SparseTensor;

pub(in crate::builtins::math::linalg) struct SparseQrDecomposition<T: SparseQrScalar> {
    qr: Qr<T>,
    fence: Option<HostAllocationReceipt>,
}

impl<T: SparseQrScalar> SparseQrDecomposition<T> {
    pub(in crate::builtins::math::linalg) async fn factor(
        sparse: &SparseTensor,
        pivot: bool,
        runtime: &RuntimeContext,
    ) -> Result<Self, Error> {
        Self::factor_optional(sparse, pivot, Some(runtime)).await
    }
    pub(in crate::builtins::math::linalg) async fn factor_optional(
        sparse: &SparseTensor,
        pivot: bool,
        runtime: Option<&RuntimeContext>,
    ) -> Result<Self, Error> {
        let fence = HostAllocationReceipt::admit(runtime, HostAllocationPlan::new(0, 0)?)?;
        let mut work = Work::for_optional_policy(runtime, fence.as_ref())?;
        work.flush().await?;
        let input = factor::read(sparse, &mut work).await?;
        let qr = factor::decompose(
            input,
            sparse.rows,
            sparse.cols,
            RankPolicy::Decomposition { pivot },
            &mut work,
        )
        .await?;
        work.flush().await?;
        drop(work);
        Ok(Self { qr, fence })
    }

    pub(in crate::builtins::math::linalg) fn check_owner(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<(), HostAllocationError> {
        self.check_owner_optional(Some(runtime))
    }
    pub(in crate::builtins::math::linalg) fn check_owner_optional(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<(), HostAllocationError> {
        super::super::managed_storage::Fence::checkpoint_optional(
            &self.fence.as_ref().map_or(
                super::super::managed_storage::Fence::Unbounded,
                super::super::managed_storage::Fence::Managed,
            ),
            runtime,
        )
    }

    /// Read-only borrow retains this exact original-issuer factor. Consumers
    /// must check_owner after every await and before publication.
    pub(in crate::builtins::math::linalg) fn permutation(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<&[usize], HostAllocationError> {
        self.check_owner(runtime)?;
        Ok(self.qr.permutation.as_slice())
    }

    pub(in crate::builtins::math::linalg) fn r_column(
        &self,
        column: usize,
        runtime: &RuntimeContext,
    ) -> Result<(&[usize], &[T]), Error> {
        self.r_column_optional(column, Some(runtime))
    }
    pub(in crate::builtins::math::linalg) fn r_column_optional(
        &self,
        column: usize,
        runtime: Option<&RuntimeContext>,
    ) -> Result<(&[usize], &[T]), Error> {
        self.check_owner_optional(runtime)?;
        if column >= self.qr.cols {
            return Err(Error::DimensionMismatch);
        }
        Ok(self.qr.r.column(column))
    }

    /// Apply Q or Q^H to admitted caller-owned columns, with no private
    /// basis-reference, dense coefficient or workspace allocation.
    pub(in crate::builtins::math::linalg) async fn apply_q(
        &self,
        columns: &mut [T],
        count: usize,
        adjoint: bool,
        runtime: &RuntimeContext,
    ) -> Result<(), Error> {
        self.apply_q_optional(columns, count, adjoint, Some(runtime))
            .await
    }
    pub(in crate::builtins::math::linalg) async fn apply_q_optional(
        &self,
        columns: &mut [T],
        count: usize,
        adjoint: bool,
        runtime: Option<&RuntimeContext>,
    ) -> Result<(), Error> {
        if self.qr.rows.checked_mul(count) != Some(columns.len()) {
            return Err(Error::DimensionMismatch);
        }
        let mut work = Work::for_optional_policy(runtime, self.fence.as_ref())?;
        work.flush().await?;
        let reflectors = self.qr.rows.min(self.qr.cols);
        for column in 0..count {
            let vector = &mut columns[column * self.qr.rows..(column + 1) * self.qr.rows];
            for step in 0..reflectors {
                let reflector = if adjoint { step } else { reflectors - 1 - step };
                householder::apply(&self.qr.reflectors, reflector, vector, &mut work).await?;
            }
            work.tick().await?;
        }
        work.flush().await?;
        Ok(())
    }
}
