use super::ManagedSparseCholesky;
use crate::builtins::math::linalg::ops::sparse_factor::{
    managed_storage::Work,
    retained::{
        transfer_csc, transfer_permutation, validate_data, RetainedFactorError as Error,
        RetainedFactorScalar,
    },
};
use crate::context::RuntimeContext;
use runmat_value::{NumericFactorCandidate, NumericFactorData};

impl<T: RetainedFactorScalar> ManagedSparseCholesky<T> {
    pub(in crate::builtins::math::linalg) async fn into_factor_data(
        self: Box<Self>,
        runtime: &RuntimeContext,
    ) -> Result<NumericFactorData, Error> {
        self.check_owner(runtime)?;
        if self.order != self.completed {
            return Err(Error::NotPositiveDefinite);
        }
        let Self {
            order,
            completed: _,
            lower_ptr,
            lower_rows,
            lower_values,
            permutation,
            fence,
            _metadata_claim: metadata_claim,
        } = *self;
        let mut work = Work::for_policy(runtime, fence.as_ref())?;
        let lower =
            transfer_csc(order, order, lower_ptr, lower_rows, lower_values, &mut work).await?;
        let permutation = transfer_permutation(permutation, None, &mut work).await?;
        let data = validate_data(
            NumericFactorCandidate::Cholesky { lower, permutation },
            &mut work,
        )
        .await?;
        drop(work);
        drop(metadata_claim);
        Ok(data)
    }
}
