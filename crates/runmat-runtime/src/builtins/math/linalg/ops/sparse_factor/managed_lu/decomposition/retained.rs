use super::PublicSparseLu;
use crate::builtins::math::linalg::ops::sparse_factor::{
    managed_storage::Work,
    retained::{
        transfer_csc, transfer_permutation, validate_data, RetainedFactorError as Error,
        RetainedFactorScalar,
    },
};
use crate::context::RuntimeContext;
use runmat_value::{NumericFactorCandidate, NumericFactorData};

impl<T: RetainedFactorScalar> PublicSparseLu<T> {
    pub(in crate::builtins::math::linalg) async fn into_factor_data(
        self: Box<Self>,
        runtime: &RuntimeContext,
    ) -> Result<NumericFactorData, Error> {
        self.check_owner(runtime)?;
        if self.rows != self.cols || self.scaling.len() != 0 {
            return Err(Error::Invalid(
                "retained LU requires square unscaled factor data".into(),
            ));
        }
        // Move, not clone, all arrays out of the original known-layout Box.
        // Its receipt remains live through every cooperative transfer turn.
        let Self {
            rows,
            cols,
            lower,
            upper,
            row_perm,
            row_inverse,
            col_perm,
            scaling: _,
            fence,
            _metadata_claim: metadata_claim,
        } = *self;
        let mut work = Work::for_policy(runtime, fence.as_ref())?;
        let lower =
            transfer_csc(rows, cols, lower.ptr, lower.rows, lower.values, &mut work).await?;
        let upper =
            transfer_csc(rows, cols, upper.ptr, upper.rows, upper.values, &mut work).await?;
        let rows = transfer_permutation(row_perm, Some(row_inverse), &mut work).await?;
        let columns = transfer_permutation(col_perm, None, &mut work).await?;
        let data = validate_data(
            NumericFactorCandidate::Lu {
                lower,
                upper,
                rows,
                columns,
                row_scale: None,
            },
            &mut work,
        )
        .await?;
        drop(work);
        drop(metadata_claim);
        Ok(data)
    }
}
