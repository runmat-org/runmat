use super::{Qr, SparseCompleteQr};
use crate::builtins::math::linalg::ops::sparse_factor::{
    managed_storage::Work,
    retained::{
        transfer_csc, transfer_permutation, validate_data, RetainedFactorError as Error,
        RetainedFactorScalar,
    },
};
use crate::context::RuntimeContext;
use runmat_value::{NumericFactorCandidate, NumericFactorData, OrthogonalFactorData};

async fn transfer<T: RetainedFactorScalar>(
    data: Qr<T>,
    work: &mut Work<'_>,
) -> Result<OrthogonalFactorData, Error> {
    let Qr {
        rows,
        cols,
        rank,
        r,
        reflectors,
        permutation,
    } = data;
    let r = transfer_csc(rows, cols, r.ptr, r.rows, r.values, work).await?;
    let reflectors = transfer_csc(
        rows,
        rows.min(cols),
        reflectors.ptr,
        reflectors.rows,
        reflectors.values,
        work,
    )
    .await?;
    let permutation = transfer_permutation(permutation, None, work).await?;
    Ok(OrthogonalFactorData {
        r,
        reflectors,
        permutation,
        rank,
    })
}

impl<T: RetainedFactorScalar> SparseCompleteQr<T> {
    pub(in crate::builtins::math::linalg) async fn into_factor_data(
        self,
        complete: bool,
        runtime: &RuntimeContext,
    ) -> Result<NumericFactorData, Error> {
        self.check_owner(runtime)?;
        if !complete && self.cod.is_some() {
            return Err(Error::Invalid(
                "basic QR cannot retain a COD completion".into(),
            ));
        }
        let Self { first, cod, fence } = self;
        let mut work = Work::for_policy(runtime, fence.as_ref())?;
        let first = transfer(first, &mut work).await?;
        let data = if complete {
            let completion = if let Some(cod) = cod {
                Some(transfer(cod, &mut work).await?)
            } else {
                None
            };
            NumericFactorCandidate::Cod { first, completion }
        } else {
            NumericFactorCandidate::Qr(first)
        };
        validate_data(data, &mut work).await
    }
}
