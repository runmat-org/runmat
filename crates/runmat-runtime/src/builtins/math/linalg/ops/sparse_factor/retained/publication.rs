//! Canonical DATA publication, preserving each moved physical owner.
use super::super::managed_storage::{ClaimedVec, Work};
use super::{RetainedFactorError, RetainedFactorScalar};
use crate::context::host_allocation::HostAllocationError;
use runmat_value::{
    FactorPermutation, NumericFactorBufferAdoption as Adoption, NumericFactorCandidate,
    NumericFactorData, NumericFactorValidation, SparseCscValidation, SparseTensor,
};
use std::num::NonZeroUsize;
pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn transfer_csc<
    T: RetainedFactorScalar,
>(
    rows: usize,
    columns: usize,
    pointers: ClaimedVec<usize>,
    indices: ClaimedVec<usize>,
    values: ClaimedVec<T>,
    work: &mut Work<'_>,
) -> Result<SparseTensor, RetainedFactorError> {
    work.flush().await?;
    let pointer_len = pointers.len();
    let len = values.len();
    if pointer_len
        != columns
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?
        || indices.len() != len
    {
        return Err(RetainedFactorError::Invalid(
            "CSC transfer extent mismatch".into(),
        ));
    }
    let pointers = Adoption::indices(
        pointers.into_adopted(Adoption::index_header_bytes(), work.required_runtime()?)?,
        pointer_len,
    )
    .map_err(RetainedFactorError::Storage)?;
    work.flush().await?;
    let indices = Adoption::indices(
        indices.into_adopted(Adoption::index_header_bytes(), work.required_runtime()?)?,
        len,
    )
    .map_err(RetainedFactorError::Storage)?;
    work.flush().await?;
    let allocation = values.into_adopted(T::header_bytes(), work.required_runtime()?)?;
    let mut validation = SparseCscValidation::new(rows, columns, pointers, indices, len)
        .map_err(RetainedFactorError::Invalid)?;
    loop {
        let count = work.capacity();
        let done = validation
            .advance(NonZeroUsize::new(count).expect("bounded factor work"))
            .map_err(RetainedFactorError::Invalid)?;
        work.charge(count).await?;
        if done {
            break;
        }
    }
    let parts = validation
        .into_validated()
        .map_err(|_| RetainedFactorError::Invalid("incomplete CSC transfer validation".into()))?;
    let result = T::adopt(parts, allocation, len)?;
    work.flush().await?;
    Ok(result)
}

pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn transfer_permutation(
    forward: ClaimedVec<usize>,
    inverse: Option<ClaimedVec<usize>>,
    work: &mut Work<'_>,
) -> Result<FactorPermutation, RetainedFactorError> {
    let len = forward.len();
    let inverse = if let Some(inverse) = inverse {
        inverse
    } else {
        let mut inverse = ClaimedVec::zeroed(len, work).await?;
        for (index, &original) in forward.as_slice().iter().enumerate() {
            if original >= len {
                return Err(RetainedFactorError::Invalid(
                    "permutation index exceeds extent".into(),
                ));
            }
            inverse.as_mut_slice()[original] = index;
            work.tick().await?;
        }
        inverse
    };
    if inverse.len() != len {
        return Err(RetainedFactorError::Invalid(
            "inverse permutation extent mismatch".into(),
        ));
    }
    work.flush().await?;
    let forward = Adoption::indices(
        forward.into_adopted(Adoption::index_header_bytes(), work.required_runtime()?)?,
        len,
    )
    .map_err(RetainedFactorError::Storage)?;
    let inverse = Adoption::indices(
        inverse.into_adopted(Adoption::index_header_bytes(), work.required_runtime()?)?,
        len,
    )
    .map_err(RetainedFactorError::Storage)?;
    work.flush().await?;
    Ok(FactorPermutation { forward, inverse })
}

pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn validate_data(
    data: NumericFactorCandidate,
    work: &mut Work<'_>,
) -> Result<NumericFactorData, RetainedFactorError> {
    let mut validation =
        NumericFactorValidation::new(data).map_err(|e| RetainedFactorError::Invalid(e.into()))?;
    loop {
        let count = work.capacity();
        let done = validation
            .advance(NonZeroUsize::new(count).expect("bounded factor work"))
            .map_err(RetainedFactorError::Invalid)?;
        work.charge(count).await?;
        if done {
            break;
        }
    }
    let data = validation
        .finish()
        .map_err(|e| RetainedFactorError::Invalid(e.into()))?;
    work.flush().await?;
    Ok(data)
}
