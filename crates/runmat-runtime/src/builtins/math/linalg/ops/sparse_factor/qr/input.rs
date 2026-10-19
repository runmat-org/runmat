use runmat_value::ComplexElement;

use super::{require_unbounded, SparseQrError};
use crate::context::RuntimeContext;

pub(super) fn validate_finite<T>(
    values: &[T],
    context: &RuntimeContext,
    finite: impl Fn(&T) -> bool,
) -> Result<(), SparseQrError> {
    for chunk in values.chunks(1024) {
        require_unbounded(context)?;
        if !chunk.iter().all(&finite) {
            return Err(SparseQrError::NonFiniteInput);
        }
    }
    Ok(())
}

pub(super) fn copy_complex<T: Copy, U>(
    values: &[ComplexElement<T>],
    context: &RuntimeContext,
    mut convert: impl FnMut(ComplexElement<T>) -> U,
) -> Result<Vec<U>, SparseQrError> {
    require_unbounded(context)?;
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(values.len())
        .map_err(|_| SparseQrError::TemporaryAllocationFailed)?;
    for chunk in values.chunks(1024) {
        require_unbounded(context)?;
        converted.extend(chunk.iter().copied().map(&mut convert));
    }
    Ok(converted)
}
