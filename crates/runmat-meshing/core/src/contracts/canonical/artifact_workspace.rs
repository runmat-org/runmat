//! Counted validation scratch; canonical authored inventories stay borrowed.
use super::{artifact::validation_checkpoint, MeshingCancellationSignal, MeshingContractError};
use std::mem::size_of;

pub(super) fn slots<T>(count: usize) -> Result<u64, MeshingContractError> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(capacity)?;
    u64::try_from(bytes).map_err(|_| capacity())
}

pub(super) fn reserve<T>(
    count: usize,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<T>, MeshingContractError> {
    validation_checkpoint(cancellation)?;
    slots::<T>(count)?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| capacity())?;
    Ok(values)
}

pub(super) fn flags(
    count: usize,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, MeshingContractError> {
    let mut values = reserve(count, cancellation)?;
    values.resize(count, 0);
    validation_checkpoint(cancellation)?;
    Ok(values)
}

pub(super) fn unique<T: Eq>(values: &[T]) -> bool {
    values
        .iter()
        .enumerate()
        .all(|(index, value)| !values[..index].contains(value))
}

fn capacity() -> MeshingContractError {
    MeshingContractError::invalid(
        "mesh validation resources",
        "requested scratch exceeds addressable storage",
    )
}
