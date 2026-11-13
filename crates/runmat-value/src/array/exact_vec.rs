//! Exact-capacity unmanaged vectors for physically quoted host values.
//!
//! `Vec::reserve_exact` may legally over-allocate. These constructors allocate
//! the element layout directly, then publish only initialized elements through
//! a `Vec` whose capacity is exactly the requested length.
//! Zero-sized elements are rejected: Rust represents their capacity as
//! `usize::MAX`, which cannot satisfy this capacity contract.

use std::alloc::{alloc, Layout};

use super::ExactHostNumericAllocationError;

/// Allocate an empty vector whose backing has exactly `capacity` element
/// slots. Callers can move already-owned values into it without cloning or
/// relying on `reserve_exact`, whose capacity is permitted to exceed the
/// request.
pub fn exact_capacity_vec<T>(capacity: usize) -> Result<Vec<T>, ExactHostNumericAllocationError> {
    if std::mem::size_of::<T>() == 0 {
        return Err(ExactHostNumericAllocationError::LayoutOverflow);
    }
    if capacity == 0 {
        return Ok(Vec::new());
    }
    let layout = Layout::array::<T>(capacity)
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
    // SAFETY: a non-null result has the exact element-array layout. No element
    // is initialized until the caller pushes it, so the initial length is 0.
    let pointer = unsafe { alloc(layout) }.cast::<T>();
    if pointer.is_null() {
        return Err(ExactHostNumericAllocationError::AllocationFailed);
    }
    Ok(unsafe { Vec::from_raw_parts(pointer, 0, capacity) })
}

pub fn exact_capacity_vec_with<T>(
    len: usize,
    mut initialize: impl FnMut(usize) -> T,
) -> Result<Vec<T>, ExactHostNumericAllocationError> {
    let mut values = exact_capacity_vec(len)?;
    for index in 0..len {
        values.push(initialize(index));
    }
    Ok(values)
}

pub fn exact_capacity_vec_from_slice<T: Clone>(
    values: &[T],
) -> Result<Vec<T>, ExactHostNumericAllocationError> {
    exact_capacity_vec_with(values.len(), |index| values[index].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_capacity_matches_initialized_length() {
        let values = exact_capacity_vec_with(17, |index| index * 3).unwrap();
        assert_eq!(values.len(), 17);
        assert_eq!(values.capacity(), 17);
        assert_eq!(values[16], 48);
        assert_eq!(
            exact_capacity_vec_with(0, |_| 0_usize).unwrap().capacity(),
            0
        );
        for len in [0, 1] {
            assert!(matches!(
                exact_capacity_vec_with(len, |_| ()),
                Err(ExactHostNumericAllocationError::LayoutOverflow)
            ));
        }
    }

    #[test]
    fn empty_exact_capacity_can_be_filled_without_reallocation() {
        let mut values = exact_capacity_vec::<u64>(9).unwrap();
        let pointer = values.as_ptr();
        values.extend(0..9);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values.capacity(), 9);
        assert!(matches!(
            exact_capacity_vec::<()>(1),
            Err(ExactHostNumericAllocationError::LayoutOverflow)
        ));
    }
}
