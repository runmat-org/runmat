//! Typed borrowed coefficient conversion; storage and release owners stay put.
use super::ManagedSparseScalar;

/// Admission selects equal native precision and either the same arithmetic
/// domain or real DATA with complex execution. The coefficient is one copied
/// scalar, never an allocation or a borrowed slice reinterpretation.
pub(super) fn coefficient_as<S: ManagedSparseScalar, T: ManagedSparseScalar>(
    value: S,
) -> Option<T> {
    let (real, imag) = value.parts();
    T::from_parts(real, imag)
}
