use std::cmp::Ordering;

use runmat_value::{ComplexElement, ComplexStorage, HostComplexBuffer, NumericStorage};

/// Read-only access to complex values independent of their owning representation.
///
/// Sorting and set operations always create a reordered result, but they do not
/// need to materialize a second input buffer first. This view keeps canonical
/// interleaved host storage borrowed while also supporting promoted tuple data.
pub(super) trait ComplexSequence<T: Copy> {
    fn len(&self) -> usize;
    fn pair_at(&self, index: usize) -> (T, T);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Copy> ComplexSequence<T> for HostComplexBuffer<T> {
    fn len(&self) -> usize {
        HostComplexBuffer::len(self)
    }

    fn pair_at(&self, index: usize) -> (T, T) {
        self[index].into()
    }
}

impl<T: Copy> ComplexSequence<T> for Vec<(T, T)> {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn pair_at(&self, index: usize) -> (T, T) {
        self[index]
    }
}

impl<T: Copy> ComplexSequence<T> for [ComplexElement<T>] {
    fn len(&self) -> usize {
        <[ComplexElement<T>]>::len(self)
    }

    fn pair_at(&self, index: usize) -> (T, T) {
        self[index].into()
    }
}

pub(super) trait SetFloat: Copy + Default + PartialOrd + std::fmt::Debug {
    fn canonical_key(self) -> u64;
    fn compare(self, other: Self) -> Ordering;
    fn is_nan(self) -> bool;
    fn abs(self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn numeric_storage(values: Vec<Self>) -> NumericStorage;
    fn complex_storage(values: Vec<(Self, Self)>) -> ComplexStorage;
}

impl SetFloat for f64 {
    fn canonical_key(self) -> u64 {
        if self.is_nan() {
            0x7ff8_0000_0000_0000
        } else if self == 0.0 {
            0
        } else {
            self.to_bits()
        }
    }

    fn compare(self, other: Self) -> Ordering {
        compare_float(self, other)
    }

    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    fn abs(self) -> Self {
        f64::abs(self)
    }

    fn hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }

    fn atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }

    fn numeric_storage(values: Vec<Self>) -> NumericStorage {
        NumericStorage::F64(values)
    }

    fn complex_storage(values: Vec<(Self, Self)>) -> ComplexStorage {
        ComplexStorage::F64(values.into())
    }
}

impl SetFloat for f32 {
    fn canonical_key(self) -> u64 {
        if self.is_nan() {
            u64::from(0x7fc0_0000u32)
        } else if self == 0.0 {
            0
        } else {
            u64::from(self.to_bits())
        }
    }

    fn compare(self, other: Self) -> Ordering {
        compare_float(self, other)
    }

    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }

    fn abs(self) -> Self {
        f32::abs(self)
    }

    fn hypot(self, other: Self) -> Self {
        f32::hypot(self, other)
    }

    fn atan2(self, other: Self) -> Self {
        f32::atan2(self, other)
    }

    fn numeric_storage(values: Vec<Self>) -> NumericStorage {
        NumericStorage::F32(values)
    }

    fn complex_storage(values: Vec<(Self, Self)>) -> ComplexStorage {
        ComplexStorage::F32(values.into())
    }
}

fn compare_float<T: SetFloat>(a: T, b: T) -> Ordering {
    if a.is_nan() {
        if b.is_nan() {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    } else if b.is_nan() {
        Ordering::Less
    } else {
        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify_float_semantics<T: SetFloat>(zero: T, negative_zero: T, one: T, nan: T) {
        assert_eq!(zero.canonical_key(), negative_zero.canonical_key());
        assert_eq!(nan.compare(nan), Ordering::Equal);
        assert_eq!(one.compare(nan), Ordering::Less);
        assert_eq!(nan.compare(one), Ordering::Greater);
    }

    #[test]
    fn set_float_semantics_match_for_single_and_double() {
        verify_float_semantics(0.0_f32, -0.0_f32, 1.0_f32, f32::NAN);
        verify_float_semantics(0.0_f64, -0.0_f64, 1.0_f64, f64::NAN);

        assert!(matches!(
            f32::numeric_storage(vec![1.0]),
            NumericStorage::F32(_)
        ));
        assert!(matches!(
            f64::complex_storage(vec![(1.0, -1.0)]),
            ComplexStorage::F64(_)
        ));
    }
}
