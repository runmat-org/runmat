//! Borrowed ABI bridge for the two concrete supported complex scalars.
//!
//! num-complex 0.4.6 and canonical ComplexElement are repr(C) pairs of the
//! same initialized scalar. No buffer owner, allocation, release authority,
//! capacity, or lifetime is converted here. All loans are tied to the input
//! slice; mutable loans require its unique mutable borrow.
use runmat_value::ComplexElement;
use std::mem::{align_of, offset_of, size_of};

macro_rules! complex_view {
    ($scalar:ty,$shared:ident,$unique:ident) => {
        const _: () = {
            assert!(
                size_of::<ComplexElement<$scalar>>() == size_of::<num_complex::Complex<$scalar>>()
            );
            assert!(
                align_of::<ComplexElement<$scalar>>()
                    == align_of::<num_complex::Complex<$scalar>>()
            );
            assert!(
                offset_of!(ComplexElement<$scalar>, 0)
                    == offset_of!(num_complex::Complex<$scalar>, re)
            );
            assert!(
                offset_of!(ComplexElement<$scalar>, 1)
                    == offset_of!(num_complex::Complex<$scalar>, im)
            );
        };
        pub(super) fn $shared(
            values: &[ComplexElement<$scalar>],
        ) -> &[num_complex::Complex<$scalar>] {
            // SAFETY: concrete compile-time layout assertions above, both
            // scalar fields have identical validity; borrow retains length,
            // alignment, initialization, owner lifetime and shared aliasing.
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
        }
        pub(super) fn $unique(
            values: &mut [ComplexElement<$scalar>],
        ) -> &mut [num_complex::Complex<$scalar>] {
            // SAFETY: identical concrete layout/validity proof. The exclusive
            // input borrow remains exclusive for the entire returned loan.
            unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
        }
    };
}
complex_view!(f32, complex_f32, complex_f32_mut);
complex_view!(f64, complex_f64, complex_f64_mut);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complex_loans_share_initialized_storage_and_do_not_move_the_owner() {
        let mut single = [ComplexElement(1.0f32, -2.0), ComplexElement(-0.0, 3.0)];
        let pointer = single.as_ptr().cast::<num_complex::Complex32>();
        assert_eq!(complex_f32(&single).as_ptr(), pointer);
        complex_f32_mut(&mut single)[0] = num_complex::Complex32::new(4.0, 5.0);
        assert_eq!(single[0], ComplexElement(4.0, 5.0));
        assert!(complex_f32(&single)[1].re.is_sign_negative());
        let mut double = [ComplexElement(1.0f64, -2.0)];
        let pointer = double.as_ptr().cast::<num_complex::Complex64>();
        assert_eq!(complex_f64(&double).as_ptr(), pointer);
        complex_f64_mut(&mut double)[0] = num_complex::Complex64::new(6.0, -7.0);
        assert_eq!(double[0], ComplexElement(6.0, -7.0));
        assert!(complex_f32_mut(&mut []).is_empty());
        assert!(complex_f64(&[]).is_empty());
    }
}
