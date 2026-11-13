use super::{factorial_error, FactorialError};
use crate::BuiltinResult;

pub(super) trait FactorialUnsigned: Copy {
    const MAX_U128: u128;
    fn into_u128(self) -> u128;
    fn from_u128(value: u128) -> Self;
}

pub(super) trait FactorialSigned: Copy {
    const MAX_U128: u128;
    fn is_negative(self) -> bool;
    fn into_u128(self) -> u128;
    fn from_u128(value: u128) -> Self;
}

macro_rules! impl_factorial_unsigned {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl FactorialUnsigned for $ty {
                const MAX_U128: u128 = <$ty>::MAX as u128;
                fn into_u128(self) -> u128 { self as u128 }
                fn from_u128(value: u128) -> Self { value as $ty }
            }
        )+
    };
}

macro_rules! impl_factorial_signed {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl FactorialSigned for $ty {
                const MAX_U128: u128 = <$ty>::MAX as u128;
                fn is_negative(self) -> bool { self < 0 }
                fn into_u128(self) -> u128 { self as u128 }
                fn from_u128(value: u128) -> Self { value as $ty }
            }
        )+
    };
}

impl_factorial_unsigned!(u8, u16, u32, u64);
impl_factorial_signed!(i8, i16, i32, i64);

pub(super) fn evaluate_unsigned<T: FactorialUnsigned>(values: Vec<T>) -> Vec<T> {
    values
        .into_iter()
        .map(|value| T::from_u128(saturating(value.into_u128(), T::MAX_U128)))
        .collect()
}

pub(super) fn evaluate_signed<T: FactorialSigned>(values: Vec<T>) -> BuiltinResult<Vec<T>> {
    if values.iter().any(|value| value.is_negative()) {
        return Err(factorial_error(
            FactorialError::InvalidInput,
            "input values must be real, finite, nonnegative integers",
        ));
    }
    Ok(values
        .into_iter()
        .map(|value| T::from_u128(saturating(value.into_u128(), T::MAX_U128)))
        .collect())
}

fn saturating(n: u128, maximum: u128) -> u128 {
    let mut value = 1_u128;
    let mut factor = 2_u128;
    while factor <= n {
        let Some(next) = value.checked_mul(factor) else {
            return maximum;
        };
        if next > maximum {
            return maximum;
        }
        value = next;
        factor += 1;
    }
    value
}
