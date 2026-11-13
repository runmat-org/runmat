use super::super::{DeepLearningConv1DPaddingValue, DeepLearningConv1DRequest};

#[cfg(test)]
mod tests;

pub(super) fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

// Geometry bounds keep the period below 2^30. The Euclidean coefficient
// product is consequently below 2^60, so u64 arithmetic is exact here.
pub(super) fn inverse_mod(value: u32, modulus: u32) -> u32 {
    if modulus == 1 {
        return 0;
    }
    let modulus = u64::from(modulus);
    let (mut old_remainder, mut remainder) = (modulus, u64::from(value) % modulus);
    let (mut old_coefficient, mut coefficient) = (0u64, 1u64);
    while remainder != 0 {
        let quotient = old_remainder / remainder;
        (old_remainder, remainder) = (remainder, old_remainder % remainder);
        (old_coefficient, coefficient) = (
            coefficient,
            (old_coefficient + modulus - (quotient * coefficient) % modulus) % modulus,
        );
    }
    debug_assert_eq!(old_remainder, 1);
    old_coefficient as u32
}

pub(super) fn reflection_stride(
    request: &DeepLearningConv1DRequest,
    input_len: usize,
) -> (u32, u32) {
    let period = match request.padding_value {
        DeepLearningConv1DPaddingValue::SymmetricIncludeEdge => input_len * 2,
        DeepLearningConv1DPaddingValue::SymmetricExcludeEdge if input_len > 1 => {
            (input_len - 1) * 2
        }
        _ => return (1, 0),
    } as u32;
    let stride = request.stride as u32;
    let divisor = greatest_common_divisor(stride, period);
    let step = period / divisor;
    (step, inverse_mod(stride / divisor, step))
}
