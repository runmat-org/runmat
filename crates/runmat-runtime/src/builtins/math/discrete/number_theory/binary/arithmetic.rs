pub(in crate::builtins::math::discrete) fn lcm(left: u128, right: u128) -> u128 {
    if left == 0 || right == 0 {
        return 0;
    }
    left / gcd(left, right) * right
}

pub(in crate::builtins::math::discrete) fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

pub(in crate::builtins::math::discrete) fn extended_gcd(
    left: u128,
    right: u128,
) -> (u128, i128, i128) {
    let (mut old_remainder, mut remainder) = (left, right);
    let (mut old_left, mut left_coefficient) = (1i128, 0i128);
    let (mut old_right, mut right_coefficient) = (0i128, 1i128);
    while remainder != 0 {
        let quotient = (old_remainder / remainder) as i128;
        (old_remainder, remainder) = (remainder, old_remainder - (quotient as u128) * remainder);
        (old_left, left_coefficient) = (left_coefficient, old_left - quotient * left_coefficient);
        (old_right, right_coefficient) =
            (right_coefficient, old_right - quotient * right_coefficient);
    }
    (old_remainder, old_left, old_right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_binary_algorithms_cover_zero_and_bezout() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(lcm(12, 18), 36);
        let (divisor, left, right) = extended_gcd(30, 56);
        assert_eq!(divisor, 2);
        assert_eq!(30i128 * left + 56i128 * right, 2);
    }
}
