//! Exact, deterministic number-theory primitives shared by discrete builtins.

pub(crate) fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == prime {
            return true;
        }
        if n.is_multiple_of(prime) {
            return false;
        }
    }
    let shift = (n - 1).trailing_zeros();
    let odd = (n - 1) >> shift;
    for base in [2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022] {
        if base % n == 0 {
            continue;
        }
        let mut value = modular_power(base % n, odd, n);
        if value == 1 || value == n - 1 {
            continue;
        }
        let mut witness = true;
        for _ in 1..shift {
            value = modular_multiply(value, value, n);
            if value == n - 1 {
                witness = false;
                break;
            }
        }
        if witness {
            return false;
        }
    }
    true
}

pub(crate) fn prime_factors(n: u64) -> Vec<u64> {
    if n < 2 {
        return vec![n];
    }
    let mut factors = Vec::new();
    factor_recursive(n, &mut factors);
    factors.sort_unstable();
    factors
}

fn factor_recursive(n: u64, factors: &mut Vec<u64>) {
    if n == 1 {
        return;
    }
    if is_prime(n) {
        factors.push(n);
        return;
    }
    let divisor = pollard_rho(n);
    factor_recursive(divisor, factors);
    factor_recursive(n / divisor, factors);
}

fn pollard_rho(n: u64) -> u64 {
    if n.is_multiple_of(2) {
        return 2;
    }
    if n.is_multiple_of(3) {
        return 3;
    }
    let mut constant = 1u64;
    loop {
        let mut slow = 2u64;
        let mut fast = 2u64;
        let mut divisor = 1u64;
        while divisor == 1 {
            slow = polynomial(slow, constant, n);
            fast = polynomial(fast, constant, n);
            fast = polynomial(fast, constant, n);
            divisor = gcd(slow.abs_diff(fast), n);
        }
        if divisor != n {
            return divisor;
        }
        constant = constant.wrapping_add(1);
    }
}

fn polynomial(value: u64, constant: u64, modulus: u64) -> u64 {
    ((modular_multiply(value, value, modulus) as u128 + constant as u128) % modulus as u128) as u64
}

fn modular_multiply(left: u64, right: u64, modulus: u64) -> u64 {
    ((left as u128 * right as u128) % modulus as u128) as u64
}

fn modular_power(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = modular_multiply(result, base, modulus);
        }
        base = modular_multiply(base, base, modulus);
        exponent >>= 1;
    }
    result
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_full_width_values() {
        assert!(is_prime(18_446_744_073_709_551_557));
        assert!(!is_prime(u64::MAX));
        assert!(!is_prime(3_215_031_751));
    }

    #[test]
    fn factors_large_semiprime() {
        assert_eq!(
            prime_factors(4_294_967_291 * 4_294_967_279),
            vec![4_294_967_279, 4_294_967_291]
        );
        assert_eq!(prime_factors(0), vec![0]);
        assert_eq!(prime_factors(1), vec![1]);
    }
}
