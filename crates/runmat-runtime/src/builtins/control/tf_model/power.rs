//! Intrinsic host-vector capacity checks for transfer-function powers.

use super::*;
use runmat_builtins::TF_METHOD_ERROR_COEFFICIENT_CAPACITY;
use std::alloc::Layout;

pub(super) fn preflight_power_capacity(model: &TfModel, exponent: i64) -> BuiltinResult<()> {
    if exponent == 0 {
        return Ok(());
    }
    let magnitude = exponent.unsigned_abs();
    coefficient_count(model.numerator.len(), magnitude)?;
    coefficient_count(model.denominator.len(), magnitude)?;
    Ok(())
}

/// Degree multiplies by the exponent; the additional slot is the constant term.
/// `Layout::array` enforces the allocator's representable byte-size bound, not
/// an arbitrary polynomial-degree policy.
fn coefficient_count(input_count: usize, exponent: u64) -> BuiltinResult<usize> {
    let count = input_count
        .checked_sub(1)
        .and_then(|degree| {
            if degree == 0 {
                Some(0)
            } else {
                usize::try_from(exponent)
                    .ok()
                    .and_then(|power| degree.checked_mul(power))
            }
        })
        .and_then(|degree| degree.checked_add(1))
        .filter(|&count| Layout::array::<Complex64>(count).is_ok());
    count.ok_or_else(|| {
        control_error(
            "tf",
            TF_METHOD_ERROR_COEFFICIENT_CAPACITY
                .identifier
                .expect("capacity error has a stable identifier"),
            TF_METHOD_ERROR_COEFFICIENT_CAPACITY.message,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_minimum_exponent_requires_one_coefficient() {
        assert_eq!(coefficient_count(1, i64::MIN.unsigned_abs()).unwrap(), 1);
    }

    #[test]
    fn nonconstant_power_rejects_unrepresentable_coefficient_layout() {
        let error = coefficient_count(2, i64::MIN.unsigned_abs()).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:tf:CoefficientCapacity"));
        let max_count = (isize::MAX as usize) / std::mem::size_of::<Complex64>();
        assert_eq!(
            coefficient_count(2, (max_count - 1) as u64).unwrap(),
            max_count
        );
        assert_eq!(
            coefficient_count(2, max_count as u64)
                .unwrap_err()
                .identifier(),
            Some("RunMat:tf:CoefficientCapacity")
        );
    }

    #[test]
    fn valid_small_degree_power_is_admitted_without_allocation() {
        assert_eq!(coefficient_count(3, 4).unwrap(), 9);
    }
}
