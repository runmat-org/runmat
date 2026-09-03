use super::*;
use crate::builtins::common::random_args::keyword_of;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RoundingMode {
    Fix,
    Floor,
    Ceil,
    Round,
}

impl RoundingMode {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
        let Some(keyword) = keyword_of(value) else {
            return Err(error(&ERROR_INVALID_INPUT, "rounding mode must be text"));
        };
        match keyword.as_str() {
            "fix" => Ok(Self::Fix),
            "floor" => Ok(Self::Floor),
            "ceil" => Ok(Self::Ceil),
            "round" => Ok(Self::Round),
            _ => Err(error(
                &ERROR_INVALID_INPUT,
                format!("unsupported rounding mode '{keyword}'"),
            )),
        }
    }

    pub(super) fn divide(self, dividend: i128, divisor: i128) -> i128 {
        let quotient = dividend / divisor;
        let remainder = dividend % divisor;
        if remainder == 0 {
            return quotient;
        }
        match self {
            Self::Fix => quotient,
            Self::Floor if (dividend < 0) != (divisor < 0) => quotient - 1,
            Self::Floor => quotient,
            Self::Ceil if (dividend < 0) == (divisor < 0) => quotient + 1,
            Self::Ceil => quotient,
            Self::Round => round_away_from_zero(dividend, divisor),
        }
    }
}

fn round_away_from_zero(dividend: i128, divisor: i128) -> i128 {
    let sign = if (dividend < 0) == (divisor < 0) {
        1
    } else {
        -1
    };
    let numerator = dividend.unsigned_abs();
    let denominator = divisor.unsigned_abs();
    let mut quotient = numerator / denominator;
    if (numerator % denominator).saturating_mul(2) >= denominator {
        quotient += 1;
    }
    (quotient as i128) * sign
}
