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
        if matches!(value, Value::CharArray(chars)
            if chars.rows != 1 || crate::builtins::common::shape::effective_rank(&chars.shape) > 2)
        {
            return Err(error(
                &ERROR_INVALID_INPUT,
                "rounding mode must be scalar text or a character row",
            ));
        }
        let Some(keyword) = keyword_of(value)? else {
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

    pub(super) fn divide(self, dividend: i128, divisor: i128) -> Option<i128> {
        let quotient = dividend.checked_div(divisor)?;
        let remainder = dividend.checked_rem(divisor)?;
        if remainder == 0 {
            return Some(quotient);
        }
        Some(match self {
            Self::Fix => quotient,
            Self::Floor if (dividend < 0) != (divisor < 0) => quotient - 1,
            Self::Floor => quotient,
            Self::Ceil if (dividend < 0) == (divisor < 0) => quotient + 1,
            Self::Ceil => quotient,
            Self::Round => {
                let remainder = remainder.unsigned_abs();
                let denominator = divisor.unsigned_abs();
                if remainder >= denominator - remainder {
                    quotient.checked_add(if (dividend < 0) == (divisor < 0) {
                        1
                    } else {
                        -1
                    })?
                } else {
                    quotient
                }
            }
        })
    }
}
