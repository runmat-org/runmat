use runmat_value::NumericDType;

use crate::BuiltinResult;

use super::super::error;

pub(super) struct ScaleInput {
    value: f64,
    lower: f64,
    upper: f64,
    input_min: f64,
    input_max: f64,
}

impl From<[f64; 5]> for ScaleInput {
    fn from(values: [f64; 5]) -> Self {
        let [value, lower, upper, input_min, input_max] = values;
        Self {
            value,
            lower,
            upper,
            input_min,
            input_max,
        }
    }
}

pub(super) fn scale(input: ScaleInput) -> BuiltinResult<f64> {
    if input.lower >= input.upper && !input.lower.is_nan() && !input.upper.is_nan() {
        return Err(error::invalid_argument(
            "lower output bounds must be less than upper output bounds",
        ));
    }
    if input.input_min > input.input_max && !input.input_min.is_nan() && !input.input_max.is_nan() {
        return Err(error::invalid_argument(
            "InputMin must be less than or equal to InputMax",
        ));
    }
    if input.input_min == input.input_max {
        return Ok(if input.lower.is_infinite() || input.upper.is_infinite() {
            f64::NAN
        } else {
            input.lower
        });
    }
    let clipped = if input.value < input.input_min {
        input.input_min
    } else if input.value > input.input_max {
        input.input_max
    } else {
        input.value
    };
    Ok(input.lower
        + ((clipped - input.input_min) / (input.input_max - input.input_min))
            * (input.upper - input.lower))
}

pub(super) fn cast(value: f64, dtype: NumericDType) -> f64 {
    if dtype == NumericDType::F32 {
        value as f32 as f64
    } else {
        value
    }
}
