use super::*;

pub(super) fn validate_numeric(option: ImwriteOption, number: f64) -> BuiltinResult<()> {
    let error = match option {
        ImwriteOption::Quality if !number.is_finite() || !(0.0..=100.0).contains(&number) => {
            Some("Quality must be a scalar from 0 to 100")
        }
        ImwriteOption::DelayTime
            if !number.is_finite() || number < 0.0 || number >= u64::MAX as f64 =>
        {
            Some(
                "DelayTime must be a finite non-negative duration representable by the GIF encoder",
            )
        }
        ImwriteOption::LoopCount if number.is_nan() || number < 0.0 => {
            Some("LoopCount must be an integer between 0 and 65535, or Inf")
        }
        ImwriteOption::LoopCount if number.is_finite() => {
            let rounded = number.round();
            if (rounded - number).abs() > 1e-6 || rounded > u16::MAX as f64 {
                Some("LoopCount must be an integer between 0 and 65535, or Inf")
            } else {
                None
            }
        }
        ImwriteOption::LoopCount if number.is_infinite() && number.is_sign_negative() => {
            Some("LoopCount must be an integer between 0 and 65535, or Inf")
        }
        _ => None,
    };
    if let Some(message) = error {
        Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            message,
        ))
    } else {
        Ok(())
    }
}

pub(super) fn parse_write_mode(mode: &str) -> BuiltinResult<WriteMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "overwrite" => Ok(WriteMode::Overwrite),
        "append" => Ok(WriteMode::Append),
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "WriteMode must be 'overwrite' or 'append'",
        )),
    }
}
