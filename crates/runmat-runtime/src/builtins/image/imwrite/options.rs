use super::*;

pub(super) fn validate_numeric(option: ImwriteOption, number: f64) -> BuiltinResult<()> {
    option
        .validate_numeric(number)
        .map_err(|message| imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_OPTION, message))
}

pub(super) fn parse_write_mode(mode: &str) -> BuiltinResult<WriteMode> {
    WriteMode::from_name(mode).ok_or_else(|| {
        imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "WriteMode must be 'overwrite' or 'append'",
        )
    })
}
