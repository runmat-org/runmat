use super::*;

pub(super) struct HostAdmission {
    pub(super) format: Vec<FormatElement>,
    pub(super) controls: Option<(Option<usize>, TextscanOptions)>,
}

pub(super) fn preflight(
    runtime: Option<&crate::context::RuntimeContext>,
    input: &Value,
    format: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<HostAdmission> {
    if !matches!(
        input,
        Value::String(_)
            | Value::StringArray(_)
            | Value::CharArray(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Tensor(_)
            | Value::GpuTensor(_)
    ) {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            "textscan: input must be scalar text or a numeric file identifier",
        ));
    }
    if !crate::dispatcher::value_contains_gpu(input) {
        if let Some(fid) = numeric_fid(input) {
            input::validate_fid(fid)?;
            let info = registry::info_for_in(fid, runtime).ok_or_else(|| {
                textscan_error_with(
                    &TEXTSCAN_ERROR_FILE,
                    format!("textscan: invalid file identifier {fid}"),
                )
            })?;
            if !info.access.read {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_FILE,
                    format!("textscan: file identifier {fid} is not open for reading"),
                ));
            }
        } else {
            string_scalar(input, "textOrFileID")?;
        }
    }
    // A GPU tensor is numeric, so it cannot become a text format by download.
    if !matches!(
        format,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            "textscan: expected formatSpec as a string scalar or character vector",
        ));
    }
    let format = parse_format_spec(&string_scalar(format, "formatSpec")?)?;
    let controls = preflight_host_options(rest)?;
    Ok(HostAdmission { format, controls })
}
