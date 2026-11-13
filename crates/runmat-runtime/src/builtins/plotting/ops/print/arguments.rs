use super::*;

#[derive(Debug, Clone)]
pub(super) struct PrintRequest {
    pub(super) figure: FigureHandle,
    pub(super) filename: String,
    pub(super) device: PrintDevice,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl PrintRequest {
    pub(super) fn output_path(&self) -> BuiltinResult<PathBuf> {
        let path = Path::new(&self.filename);
        match self.device {
            PrintDevice::Png if has_extension(path, "png") => Ok(path.to_path_buf()),
            PrintDevice::Png => {
                let Some(file_name) = path.file_name() else {
                    return Err(print_error_with_detail(
                        &PRINT_ERROR_INVALID_INPUT,
                        "filename must include a file name before appending '.png'",
                    ));
                };
                let mut file_name = OsString::from(file_name);
                file_name.push(".png");
                Ok(path.with_file_name(file_name))
            }
        }
    }
}

pub(super) fn parse_print_args(
    args: crate::call::FunctionArgumentView<'_>,
    selected_figure: FigureHandle,
) -> BuiltinResult<PrintRequest> {
    if args.is_empty() {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename is required",
        ));
    }

    let mut idx = 0usize;
    let mut figure = selected_figure;
    if let Some(handle) = figure_handle_arg(&args[0])? {
        figure = handle;
        idx = 1;
    }

    let mut filename: Option<String> = None;
    let mut device: Option<PrintDevice> = None;
    let mut dpi = PRINT_DEFAULT_DPI;

    while idx < args.len() {
        let token = string_arg(&args[idx])?;
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_INPUT,
                "empty string arguments are not valid print options",
            ));
        }

        if trimmed.starts_with('-') {
            parse_option(trimmed, &mut device, &mut dpi)?;
        } else if filename.is_none() {
            filename = Some(trimmed.to_string());
        } else {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_INPUT,
                format!("unexpected extra filename argument '{trimmed}'"),
            ));
        }
        idx += 1;
    }

    let filename = filename.ok_or_else(|| {
        print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename is required for RunMat figure export",
        )
    })?;
    if filename.contains('\0') {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename must not contain NUL bytes",
        ));
    }

    let inferred = infer_device_from_filename(&filename);
    let device = match (device, inferred) {
        (Some(device), _) => device,
        (None, Some(device)) => device,
        (None, None) => PrintDevice::Png,
    };
    let (width, height) = print_dimensions(dpi).map_err(|error| {
        print_error_with_detail(
            &PRINT_ERROR_INVALID_OPTION,
            format!("unsupported resolution: {error:?}"),
        )
    })?;
    Ok(PrintRequest {
        figure,
        filename,
        device,
        width,
        height,
    })
}

pub(super) fn preflight_host_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<PathBuf>> {
    if args.is_empty() {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename is required",
        ));
    }
    let mut filenames = 0usize;
    let mut known_filename = None;
    let mut devices = 0usize;
    let mut dpi = PRINT_DEFAULT_DPI;
    let mut dynamic_text = false;
    for (index, value) in args.iter().enumerate() {
        if index == 0 && figure_handle_arg(value)?.is_some() {
            continue;
        }
        let token = match value {
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => string_arg(value)?,
            // Only a directly resident operand can materialize into a valid
            // scalar handle or text control. Containers are never print
            // controls, even when they contain resident descendants.
            Value::GpuTensor(_) => {
                dynamic_text = true;
                continue;
            }
            other => {
                return Err(print_error_with_detail(
                    &PRINT_ERROR_INVALID_INPUT,
                    format!("expected string option or filename, got {other:?}"),
                ));
            }
        };
        let token = token.trim();
        if token.is_empty() {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_INPUT,
                "empty string arguments are not valid print options",
            ));
        }
        if token.starts_with('-') {
            let mut device = None;
            parse_option(token, &mut device, &mut dpi)?;
            if device.is_some() {
                devices += 1;
            }
        } else {
            filenames += 1;
            if token.contains('\0') {
                return Err(print_error_with_detail(
                    &PRINT_ERROR_INVALID_INPUT,
                    "filename must not contain NUL bytes",
                ));
            }
            let path = Path::new(token);
            if !has_extension(path, "png") && path.file_name().is_none() {
                return Err(print_error_with_detail(
                    &PRINT_ERROR_INVALID_INPUT,
                    "filename must include a file name before appending '.png'",
                ));
            }
            known_filename = Some(path.to_path_buf());
        }
        if filenames > 1 || devices > 1 {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_INPUT,
                "too many filenames or device options",
            ));
        }
    }
    if filenames == 0 && !dynamic_text {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename is required for RunMat figure export",
        ));
    }
    Ok(known_filename)
}

pub(super) fn figure_handle_arg(value: &Value) -> BuiltinResult<Option<FigureHandle>> {
    match value {
        Value::Num(v) => Ok(Some(handle_from_scalar(*v, BUILTIN_NAME)?)),
        Value::Int(i) => Ok(Some(handle_from_integer(i, BUILTIN_NAME)?)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return Ok(Some(handle_from_integer(
                    &storage.value_at(0).expect("one-element integer storage"),
                    BUILTIN_NAME,
                )?));
            }
            Ok(Some(handle_from_scalar(
                tensor_utils::tensor_value_f64(tensor, 0),
                BUILTIN_NAME,
            )?))
        }
        _ => Ok(None),
    }
}

fn parse_option(token: &str, device: &mut Option<PrintDevice>, dpi: &mut u32) -> BuiltinResult<()> {
    match PrintControl::decode(token) {
        Ok(PrintControl::Device(parsed)) => {
            if device.replace(parsed).is_some() {
                return Err(print_error_with_detail(
                    &PRINT_ERROR_INVALID_OPTION,
                    "multiple output devices were specified",
                ));
            }
            Ok(())
        }
        Ok(PrintControl::Resolution(parsed)) => {
            *dpi = parsed;
            Ok(())
        }
        Err(PrintControlError::UnsupportedDevice) => Err(print_error_with_detail(
            &PRINT_ERROR_UNSUPPORTED_DEVICE,
            format!("'{token}' is not available yet; RunMat currently supports '-dpng'"),
        )),
        Err(PrintControlError::InvalidOption) => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_OPTION,
            format!("unsupported option '{token}'"),
        )),
        Err(PrintControlError::ResolutionParse(source)) => Err(print_error_with_source(
            format!(
                "{}: expected '-r<N>' with a nonnegative integer DPI",
                PRINT_ERROR_INVALID_OPTION.message
            ),
            &PRINT_ERROR_INVALID_OPTION,
            source,
        )),
        Err(PrintControlError::UnsupportedResolution(scaled)) => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_OPTION,
            format!(
                "resolution '{token}' would produce an unsupported image dimension ({scaled}px)"
            ),
        )),
    }
}

fn infer_device_from_filename(filename: &str) -> Option<PrintDevice> {
    let ext = Path::new(filename)
        .extension()?
        .to_str()?
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some(PrintDevice::Png),
        _ => None,
    }
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

fn string_arg(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(runmat_value::decode_utf16_code_units(
            ca.data(),
        )
        .map_err(|source| {
            let mut error = print_error_with_detail(&PRINT_ERROR_INVALID_INPUT, source.to_string());
            error.source = Some(Box::new(source));
            error
        })?),
        Value::CharArray(_) => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "character vector arguments must be 1-by-N",
        )),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::StringArray(_) => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "string array arguments must be scalar",
        )),
        other => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            format!("expected string option or filename, got {other:?}"),
        )),
    }
}
