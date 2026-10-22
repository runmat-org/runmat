use super::*;

pub(super) async fn parse_invocation(
    args: &[Value],
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<Invocation> {
    if args.len() < 2 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_ARGUMENT,
            "expected image data and filename",
        ));
    }

    let (image, map, filename_index) = if is_string_like(&args[1]) {
        (args[0].clone(), None, 1usize)
    } else {
        if args.len() < 3 {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_ARGUMENT,
                "indexed images require X, map, and filename",
            ));
        }
        (args[0].clone(), Some(args[1].clone()), 2usize)
    };

    let filename = string_arg(
        "filename",
        &args[filename_index],
        &IMWRITE_ERROR_INVALID_FILENAME,
    )?;
    if filename.trim().is_empty() {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_FILENAME,
            "filename must not be empty",
        ));
    }
    let path = PathBuf::from(filename);
    let mut idx = filename_index + 1;

    let mut explicit_format = None;
    if idx < args.len() {
        if let Some(text) =
            crate::builtins::image::checked_text::scalar_text(&args[idx]).map_err(|source| {
                let mut error =
                    imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_OPTION, source.to_string());
                error.source = Some(Box::new(source));
                error
            })?
        {
            if ImwriteOption::from_name(&text).is_none() {
                explicit_format = Some(parse_format_hint(&text)?);
                idx += 1;
            }
        }
    }

    let mut options = ImwriteOptions::default();
    let mut alpha = None;
    while idx < args.len() {
        let name = string_arg("option name", &args[idx], &IMWRITE_ERROR_INVALID_OPTION)?;
        idx += 1;
        if idx >= args.len() {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_OPTION,
                format!("option '{name}' requires a value"),
            ));
        }
        let value = &args[idx];
        idx += 1;

        match ImwriteOption::from_name(&name) {
            Some(ImwriteOption::Alpha) => {
                let alpha_tensor = tensor_from_numeric_like(value, "Alpha", cooperation).await?;
                ensure_documented_alpha_class(&alpha_tensor)?;
                alpha = Some(alpha_tensor);
            }
            Some(ImwriteOption::Quality) => {
                let q = numeric_scalar(value, "Quality")?;
                options::validate_numeric(ImwriteOption::Quality, q)?;
                options.quality = q.round() as u8;
            }
            Some(ImwriteOption::WriteMode) => {
                let mode = string_arg("WriteMode", value, &IMWRITE_ERROR_INVALID_OPTION)?;
                options.write_mode = options::parse_write_mode(&mode)?;
            }
            Some(ImwriteOption::DelayTime) => {
                let delay = numeric_scalar(value, "DelayTime")?;
                options::validate_numeric(ImwriteOption::DelayTime, delay)?;
                options.delay_time = Some(delay);
            }
            Some(ImwriteOption::LoopCount) => {
                let count = numeric_scalar(value, "LoopCount")?;
                options::validate_numeric(ImwriteOption::LoopCount, count)?;
                options.loop_count = Some(count);
            }
            Some(ImwriteOption::UnsupportedAdvanced) => {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    format!("option '{name}' is not supported yet"),
                ));
            }
            None => {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    format!("unsupported option '{name}'"),
                ))
            }
        }
    }

    let format = match explicit_format {
        Some(format) => format,
        None => infer_format_from_path(&path)?,
    };

    Ok(Invocation {
        image,
        map,
        alpha,
        path,
        format,
        options,
    })
}

pub(super) fn is_string_like(value: &Value) -> bool {
    matches!(value, Value::String(_))
        || matches!(value, Value::StringArray(a) if a.data().len()==1)
        || matches!(value, Value::CharArray(a) if a.rows==1)
}

pub(super) fn string_arg(
    label: &str,
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    crate::builtins::image::checked_text::scalar_text(value)
        .map_err(|source| {
            let mut error = imwrite_error_with_detail(error, source.to_string());
            error.source = Some(Box::new(source));
            error
        })?
        .ok_or_else(|| {
            imwrite_error_with_detail(
                error,
                format!("{label} must be a string scalar or char vector"),
            )
        })
}

pub(super) fn numeric_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    let scalar = match value {
        Value::Num(n) => return Ok(*n),
        Value::Int(i) => NumericScalar::from(i.clone()),
        Value::Bool(b) => return Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            t.numeric_value_at(0).ok_or_else(|| {
                imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    format!("{label} scalar storage is unavailable"),
                )
            })?
        }
        Value::LogicalArray(a) if a.data.len() == 1 => {
            return Ok(if a.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            format!("{label} must be a numeric scalar"),
        ))?,
    };
    numeric_scalar_to_exact_f64(scalar, label)
}

fn numeric_scalar_to_exact_f64(value: NumericScalar, label: &str) -> BuiltinResult<f64> {
    let converted = match value {
        NumericScalar::F64(value) => return Ok(value),
        NumericScalar::F32(value) => return Ok(f64::from(value)),
        NumericScalar::I8(value) => f64::from(value),
        NumericScalar::I16(value) => f64::from(value),
        NumericScalar::I32(value) => f64::from(value),
        NumericScalar::I64(value)
            if crate::builtins::common::validation::integer_is_exact_f64(
                &runmat_value::IntValue::I64(value),
            ) =>
        {
            value as f64
        }
        NumericScalar::U8(value) => f64::from(value),
        NumericScalar::U16(value) => f64::from(value),
        NumericScalar::U32(value) => f64::from(value),
        NumericScalar::U64(value)
            if crate::builtins::common::validation::integer_is_exact_f64(
                &runmat_value::IntValue::U64(value),
            ) =>
        {
            value as f64
        }
        NumericScalar::I64(_) | NumericScalar::U64(_) => {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_OPTION,
                format!("{label} integer value must be exactly representable as double"),
            ))
        }
    };
    Ok(converted)
}

pub(super) fn parse_format_hint(value: &str) -> BuiltinResult<ImageFormat> {
    let label = value.trim().trim_start_matches('.').to_ascii_lowercase();
    if label.is_empty() {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_FORMAT,
            "format hint must not be empty",
        ));
    }
    if let Some(format) = runmat_builtins::ImwriteFormat::from_hint(&label) {
        return Ok(match format {
            runmat_builtins::ImwriteFormat::Png => ImageFormat::Png,
            runmat_builtins::ImwriteFormat::Jpeg => ImageFormat::Jpeg,
            runmat_builtins::ImwriteFormat::Bmp => ImageFormat::Bmp,
            runmat_builtins::ImwriteFormat::Gif => ImageFormat::Gif,
            runmat_builtins::ImwriteFormat::Tiff => ImageFormat::Tiff,
        });
    }
    ImageFormat::from_extension(&label)
        .filter(is_supported_format)
        .ok_or_else(|| {
            imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_FORMAT,
                format!("unsupported image format '{label}'"),
            )
        })
}

pub(super) fn infer_format_from_path(path: &Path) -> BuiltinResult<ImageFormat> {
    ImageFormat::from_path(path)
        .ok()
        .filter(is_supported_format)
        .ok_or_else(|| {
            imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_FORMAT,
                format!(
                    "could not infer supported image format from '{}'",
                    path.display()
                ),
            )
        })
}

fn is_supported_format(format: &ImageFormat) -> bool {
    matches!(
        format,
        ImageFormat::Png
            | ImageFormat::Jpeg
            | ImageFormat::Bmp
            | ImageFormat::Gif
            | ImageFormat::Tiff
    )
}
