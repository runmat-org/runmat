use super::*;

pub(super) fn preflight_option_text(
    args: &[Value],
    mut index: usize,
    filename: &str,
) -> BuiltinResult<ImageFormat> {
    let mut explicit_format = None;
    let mut append = false;
    if let Some(first) = args.get(index) {
        let text = string_arg(
            "format or option name",
            first,
            &IMWRITE_ERROR_INVALID_OPTION,
        )?;
        if ImwriteOption::from_name(&text).is_none() {
            explicit_format = Some(parse_format_hint(&text)?);
            index += 1;
        }
    }
    while index < args.len() {
        let name = string_arg("option name", &args[index], &IMWRITE_ERROR_INVALID_OPTION)?;
        let option = ImwriteOption::from_name(&name);
        match option {
            None => {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    format!("unsupported option '{name}'"),
                ));
            }
            Some(ImwriteOption::UnsupportedAdvanced) => {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    format!("option '{name}' is not supported yet"),
                ));
            }
            Some(_) => {}
        }
        if index + 1 >= args.len() {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_OPTION,
                format!("option '{name}' requires a value"),
            ));
        }
        preflight_known_option_value(option.expect("validated above"), &args[index + 1], &args[0])?;
        if option == Some(ImwriteOption::WriteMode)
            && super::checked_text::scalar_text(&args[index + 1])
                .is_some_and(|mode| mode.trim().eq_ignore_ascii_case("append"))
        {
            append = true;
        }
        index += 2;
    }
    let format = explicit_format.map_or_else(|| infer_format_from_path(Path::new(filename)), Ok)?;
    if append && format != ImageFormat::Gif {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "WriteMode append is supported only for GIF",
        ));
    }
    Ok(format)
}

pub(super) fn image_is_single(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    }
}

fn preflight_known_option_value(
    option: ImwriteOption,
    value: &Value,
    image: &Value,
) -> BuiltinResult<()> {
    match option {
        ImwriteOption::Alpha => {
            let shape = match value {
                Value::GpuTensor(handle) => {
                    if runmat_accelerate_api::handle_integer_type(handle).is_some_and(|integer| {
                        !matches!(
                            integer,
                            runmat_accelerate_api::IntegerElementType::U8
                                | runmat_accelerate_api::IntegerElementType::U16
                        )
                    }) {
                        return Err(imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_OPTION,
                            "Alpha class must be double, single, uint8, or uint16",
                        ));
                    }
                    handle.shape().to_vec()
                }
                _ => {
                    let alpha = tensor_from_numeric_like(value, "Alpha")?;
                    ensure_documented_alpha_class(&alpha)?;
                    alpha.shape
                }
            };
            let image_shape = match image {
                Value::GpuTensor(handle) => image_dimensions_from_shape(handle.shape())?,
                Value::Tensor(tensor) => image_dimensions(tensor)?,
                Value::LogicalArray(array) => image_dimensions_from_shape(&array.shape)?,
                _ => (1, 1, 1),
            };
            if shape.as_slice() != [image_shape.0, image_shape.1] {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    "Alpha must be an MxN array matching the image dimensions",
                ));
            }
        }
        ImwriteOption::Quality | ImwriteOption::DelayTime | ImwriteOption::LoopCount => {
            let Value::GpuTensor(handle) = value else {
                let number = numeric_scalar(value, "writer control")?;
                match option {
                    ImwriteOption::Quality
                        if !number.is_finite() || !(0.0..=100.0).contains(&number) =>
                    {
                        return Err(imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_OPTION,
                            "Quality must be a scalar from 0 to 100",
                        ));
                    }
                    ImwriteOption::DelayTime if !number.is_finite() || number < 0.0 => {
                        return Err(imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_OPTION,
                            "DelayTime must be a finite non-negative scalar in seconds",
                        ));
                    }
                    ImwriteOption::LoopCount if number.is_nan() || number < 0.0 => {
                        return Err(imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_OPTION,
                            "LoopCount must be non-negative or Inf",
                        ));
                    }
                    _ => {}
                }
                return Ok(());
            };
            if handle
                .shape()
                .iter()
                .try_fold(1usize, |size, dim| size.checked_mul(*dim))
                != Some(1)
            {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    "writer control must be a numeric scalar",
                ));
            }
        }
        ImwriteOption::WriteMode => {
            let mode = string_arg("WriteMode", value, &IMWRITE_ERROR_INVALID_OPTION)?;
            if !matches!(
                mode.trim().to_ascii_lowercase().as_str(),
                "overwrite" | "append"
            ) {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    "WriteMode must be 'overwrite' or 'append'",
                ));
            }
        }
        ImwriteOption::UnsupportedAdvanced => unreachable!("rejected above"),
    }
    Ok(())
}

pub(super) fn preflight_resident_argument_roles(args: &[Value]) -> BuiltinResult<()> {
    for value in args {
        if let Value::GpuTensor(handle) = value {
            validate_resident_numeric_metadata(handle, &IMWRITE_ERROR_INVALID_ARGUMENT)?;
        }
    }
    let Some(image) = args.first() else {
        return Ok(());
    };
    preflight_resident_image_class(image)?;
    if args.get(1).is_some_and(|value| !is_string_like(value)) {
        let map = &args[1];
        let shape = match map {
            Value::GpuTensor(handle) => {
                let owner =
                    validate_resident_numeric_metadata(handle, &IMWRITE_ERROR_INVALID_COLORMAP)?;
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
                    || runmat_accelerate_api::handle_is_logical(handle)
                    || runmat_accelerate_api::handle_precision(handle)
                        != Some(runmat_accelerate_api::ProviderPrecision::F64)
                    || owner.precision() != runmat_accelerate_api::ProviderPrecision::F64
                {
                    return Err(imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_COLORMAP,
                        "map must be a double Nx3 colormap",
                    ));
                }
                handle.shape()
            }
            Value::Tensor(tensor) => {
                ensure_double_colormap(map)?;
                tensor.shape.as_slice()
            }
            _ => {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_COLORMAP,
                    "map must be a double Nx3 colormap",
                ));
            }
        };
        if shape.len() != 2 || shape[0] == 0 || shape[1] != 3 {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_COLORMAP,
                "map must be an Nx3 colormap",
            ));
        }
    }
    Ok(())
}

fn validate_resident_numeric_metadata(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<&'static dyn runmat_accelerate_api::AccelProvider> {
    let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| {
            imwrite_error_with_detail(error, "no acceleration provider owns the gpuArray handle")
        })?;
    if runmat_accelerate_api::handle_storage(handle)
        != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Err(imwrite_error_with_detail(
            error,
            "gpuArray arguments must use real numeric storage",
        ));
    }
    let precision = runmat_accelerate_api::handle_precision(handle);
    let integer = runmat_accelerate_api::handle_integer_type(handle);
    let logical = runmat_accelerate_api::handle_is_logical(handle);
    if !crate::builtins::common::gpu_helpers::gpu_class_metadata_matches(
        handle, precision, integer, logical,
    ) {
        return Err(imwrite_error_with_detail(
            error,
            "gpuArray class metadata contradicts its physical storage",
        ));
    }
    if integer.is_none() && precision != Some(owner.precision()) {
        return Err(imwrite_error_with_detail(
            error,
            "gpuArray precision metadata contradicts its owning provider",
        ));
    }
    Ok(owner)
}

pub(super) async fn gather_imwrite_argument(value: &Value) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = value else {
        return Ok(value.clone());
    };
    let owner = validate_resident_numeric_metadata(handle, &IMWRITE_ERROR_INVALID_ARGUMENT)?;
    let _residency = runmat_accelerate_api::ResidencyRestoreGuard::new(handle);
    let result = crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
        owner, handle,
    )
    .await;
    result.map_err(|err| {
        imwrite_error_with_source(
            &IMWRITE_ERROR_INVALID_ARGUMENT,
            "provider download failed",
            err,
        )
    })
}

fn preflight_resident_image_class(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::GpuTensor(handle) => {
            validate_resident_numeric_metadata(handle, &IMWRITE_ERROR_INVALID_IMAGE)?;
            if matches!(
                runmat_accelerate_api::handle_integer_type(handle),
                Some(
                    runmat_accelerate_api::IntegerElementType::I8
                        | runmat_accelerate_api::IntegerElementType::I16
                        | runmat_accelerate_api::IntegerElementType::I32
                        | runmat_accelerate_api::IntegerElementType::I64
                        | runmat_accelerate_api::IntegerElementType::U32
                        | runmat_accelerate_api::IntegerElementType::U64
                )
            ) {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_IMAGE,
                    "supported integer image classes are uint8 and uint16",
                ));
            }
            image_dimensions_from_shape(handle.shape())?;
        }
        Value::Tensor(tensor) => {
            ensure_documented_image_class(value, false, "image")?;
            image_dimensions(tensor)?;
        }
        Value::LogicalArray(array) => {
            image_dimensions_from_shape(&array.shape)?;
        }
        _ => ensure_documented_image_class(value, false, "image")?,
    }
    Ok(())
}
