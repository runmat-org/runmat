use super::*;

pub(super) fn preflight_option_text(
    args: &[Value],
    mut index: usize,
    filename: &str,
    indexed_source: bool,
) -> BuiltinResult<ImageFormat> {
    let mut explicit_format = None;
    let mut append = false;
    let mut alpha_applied = false;
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
        if option == Some(ImwriteOption::Alpha) {
            alpha_applied = true;
        }
        if option == Some(ImwriteOption::WriteMode)
            && crate::builtins::image::checked_text::scalar_text(&args[index + 1])
                .and_then(|mode| WriteMode::from_name(&mode))
                == Some(WriteMode::Append)
        {
            append = true;
        }
        index += 2;
    }
    let format = explicit_format.map_or_else(|| infer_format_from_path(Path::new(filename)), Ok)?;
    let (_, _, direct_channels) = match &args[0] {
        Value::GpuTensor(handle) => image_dimensions_from_shape(handle.shape())?,
        Value::Tensor(tensor) => image_dimensions(tensor)?,
        Value::LogicalArray(array) => image_dimensions_from_shape(&array.shape)?,
        _ => (1, 1, 1),
    };
    let sixteen_bit = !indexed_source
        && match &args[0] {
            Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::U16,
            Value::Int(IntValue::U16(_)) => true,
            Value::GpuTensor(handle) => {
                runmat_accelerate_api::handle_integer_type(handle)
                    == Some(runmat_accelerate_api::IntegerElementType::U16)
            }
            _ => false,
        };
    admission::validate(&admission::EncoderFacts {
        format,
        channels: if alpha_applied {
            4
        } else if indexed_source {
            3
        } else {
            direct_channels
        },
        sixteen_bit,
        indexed_source,
        alpha_applied,
        append,
    })?;
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
                    if runmat_accelerate_api::handle_is_logical(handle)
                        || (runmat_accelerate_api::handle_integer_type(handle).is_none()
                            && !matches!(
                                runmat_accelerate_api::handle_precision(handle),
                                Some(
                                    runmat_accelerate_api::ProviderPrecision::F32
                                        | runmat_accelerate_api::ProviderPrecision::F64
                                )
                            ))
                    {
                        return Err(imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_OPTION,
                            "Alpha class must be double, single, uint8, or uint16",
                        ));
                    }
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
            if shape.len() != 2 || shape[0] != image_shape.0 || shape[1] != image_shape.1 {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    "Alpha must be an MxN array matching the image dimensions",
                ));
            }
        }
        ImwriteOption::Quality | ImwriteOption::DelayTime | ImwriteOption::LoopCount => {
            let Value::GpuTensor(handle) = value else {
                let number = numeric_scalar(value, "writer control")?;
                options::validate_numeric(option, number)?;
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
            options::parse_write_mode(&mode)?;
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
                if !crate::builtins::common::tensor::tensor_values_f64_cow(tensor)
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                {
                    return Err(imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_COLORMAP,
                        "map values must be finite and in the range [0, 1]",
                    ));
                }
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
        // Host X values and the resident map's row count are both known now.
        // Use the same conversion/range rule as materialization before any map readback.
        if !matches!(image, Value::GpuTensor(_)) {
            let owned;
            let indexed = match image {
                Value::Tensor(tensor) => tensor,
                _ => {
                    owned = tensor_from_numeric_like(image, "image")?;
                    &owned
                }
            };
            for index in 0..indexed.len() {
                let value = indexed.numeric_value_at(index).ok_or_else(|| {
                    imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_IMAGE,
                        format!("indexed image storage is unavailable at element {index}"),
                    )
                })?;
                super::materialize::map_index(value, shape[0])?;
            }
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
