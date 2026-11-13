use super::*;

pub(super) fn preflight_option_text(
    args: crate::call::FunctionArgumentView<'_>,
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
        if option == Some(ImwriteOption::WriteMode) {
            append = crate::builtins::image::checked_text::scalar_text(&args[index + 1])
                .map_err(|source| {
                    let mut error = imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_OPTION,
                        source.to_string(),
                    );
                    error.source = Some(Box::new(source));
                    error
                })?
                .and_then(|mode| WriteMode::from_name(&mode))
                == Some(WriteMode::Append);
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
                Value::Tensor(alpha) => {
                    ensure_documented_alpha_class(alpha)?;
                    alpha.shape().to_vec()
                }
                Value::LogicalArray(alpha) => alpha.shape.to_vec(),
                Value::Num(_) | Value::Bool(_) | Value::Int(IntValue::U8(_) | IntValue::U16(_)) => {
                    vec![1, 1]
                }
                Value::Int(_) => {
                    return Err(imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_OPTION,
                        "Alpha class must be double, single, uint8, or uint16",
                    ));
                }
                _ => {
                    return Err(imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_IMAGE,
                        "Alpha must be numeric or logical",
                    ));
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

/// Metadata-only validation retains class, ownership and geometry precedence
/// without scanning host pixels before option syntax/admission is known.
pub(super) fn preflight_resident_argument_roles(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
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
                // Shared metadata validation already proves that floating
                // precision agrees with the exact source owner's precision.
                validate_resident_numeric_metadata(handle, &IMWRITE_ERROR_INVALID_COLORMAP)?;
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
                    || runmat_accelerate_api::handle_is_logical(handle)
                    || runmat_accelerate_api::handle_precision(handle)
                        != Some(runmat_accelerate_api::ProviderPrecision::F64)
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
                tensor.shape()
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

/// Value-level scans run only after the complete known option tail has passed
/// admission. They still precede any provider readback or filesystem effect.
pub(super) async fn preflight_known_host_pixels(
    args: crate::call::FunctionArgumentView<'_>,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<()> {
    let Some(image) = args.first() else {
        return Ok(());
    };
    if args.get(1).is_some_and(|value| !is_string_like(value)) {
        let map = &args[1];
        let shape = match map {
            Value::GpuTensor(handle) => handle.shape(),
            Value::Tensor(tensor) => {
                let map_values = crate::builtins::common::tensor::tensor_values_f64_cow(tensor);
                for chunk in map_values.chunks(IMAGE_WORK_CHUNK) {
                    if !chunk
                        .iter()
                        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                    {
                        return Err(imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_COLORMAP,
                            "map values must be finite and in the range [0, 1]",
                        ));
                    }
                    cooperation.completed(chunk.len()).await?;
                }
                tensor.shape()
            }
            _ => unreachable!("metadata admission validated map class"),
        };
        // Host X values and the resident map's row count are both known now.
        // Use the same conversion/range rule as materialization before any map readback.
        if !matches!(image, Value::GpuTensor(_)) {
            match image {
                Value::Tensor(indexed) => {
                    let mut pending = 0;
                    for index in 0..indexed.len() {
                        let value = indexed.numeric_value_at(index).ok_or_else(|| {
                            imwrite_error_with_detail(
                                &IMWRITE_ERROR_INVALID_IMAGE,
                                format!("indexed image storage is unavailable at element {index}"),
                            )
                        })?;
                        super::materialize::map_index(value, shape[0])?;
                        pending += 1;
                        if pending == IMAGE_WORK_CHUNK {
                            cooperation.completed(pending).await?;
                            pending = 0;
                        }
                    }
                    cooperation.completed(pending).await?;
                }
                Value::LogicalArray(indexed) => {
                    let mut pending = 0;
                    for &value in &indexed.data {
                        super::materialize::map_index(
                            NumericScalar::F64(if value == 0 { 0.0 } else { 1.0 }),
                            shape[0],
                        )?;
                        pending += 1;
                        if pending == IMAGE_WORK_CHUNK {
                            cooperation.completed(pending).await?;
                            pending = 0;
                        }
                    }
                    cooperation.completed(pending).await?;
                }
                Value::Num(value) => {
                    super::materialize::map_index(NumericScalar::F64(*value), shape[0])?;
                }
                Value::Bool(value) => {
                    super::materialize::map_index(
                        NumericScalar::F64(if *value { 1.0 } else { 0.0 }),
                        shape[0],
                    )?;
                }
                Value::Int(value) => {
                    let indexed =
                        Tensor::new_integer(IntegerStorage::from_scalar(value.clone()), vec![1, 1])
                            .map_err(|err| {
                                imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, err)
                            })?;
                    super::materialize::map_index(
                        indexed.numeric_value_at(0).expect("scalar storage"),
                        shape[0],
                    )?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn validate_resident_numeric_metadata<'h>(
    handle: &'h runmat_accelerate_api::GpuTensorHandle,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<runmat_accelerate_api::NumericReadback<'h>> {
    let readback = runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|_| {
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
    let owner_precision = readback.owner_precision().map_err(|source| {
        imwrite_error_with_source(
            error,
            "resident owner metadata is unavailable",
            runmat_accelerate_api::ProviderExecutionError::from(source),
        )
    })?;
    if integer.is_none() && precision != Some(owner_precision) {
        return Err(imwrite_error_with_detail(
            error,
            "gpuArray precision metadata contradicts its owning provider",
        ));
    }
    Ok(readback)
}

pub(super) async fn gather_imwrite_argument(value: &Value) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = value else {
        return Ok(value.clone());
    };
    validate_resident_numeric_metadata(handle, &IMWRITE_ERROR_INVALID_ARGUMENT)?;
    let _residency = runmat_accelerate_api::ResidencyRestoreGuard::new(handle);
    let result = crate::builtins::common::gpu_helpers::gather_value_async(value).await;
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
