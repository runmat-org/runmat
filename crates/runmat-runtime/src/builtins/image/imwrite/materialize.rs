use super::*;

pub(super) fn tensor_from_numeric_like(value: &Value, label: &str) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t.clone()),
        Value::LogicalArray(a) => logical_to_tensor(a),
        Value::Num(n) => Tensor::new(vec![*n], vec![1, 1]).map_err(|err| {
            imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, format!("{label}: {err}"))
        }),
        Value::Int(i) => Tensor::new_integer(IntegerStorage::from_scalar(i.clone()), vec![1, 1])
            .map_err(|err| {
                imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, format!("{label}: {err}"))
            }),
        Value::Bool(b) => {
            Tensor::new(vec![if *b { 1.0 } else { 0.0 }], vec![1, 1]).map_err(|err| {
                imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, format!("{label}: {err}"))
            })
        }
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("{label} must be numeric or logical"),
        )),
    }
}

fn logical_to_tensor(value: &LogicalArray) -> BuiltinResult<Tensor> {
    let mut data = geometry::reserve::<f64>(value.data.len())?;
    data.extend(value.data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }));
    Tensor::new(data, value.shape.clone())
        .map_err(|err| imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, err))
}

pub(super) fn materialize_image(
    image: &Value,
    map: Option<&Value>,
    alpha: Option<&Tensor>,
) -> BuiltinResult<MaterializedImage> {
    ensure_documented_image_class(image, map.is_some(), "image")?;
    if let Some(map) = map {
        ensure_double_colormap(map)?;
    }
    let tensor = tensor_from_numeric_like(image, "image")?;
    let mut out = if let Some(map_value) = map {
        materialize_indexed_image(&tensor, &tensor_from_numeric_like(map_value, "map")?)?
    } else {
        materialize_direct_image(&tensor)?
    };

    if let Some(alpha) = alpha {
        apply_alpha(&mut out, alpha)?;
    }
    Ok(out)
}

pub(super) fn ensure_documented_image_class(
    value: &Value,
    indexed: bool,
    label: &str,
) -> BuiltinResult<()> {
    let supported = match value {
        Value::Num(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Int(IntValue::U8(_) | IntValue::U16(_)) => true,
        Value::Tensor(tensor) => matches!(
            tensor.numeric_dtype(),
            NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
        ),
        _ => false,
    };
    if !supported {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!(
                "{label} class is unsupported; expected double, single, uint8, uint16, or logical{}",
                if indexed { " indexed data" } else { "" }
            ),
        ));
    }
    Ok(())
}

pub(super) fn ensure_double_colormap(value: &Value) -> BuiltinResult<()> {
    let valid = matches!(value, Value::Num(_))
        || matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64);
    if valid {
        Ok(())
    } else {
        Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_COLORMAP,
            "map must be a double Nx3 colormap",
        ))
    }
}

pub(super) fn ensure_documented_alpha_class(alpha: &Tensor) -> BuiltinResult<()> {
    if matches!(
        alpha.numeric_dtype(),
        NumericDType::F32 | NumericDType::F64 | NumericDType::U8 | NumericDType::U16
    ) {
        Ok(())
    } else {
        Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "Alpha class must be double, single, uint8, or uint16",
        ))
    }
}

pub(super) fn image_dimensions(tensor: &Tensor) -> BuiltinResult<(usize, usize, usize)> {
    image_dimensions_from_shape(&tensor.shape)
}

pub(super) fn image_dimensions_from_shape(shape: &[usize]) -> BuiltinResult<(usize, usize, usize)> {
    match shape.len() {
        0 => Ok((1, 1, 1)),
        1 => Ok((1, shape[0], 1)),
        2 => Ok((shape[0], shape[1], 1)),
        3 if matches!(shape[2], 1 | 3 | 4) => Ok((shape[0], shape[1], shape[2])),
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "image must be MxN, MxNx3, or MxNx4",
        )),
    }
}

fn materialize_direct_image(tensor: &Tensor) -> BuiltinResult<MaterializedImage> {
    let (rows, cols, channels) = image_dimensions(tensor)?;
    let pixels = geometry::pixels(rows, cols)?;
    let samples = geometry::samples(rows, cols, channels)?;
    if tensor.len() != samples {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "image data length does not match shape",
        ));
    }

    let mut data = if tensor.numeric_dtype() == NumericDType::U16 {
        PixelData::U16(geometry::buffer(samples)?)
    } else {
        PixelData::U8(geometry::buffer(samples)?)
    };
    for row in 0..rows {
        for col in 0..cols {
            for channel in 0..channels {
                let src = row + rows * col + pixels * channel;
                let dst = (row * cols + col) * channels + channel;
                let value = tensor_numeric_value(tensor, src, "image")?;
                match &mut data {
                    PixelData::U8(data) => data[dst] = value_to_u8(value),
                    PixelData::U16(data) => data[dst] = value_to_u16(value),
                }
            }
        }
    }
    Ok(MaterializedImage {
        rows,
        cols,
        channels,
        data,
        alpha_applied: false,
        indexed_source: false,
    })
}

fn materialize_indexed_image(indexed: &Tensor, map: &Tensor) -> BuiltinResult<MaterializedImage> {
    let (rows, cols, channels) = image_dimensions(indexed)?;
    if channels != 1 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image X must be a 2-D array",
        ));
    }
    if map.shape.len() != 2 || map.shape[1] != 3 || map.shape[0] == 0 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_COLORMAP,
            "map must be an Nx3 colormap",
        ));
    }
    let pixels = geometry::pixels(rows, cols)?;
    let byte_len = geometry::samples(rows, cols, 3)?;
    let map_len = geometry::samples(map.shape[0], 1, 3)?;
    if indexed.len() != pixels || map.len() != map_len {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image or colormap data length does not match shape",
        ));
    }
    let map_values = tensor::tensor_values_f64_cow(map);
    if !map_values
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_COLORMAP,
            "map values must be finite and in the range [0, 1]",
        ));
    }

    let mut data = geometry::buffer::<u8>(byte_len)?;
    for row in 0..rows {
        for col in 0..cols {
            let pixel = row + rows * col;
            let map_idx = map_index(
                tensor_numeric_value(indexed, pixel, "indexed image")?,
                map.shape[0],
            )?;
            let dst = (row * cols + col) * 3;
            for channel in 0..3 {
                let src = map_idx + map.shape[0] * channel;
                data[dst + channel] = value_to_u8(tensor_numeric_value(map, src, "colormap")?);
            }
        }
    }
    Ok(MaterializedImage {
        rows,
        cols,
        channels: 3,
        data: PixelData::U8(data),
        alpha_applied: false,
        indexed_source: true,
    })
}

fn tensor_numeric_value(
    tensor: &Tensor,
    index: usize,
    label: &str,
) -> BuiltinResult<NumericScalar> {
    tensor.numeric_value_at(index).ok_or_else(|| {
        imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!(
                "{label} {} storage is unavailable at element {index}",
                tensor.numeric_dtype().class_name()
            ),
        )
    })
}

pub(super) fn map_index(value: NumericScalar, map_rows: usize) -> BuiltinResult<usize> {
    let index = match value {
        NumericScalar::F64(value) => floating_map_index(value)?,
        NumericScalar::F32(value) => floating_map_index(f64::from(value))?,
        NumericScalar::U8(value) => usize::from(value),
        NumericScalar::U16(value) => usize::from(value),
        NumericScalar::I8(value) => one_based_signed_index(i128::from(value))?,
        NumericScalar::I16(value) => one_based_signed_index(i128::from(value))?,
        NumericScalar::I32(value) => one_based_signed_index(i128::from(value))?,
        NumericScalar::I64(value) => one_based_signed_index(i128::from(value))?,
        NumericScalar::U32(value) => one_based_unsigned_index(u128::from(value))?,
        NumericScalar::U64(value) => one_based_unsigned_index(u128::from(value))?,
    };
    if index >= map_rows {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!(
                "indexed image value {} is outside the colormap",
                numeric_scalar_text(value)
            ),
        ));
    }
    Ok(index)
}

fn floating_map_index(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image values must be finite",
        ));
    }
    let rounded = value.round();
    if rounded < 1.0 || rounded > usize::MAX as f64 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("indexed image value {value} is outside the colormap"),
        ));
    }
    Ok(rounded as usize - 1)
}

fn one_based_signed_index(value: i128) -> BuiltinResult<usize> {
    if value < 1 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("indexed image value {value} is outside the colormap"),
        ));
    }
    usize::try_from(value - 1).map_err(|_| {
        imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("indexed image value {value} is outside the colormap"),
        )
    })
}

fn one_based_unsigned_index(value: u128) -> BuiltinResult<usize> {
    if value == 0 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image value 0 is outside the colormap",
        ));
    }
    usize::try_from(value - 1).map_err(|_| {
        imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("indexed image value {value} is outside the colormap"),
        )
    })
}

pub(super) fn value_to_u8(value: NumericScalar) -> u8 {
    match value {
        NumericScalar::F64(value) => normalized_float_to_u8(value),
        NumericScalar::F32(value) => normalized_float_to_u8(f64::from(value)),
        integer => scaled_integer(integer, u8::MAX as u128) as u8,
    }
}

pub(super) fn value_to_u16(value: NumericScalar) -> u16 {
    match value {
        NumericScalar::F64(value) => normalized_float_to_u16(value),
        NumericScalar::F32(value) => normalized_float_to_u16(f64::from(value)),
        integer => scaled_integer(integer, u16::MAX as u128) as u16,
    }
}

fn normalized_float_to_u8(value: f64) -> u8 {
    let scaled = value.clamp(0.0, 1.0) * u8::MAX as f64;
    if scaled.is_nan() {
        0
    } else {
        scaled.round() as u8
    }
}

fn normalized_float_to_u16(value: f64) -> u16 {
    let scaled = value.clamp(0.0, 1.0) * u16::MAX as f64;
    if scaled.is_nan() {
        0
    } else {
        scaled.round() as u16
    }
}

fn scaled_integer(value: NumericScalar, output_max: u128) -> u128 {
    let (offset, input_max) = match value {
        NumericScalar::I8(value) => (
            (i128::from(value) - i128::from(i8::MIN)) as u128,
            u128::from(u8::MAX),
        ),
        NumericScalar::I16(value) => (
            (i128::from(value) - i128::from(i16::MIN)) as u128,
            u128::from(u16::MAX),
        ),
        NumericScalar::I32(value) => (
            (i128::from(value) - i128::from(i32::MIN)) as u128,
            u128::from(u32::MAX),
        ),
        NumericScalar::I64(value) => (
            (i128::from(value) - i128::from(i64::MIN)) as u128,
            u128::from(u64::MAX),
        ),
        NumericScalar::U8(value) => (u128::from(value), u128::from(u8::MAX)),
        NumericScalar::U16(value) => (u128::from(value), u128::from(u16::MAX)),
        NumericScalar::U32(value) => (u128::from(value), u128::from(u32::MAX)),
        NumericScalar::U64(value) => (u128::from(value), u128::from(u64::MAX)),
        NumericScalar::F64(_) | NumericScalar::F32(_) => {
            unreachable!("floating image samples use normalized conversion")
        }
    };
    (offset * output_max + input_max / 2) / input_max
}

fn numeric_scalar_text(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => value.to_string(),
        NumericScalar::F32(value) => value.to_string(),
        NumericScalar::I8(value) => value.to_string(),
        NumericScalar::I16(value) => value.to_string(),
        NumericScalar::I32(value) => value.to_string(),
        NumericScalar::I64(value) => value.to_string(),
        NumericScalar::U8(value) => value.to_string(),
        NumericScalar::U16(value) => value.to_string(),
        NumericScalar::U32(value) => value.to_string(),
        NumericScalar::U64(value) => value.to_string(),
    }
}

fn apply_alpha(image: &mut MaterializedImage, alpha: &Tensor) -> BuiltinResult<()> {
    if alpha.shape.len() != 2 || alpha.shape[0] != image.rows || alpha.shape[1] != image.cols {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "Alpha must be an MxN array matching the image dimensions",
        ));
    }
    let pixels = geometry::pixels(image.rows, image.cols)?;
    if alpha.len() != pixels {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "Alpha data length does not match shape",
        ));
    }

    let rgba_samples = geometry::samples(image.rows, image.cols, 4)?;
    image.data = match &image.data {
        PixelData::U8(data) => {
            let mut rgba = geometry::buffer::<u8>(rgba_samples)?;
            for row in 0..image.rows {
                for col in 0..image.cols {
                    let pixel = row * image.cols + col;
                    let alpha_idx = row + image.rows * col;
                    let dst = pixel * 4;
                    match image.channels {
                        1 => {
                            let gray = data[pixel];
                            rgba[dst] = gray;
                            rgba[dst + 1] = gray;
                            rgba[dst + 2] = gray;
                        }
                        3 | 4 => {
                            let src = pixel * image.channels;
                            rgba[dst] = data[src];
                            rgba[dst + 1] = data[src + 1];
                            rgba[dst + 2] = data[src + 2];
                        }
                        _ => unreachable!(),
                    }
                    rgba[dst + 3] = value_to_u8(tensor_numeric_value(alpha, alpha_idx, "Alpha")?);
                }
            }
            PixelData::U8(rgba)
        }
        PixelData::U16(data) => {
            let mut rgba = geometry::buffer::<u16>(rgba_samples)?;
            for row in 0..image.rows {
                for col in 0..image.cols {
                    let pixel = row * image.cols + col;
                    let alpha_idx = row + image.rows * col;
                    let dst = pixel * 4;
                    match image.channels {
                        1 => {
                            let gray = data[pixel];
                            rgba[dst] = gray;
                            rgba[dst + 1] = gray;
                            rgba[dst + 2] = gray;
                        }
                        3 | 4 => {
                            let src = pixel * image.channels;
                            rgba[dst] = data[src];
                            rgba[dst + 1] = data[src + 1];
                            rgba[dst + 2] = data[src + 2];
                        }
                        _ => unreachable!(),
                    }
                    rgba[dst + 3] = value_to_u16(tensor_numeric_value(alpha, alpha_idx, "Alpha")?);
                }
            }
            PixelData::U16(rgba)
        }
    };
    image.channels = 4;
    image.alpha_applied = true;
    Ok(())
}
