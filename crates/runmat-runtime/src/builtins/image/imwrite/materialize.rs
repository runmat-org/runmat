use super::*;

pub(super) async fn tensor_from_numeric_like(
    value: &Value,
    label: &str,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t.clone()),
        Value::LogicalArray(a) => logical_to_tensor(a, cooperation).await,
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

async fn logical_to_tensor(
    value: &LogicalArray,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<Tensor> {
    cooperation.completed(0).await?;
    let mut data = geometry::reserve::<f64>(value.data.len())?;
    for chunk in value.data.chunks(IMAGE_WORK_CHUNK) {
        data.extend(chunk.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }));
        cooperation.completed(chunk.len()).await?;
    }
    Tensor::new(data, value.shape.clone())
        .map_err(|err| imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, err))
}

pub(super) async fn materialize_image(
    image: &Value,
    map: Option<&Value>,
    alpha: Option<&Tensor>,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<MaterializedImage> {
    ensure_documented_image_class(image, map.is_some(), "image")?;
    if let Some(map) = map {
        ensure_double_colormap(map)?;
    }
    let tensor = tensor_from_numeric_like(image, "image", cooperation).await?;
    let mut out = if let Some(map_value) = map {
        materialize_indexed_image(
            &tensor,
            &tensor_from_numeric_like(map_value, "map", cooperation).await?,
            cooperation,
        )
        .await?
    } else {
        materialize_direct_image(&tensor, cooperation).await?
    };

    if let Some(alpha) = alpha {
        apply_alpha(&mut out, alpha, cooperation).await?;
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
    image_dimensions_from_shape(tensor.shape())
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

async fn materialize_direct_image(
    tensor: &Tensor,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<MaterializedImage> {
    let (rows, cols, channels) = image_dimensions(tensor)?;
    let pixels = geometry::pixels(rows, cols)?;
    let samples = geometry::samples(rows, cols, channels)?;
    if tensor.len() != samples {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "image data length does not match shape",
        ));
    }

    cooperation.completed(0).await?;
    let mut data = if tensor.numeric_dtype() == NumericDType::U16 {
        let mut out = geometry::reserve::<u16>(samples)?;
        cooperation.fill(&mut out, samples).await?;
        PixelData::U16(out)
    } else {
        let mut out = geometry::reserve::<u8>(samples)?;
        cooperation.fill(&mut out, samples).await?;
        PixelData::U8(out)
    };
    let mut pending = 0;
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
                pending += 1;
                if pending == IMAGE_WORK_CHUNK {
                    cooperation.completed(pending).await?;
                    pending = 0;
                }
            }
        }
    }
    cooperation.completed(pending).await?;
    Ok(MaterializedImage {
        rows,
        cols,
        channels,
        data,
        alpha_applied: false,
        indexed_source: false,
    })
}

async fn materialize_indexed_image(
    indexed: &Tensor,
    map: &Tensor,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<MaterializedImage> {
    let (rows, cols, channels) = image_dimensions(indexed)?;
    if channels != 1 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image X must be a 2-D array",
        ));
    }
    let map_shape = map.shape();
    if map_shape.len() != 2 || map_shape[1] != 3 || map_shape[0] == 0 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_COLORMAP,
            "map must be an Nx3 colormap",
        ));
    }
    let pixels = geometry::pixels(rows, cols)?;
    let byte_len = geometry::samples(rows, cols, 3)?;
    let map_len = geometry::samples(map_shape[0], 1, 3)?;
    if indexed.len() != pixels || map.len() != map_len {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image or colormap data length does not match shape",
        ));
    }
    let map_values = tensor::tensor_values_f64_cow(map);
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

    cooperation.completed(0).await?;
    let mut data = geometry::reserve::<u8>(byte_len)?;
    cooperation.fill(&mut data, byte_len).await?;
    let mut pending = 0;
    for row in 0..rows {
        for col in 0..cols {
            let pixel = row + rows * col;
            let map_idx = map_index(
                tensor_numeric_value(indexed, pixel, "indexed image")?,
                map_shape[0],
            )?;
            let dst = (row * cols + col) * 3;
            for channel in 0..3 {
                let src = map_idx + map_shape[0] * channel;
                data[dst + channel] = value_to_u8(tensor_numeric_value(map, src, "colormap")?);
            }
            pending += 3;
            if pending >= IMAGE_WORK_CHUNK {
                cooperation.completed(pending).await?;
                pending = 0;
            }
        }
    }
    cooperation.completed(pending).await?;
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

mod numeric;
pub(super) use numeric::{map_index, value_to_u16, value_to_u8};

async fn apply_alpha(
    image: &mut MaterializedImage,
    alpha: &Tensor,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<()> {
    let alpha_shape = alpha.shape();
    if alpha_shape.len() != 2 || alpha_shape[0] != image.rows || alpha_shape[1] != image.cols {
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
    cooperation.completed(0).await?;
    let mut pending = 0;
    image.data = match &image.data {
        PixelData::U8(data) => {
            let mut rgba = geometry::reserve::<u8>(rgba_samples)?;
            cooperation.fill(&mut rgba, rgba_samples).await?;
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
                    pending += 4;
                    if pending >= IMAGE_WORK_CHUNK {
                        cooperation.completed(pending).await?;
                        pending = 0;
                    }
                }
            }
            PixelData::U8(rgba)
        }
        PixelData::U16(data) => {
            let mut rgba = geometry::reserve::<u16>(rgba_samples)?;
            cooperation.fill(&mut rgba, rgba_samples).await?;
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
                    pending += 4;
                    if pending >= IMAGE_WORK_CHUNK {
                        cooperation.completed(pending).await?;
                        pending = 0;
                    }
                }
            }
            PixelData::U16(rgba)
        }
    };
    cooperation.completed(pending).await?;
    image.channels = 4;
    image.alpha_applied = true;
    Ok(())
}

#[cfg(test)]
#[path = "materialize/cooperation_tests.rs"]
mod cooperation_tests;
