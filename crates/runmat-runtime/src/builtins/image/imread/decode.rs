use super::*;

pub(super) fn decode_image_bytes(
    bytes: &[u8],
    format: Option<ImageFormat>,
) -> BuiltinResult<DynamicImage> {
    let reader = if let Some(format) = format {
        ImageReader::with_format(Cursor::new(bytes), format)
    } else {
        ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|err| {
                imread_error_with_source(
                    &IMREAD_ERROR_DECODE,
                    format!("unable to detect image format: {err}"),
                    err,
                )
            })?
    };
    reader.decode().map_err(|err| {
        imread_error_with_source(
            &IMREAD_ERROR_DECODE,
            format!("unable to decode image: {err}"),
            err,
        )
    })
}

pub(super) struct MaterializedImage {
    pub(super) image: Tensor,
    pub(super) alpha: Option<Tensor>,
}

pub(super) fn materialize_image(image: DynamicImage) -> BuiltinResult<MaterializedImage> {
    if let Some(buffer) = image.as_luma8() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                1,
                1,
                NumericDType::U8,
            )?,
            alpha: None,
        });
    }
    if let Some(buffer) = image.as_luma_alpha8() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                2,
                1,
                NumericDType::U8,
            )?,
            alpha: Some(alpha_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                2,
                1,
                NumericDType::U8,
            )?),
        });
    }
    if let Some(buffer) = image.as_rgb8() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                3,
                3,
                NumericDType::U8,
            )?,
            alpha: None,
        });
    }
    if let Some(buffer) = image.as_rgba8() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                4,
                3,
                NumericDType::U8,
            )?,
            alpha: Some(alpha_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                4,
                3,
                NumericDType::U8,
            )?),
        });
    }
    if let Some(buffer) = image.as_luma16() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                1,
                1,
                NumericDType::U16,
            )?,
            alpha: None,
        });
    }
    if let Some(buffer) = image.as_luma_alpha16() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                2,
                1,
                NumericDType::U16,
            )?,
            alpha: Some(alpha_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                2,
                1,
                NumericDType::U16,
            )?),
        });
    }
    if let Some(buffer) = image.as_rgb16() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                3,
                3,
                NumericDType::U16,
            )?,
            alpha: None,
        });
    }
    if let Some(buffer) = image.as_rgba16() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                4,
                3,
                NumericDType::U16,
            )?,
            alpha: Some(alpha_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                4,
                3,
                NumericDType::U16,
            )?),
        });
    }
    if let Some(buffer) = image.as_rgb32f() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                3,
                3,
                NumericDType::F32,
            )?,
            alpha: None,
        });
    }
    if let Some(buffer) = image.as_rgba32f() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                4,
                3,
                NumericDType::F32,
            )?,
            alpha: Some(alpha_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                4,
                3,
                NumericDType::F32,
            )?),
        });
    }

    let rgba = image.to_rgba8();
    Ok(MaterializedImage {
        image: tensor_from_interleaved(
            rgba.as_raw(),
            rgba.width(),
            rgba.height(),
            4,
            3,
            NumericDType::U8,
        )?,
        alpha: Some(alpha_from_interleaved(
            rgba.as_raw(),
            rgba.width(),
            rgba.height(),
            4,
            3,
            NumericDType::U8,
        )?),
    })
}

fn tensor_from_interleaved<T>(
    raw: &[T],
    width: u32,
    height: u32,
    input_channels: usize,
    output_channels: usize,
    dtype: NumericDType,
) -> BuiltinResult<Tensor>
where
    T: Copy + Into<f64>,
{
    let rows = height as usize;
    let cols = width as usize;
    let pixels = rows.saturating_mul(cols);
    let mut data = vec![0.0; pixels.saturating_mul(output_channels)];
    for row in 0..rows {
        for col in 0..cols {
            let source_base = (row * cols + col) * input_channels;
            let dest_base = row + rows * col;
            for channel in 0..output_channels {
                data[dest_base + pixels * channel] = raw[source_base + channel].into();
            }
        }
    }
    let shape = if output_channels == 1 {
        vec![rows, cols]
    } else {
        vec![rows, cols, output_channels]
    };
    let tensor = Tensor::new(data, shape)
        .map_err(|err| imread_error_with_detail(&IMREAD_ERROR_SHAPE, &err))?;
    Ok(tensor::coerce_tensor_dtype(tensor, dtype))
}

fn alpha_from_interleaved<T>(
    raw: &[T],
    width: u32,
    height: u32,
    input_channels: usize,
    alpha_channel: usize,
    dtype: NumericDType,
) -> BuiltinResult<Tensor>
where
    T: Copy + Into<f64>,
{
    let rows = height as usize;
    let cols = width as usize;
    let mut data = vec![0.0; rows.saturating_mul(cols)];
    for row in 0..rows {
        for col in 0..cols {
            let source_index = (row * cols + col) * input_channels + alpha_channel;
            let dest_index = row + rows * col;
            data[dest_index] = raw[source_index].into();
        }
    }
    let tensor = Tensor::new(data, vec![rows, cols])
        .map_err(|err| imread_error_with_detail(&IMREAD_ERROR_SHAPE, &err))?;
    Ok(tensor::coerce_tensor_dtype(tensor, dtype))
}

pub(super) fn empty_tensor_value() -> BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|err| imread_error_with_detail(&IMREAD_ERROR_SHAPE, &err))
}
