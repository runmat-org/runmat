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

trait DecodedSample: Copy + Default {
    const DTYPE: NumericDType;
    fn into_storage(values: Vec<Self>) -> NumericStorage;
}

impl DecodedSample for u8 {
    const DTYPE: NumericDType = NumericDType::U8;
    fn into_storage(values: Vec<Self>) -> NumericStorage {
        NumericStorage::U8(values)
    }
}

impl DecodedSample for u16 {
    const DTYPE: NumericDType = NumericDType::U16;
    fn into_storage(values: Vec<Self>) -> NumericStorage {
        NumericStorage::U16(values)
    }
}

impl DecodedSample for f32 {
    const DTYPE: NumericDType = NumericDType::F32;
    fn into_storage(values: Vec<Self>) -> NumericStorage {
        NumericStorage::F32(values)
    }
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
    T: DecodedSample,
{
    let rows = height as usize;
    let cols = width as usize;
    let (pixels, _, output_len) =
        checked_interleaved_geometry(rows, cols, input_channels, output_channels, raw.len())?;
    if dtype != T::DTYPE {
        return Err(imread_error_with_detail(
            &IMREAD_ERROR_SHAPE,
            "decoded sample class does not match image storage",
        ));
    }
    let mut data = vec![T::default(); output_len];
    for row in 0..rows {
        for col in 0..cols {
            let source_base = (row * cols + col) * input_channels;
            let dest_base = row + rows * col;
            for channel in 0..output_channels {
                data[dest_base + pixels * channel] = raw[source_base + channel];
            }
        }
    }
    let shape = if output_channels == 1 {
        vec![rows, cols]
    } else {
        vec![rows, cols, output_channels]
    };
    Tensor::from_numeric_storage(T::into_storage(data), shape)
        .map_err(|err| imread_error_with_detail(&IMREAD_ERROR_SHAPE, &err))
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
    T: DecodedSample,
{
    let rows = height as usize;
    let cols = width as usize;
    let (pixels, _, _) = checked_interleaved_geometry(rows, cols, input_channels, 1, raw.len())?;
    if alpha_channel >= input_channels {
        return Err(imread_error_with_detail(
            &IMREAD_ERROR_SHAPE,
            "alpha channel is outside decoded image storage",
        ));
    }
    if dtype != T::DTYPE {
        return Err(imread_error_with_detail(
            &IMREAD_ERROR_SHAPE,
            "decoded alpha class does not match image storage",
        ));
    }
    let mut data = vec![T::default(); pixels];
    for row in 0..rows {
        for col in 0..cols {
            let source_index = (row * cols + col) * input_channels + alpha_channel;
            let dest_index = row + rows * col;
            data[dest_index] = raw[source_index];
        }
    }
    Tensor::from_numeric_storage(T::into_storage(data), vec![rows, cols])
        .map_err(|err| imread_error_with_detail(&IMREAD_ERROR_SHAPE, &err))
}

fn checked_interleaved_geometry(
    rows: usize,
    cols: usize,
    input_channels: usize,
    output_channels: usize,
    actual_len: usize,
) -> BuiltinResult<(usize, usize, usize)> {
    let lengths = rows
        .checked_mul(cols)
        .and_then(|pixels| {
            Some((
                pixels,
                pixels.checked_mul(input_channels)?,
                pixels.checked_mul(output_channels)?,
            ))
        })
        .ok_or_else(|| {
            imread_error_with_detail(&IMREAD_ERROR_SHAPE, "decoded image dimensions overflow")
        })?;
    if lengths.1 != actual_len {
        return Err(imread_error_with_detail(
            &IMREAD_ERROR_SHAPE,
            "decoded pixel storage does not match image dimensions",
        ));
    }
    Ok(lengths)
}

pub(super) fn empty_tensor_value() -> BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|err| imread_error_with_detail(&IMREAD_ERROR_SHAPE, &err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_geometry_rejects_overflow_and_truncated_pixel_storage() {
        let error =
            checked_interleaved_geometry(usize::MAX, 2, 4, 3, 0).expect_err("geometry overflow");
        assert_eq!(error.identifier(), IMREAD_ERROR_SHAPE.identifier);
        let error = checked_interleaved_geometry(2, 3, 4, 3, 23).expect_err("truncated storage");
        assert_eq!(error.identifier(), IMREAD_ERROR_SHAPE.identifier);
        assert_eq!(
            checked_interleaved_geometry(2, 3, 4, 3, 24).unwrap(),
            (6, 24, 18)
        );
    }

    #[test]
    fn valid_interleaved_alpha_keeps_native_class_and_column_major_order() {
        let raw = [10_u8, 20, 30, 40, 50, 60, 70, 80];
        let image = tensor_from_interleaved(&raw, 2, 1, 4, 3, NumericDType::U8).unwrap();
        let alpha = alpha_from_interleaved(&raw, 2, 1, 4, 3, NumericDType::U8).unwrap();
        assert_eq!(image.numeric_dtype(), NumericDType::U8);
        assert_eq!(image.shape, vec![1, 2, 3]);
        assert_eq!(
            image.materialize_f64(),
            vec![10.0, 50.0, 20.0, 60.0, 30.0, 70.0]
        );
        assert_eq!(alpha.numeric_dtype(), NumericDType::U8);
        assert_eq!(alpha.shape, vec![1, 2]);
        assert_eq!(alpha.materialize_f64(), vec![40.0, 80.0]);
    }
}
