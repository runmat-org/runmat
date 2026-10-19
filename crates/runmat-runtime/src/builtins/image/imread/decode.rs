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

pub(super) async fn materialize_image(
    image: DynamicImage,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<MaterializedImage> {
    if let Some(buffer) = image.as_luma8() {
        return Ok(MaterializedImage {
            image: tensor_from_interleaved(
                buffer.as_raw(),
                buffer.width(),
                buffer.height(),
                1,
                1,
                NumericDType::U8,
                cooperation,
            )
            .await?,
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
                cooperation,
            )
            .await?,
            alpha: Some(
                alpha_from_interleaved(
                    buffer.as_raw(),
                    buffer.width(),
                    buffer.height(),
                    2,
                    1,
                    NumericDType::U8,
                    cooperation,
                )
                .await?,
            ),
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
                cooperation,
            )
            .await?,
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
                cooperation,
            )
            .await?,
            alpha: Some(
                alpha_from_interleaved(
                    buffer.as_raw(),
                    buffer.width(),
                    buffer.height(),
                    4,
                    3,
                    NumericDType::U8,
                    cooperation,
                )
                .await?,
            ),
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
                cooperation,
            )
            .await?,
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
                cooperation,
            )
            .await?,
            alpha: Some(
                alpha_from_interleaved(
                    buffer.as_raw(),
                    buffer.width(),
                    buffer.height(),
                    2,
                    1,
                    NumericDType::U16,
                    cooperation,
                )
                .await?,
            ),
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
                cooperation,
            )
            .await?,
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
                cooperation,
            )
            .await?,
            alpha: Some(
                alpha_from_interleaved(
                    buffer.as_raw(),
                    buffer.width(),
                    buffer.height(),
                    4,
                    3,
                    NumericDType::U16,
                    cooperation,
                )
                .await?,
            ),
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
                cooperation,
            )
            .await?,
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
                cooperation,
            )
            .await?,
            alpha: Some(
                alpha_from_interleaved(
                    buffer.as_raw(),
                    buffer.width(),
                    buffer.height(),
                    4,
                    3,
                    NumericDType::F32,
                    cooperation,
                )
                .await?,
            ),
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
            cooperation,
        )
        .await?,
        alpha: Some(
            alpha_from_interleaved(
                rgba.as_raw(),
                rgba.width(),
                rgba.height(),
                4,
                3,
                NumericDType::U8,
                cooperation,
            )
            .await?,
        ),
    })
}

async fn tensor_from_interleaved<T>(
    raw: &[T],
    width: u32,
    height: u32,
    input_channels: usize,
    output_channels: usize,
    dtype: NumericDType,
    cooperation: &mut ImageCooperation,
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
    cooperation.completed(0).await?;
    let mut data = native_buffer::<T>(output_len)?;
    cooperation.fill(&mut data, output_len).await?;
    let mut completed = 0;
    for row in 0..rows {
        for col in 0..cols {
            let source_base = (row * cols + col) * input_channels;
            let dest_base = row + rows * col;
            for channel in 0..output_channels {
                data[dest_base + pixels * channel] = raw[source_base + channel];
                completed += 1;
                if completed == crate::builtins::image::io_cooperation::IMAGE_WORK_CHUNK {
                    cooperation.completed(completed).await?;
                    completed = 0;
                }
            }
        }
    }
    cooperation.completed(completed).await?;
    let shape = if output_channels == 1 {
        vec![rows, cols]
    } else {
        vec![rows, cols, output_channels]
    };
    Tensor::from_numeric_storage(T::into_storage(data), shape)
        .map_err(|err| imread_error_with_detail(&IMREAD_ERROR_SHAPE, &err))
}

async fn alpha_from_interleaved<T>(
    raw: &[T],
    width: u32,
    height: u32,
    input_channels: usize,
    alpha_channel: usize,
    dtype: NumericDType,
    cooperation: &mut ImageCooperation,
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
    cooperation.completed(0).await?;
    let mut data = native_buffer::<T>(pixels)?;
    cooperation.fill(&mut data, pixels).await?;
    let mut completed = 0;
    for row in 0..rows {
        for col in 0..cols {
            let source_index = (row * cols + col) * input_channels + alpha_channel;
            let dest_index = row + rows * col;
            data[dest_index] = raw[source_index];
            completed += 1;
            if completed == crate::builtins::image::io_cooperation::IMAGE_WORK_CHUNK {
                cooperation.completed(completed).await?;
                completed = 0;
            }
        }
    }
    cooperation.completed(completed).await?;
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
    if input_channels == 0 || output_channels == 0 || output_channels > input_channels {
        return Err(imread_error_with_detail(
            &IMREAD_ERROR_SHAPE,
            "decoded image channel geometry is invalid",
        ));
    }
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

fn native_buffer<T: DecodedSample>(elements: usize) -> BuiltinResult<Vec<T>> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| {
            imread_error_with_detail(
                &IMREAD_ERROR_SHAPE,
                "decoded image allocation size overflows",
            )
        })?;
    if bytes > isize::MAX as usize {
        return Err(imread_error_with_detail(
            &IMREAD_ERROR_SHAPE,
            "decoded image allocation exceeds addressable storage",
        ));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        imread_error_with_source(
            &IMREAD_ERROR_SHAPE,
            "decoded image allocation is unavailable",
            error,
        )
    })?;
    Ok(values)
}

pub(super) fn empty_tensor_value() -> BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|err| imread_error_with_detail(&IMREAD_ERROR_SHAPE, &err))
}

#[cfg(test)]
#[path = "decode/tests.rs"]
mod tests;
