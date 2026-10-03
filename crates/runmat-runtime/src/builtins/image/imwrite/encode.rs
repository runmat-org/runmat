use super::*;

pub(super) async fn encode_image(
    image: &MaterializedImage,
    invocation: &Invocation,
) -> BuiltinResult<Vec<u8>> {
    admission::validate(&admission::EncoderFacts {
        format: invocation.format,
        channels: image.channels,
        sixteen_bit: matches!(&image.data, PixelData::U16(_)),
        indexed_source: image.indexed_source,
        alpha_applied: image.alpha_applied,
        append: invocation.options.write_mode == WriteMode::Append,
    })?;

    match invocation.format {
        ImageFormat::Gif => encode_gif(image, invocation).await,
        ImageFormat::Jpeg => write_dynamic_image(
            image_to_dynamic(&image_as_8bit(image), false)?,
            ImageOutputFormat::Jpeg(invocation.options.quality),
        ),
        ImageFormat::Bmp => write_dynamic_image(
            image_to_dynamic(&image_as_8bit(image), false)?,
            ImageOutputFormat::Bmp,
        ),
        ImageFormat::Png => {
            write_dynamic_image(image_to_dynamic(image, true)?, ImageOutputFormat::Png)
        }
        ImageFormat::Tiff => {
            write_dynamic_image(image_to_dynamic(image, true)?, ImageOutputFormat::Tiff)
        }
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_FORMAT,
            "unsupported image format",
        )),
    }
}

fn image_to_dynamic(image: &MaterializedImage, keep_alpha: bool) -> BuiltinResult<DynamicImage> {
    let width = u32::try_from(image.cols).map_err(|_| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image width is too large")
    })?;
    let height = u32::try_from(image.rows).map_err(|_| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image height is too large")
    })?;

    match image.channels {
        1 => match &image.data {
            PixelData::U8(data) => {
                ImageBuffer::<Luma<u8>, _>::from_raw(width, height, data.clone())
                    .map(DynamicImage::ImageLuma8)
                    .ok_or_else(|| {
                        imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_IMAGE,
                            "invalid grayscale image buffer",
                        )
                    })
            }
            PixelData::U16(data) => {
                ImageBuffer::<Luma<u16>, _>::from_raw(width, height, data.clone())
                    .map(DynamicImage::ImageLuma16)
                    .ok_or_else(|| {
                        imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_IMAGE,
                            "invalid grayscale image buffer",
                        )
                    })
            }
        },
        3 => match &image.data {
            PixelData::U8(data) => ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, data.clone())
                .map(DynamicImage::ImageRgb8)
                .ok_or_else(|| {
                    imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_IMAGE,
                        "invalid RGB image buffer",
                    )
                }),
            PixelData::U16(data) => {
                ImageBuffer::<Rgb<u16>, _>::from_raw(width, height, data.clone())
                    .map(DynamicImage::ImageRgb16)
                    .ok_or_else(|| {
                        imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_IMAGE,
                            "invalid RGB image buffer",
                        )
                    })
            }
        },
        4 if keep_alpha => match &image.data {
            PixelData::U8(data) => {
                ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data.clone())
                    .map(DynamicImage::ImageRgba8)
                    .ok_or_else(|| {
                        imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_IMAGE,
                            "invalid RGBA image buffer",
                        )
                    })
            }
            PixelData::U16(data) => {
                ImageBuffer::<Rgba<u16>, _>::from_raw(width, height, data.clone())
                    .map(DynamicImage::ImageRgba16)
                    .ok_or_else(|| {
                        imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_IMAGE,
                            "invalid RGBA image buffer",
                        )
                    })
            }
        },
        4 => match &image.data {
            PixelData::U8(data) => {
                let mut rgb = Vec::with_capacity(image.rows * image.cols * 3);
                for chunk in data.chunks_exact(4) {
                    rgb.extend_from_slice(&chunk[..3]);
                }
                ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb)
                    .map(DynamicImage::ImageRgb8)
                    .ok_or_else(|| {
                        imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_IMAGE,
                            "invalid RGB image buffer",
                        )
                    })
            }
            PixelData::U16(data) => {
                let mut rgb = Vec::with_capacity(image.rows * image.cols * 3);
                for chunk in data.chunks_exact(4) {
                    rgb.extend_from_slice(&chunk[..3]);
                }
                ImageBuffer::<Rgb<u16>, _>::from_raw(width, height, rgb)
                    .map(DynamicImage::ImageRgb16)
                    .ok_or_else(|| {
                        imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_IMAGE,
                            "invalid RGB image buffer",
                        )
                    })
            }
        },
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "image must have 1, 3, or 4 channels",
        )),
    }
}

fn write_dynamic_image(image: DynamicImage, format: ImageOutputFormat) -> BuiltinResult<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    image.write_to(&mut cursor, format).map_err(|err| {
        imwrite_error_with_source(&IMWRITE_ERROR_ENCODE, "unable to encode image", err)
    })?;
    Ok(cursor.into_inner())
}

async fn encode_gif(image: &MaterializedImage, invocation: &Invocation) -> BuiltinResult<Vec<u8>> {
    let mut frames = Vec::new();
    let mut existing_repeat = None;
    if invocation.options.write_mode == WriteMode::Append {
        // GIF append is inherently read-modify-write with the current filesystem
        // abstraction; providers do not expose a portable advisory/exclusive lock.
        let existing = runmat_filesystem::read_async(&invocation.path)
            .await
            .map_err(|err| {
                imwrite_error_with_source(
                    &IMWRITE_ERROR_IO,
                    format!(
                        "failed to read GIF for append '{}'",
                        invocation.path.display()
                    ),
                    err,
                )
            })?;
        existing_repeat = gif_repeat_from_bytes(&existing);
        let decoder = GifDecoder::new(Cursor::new(existing)).map_err(|err| {
            imwrite_error_with_source(&IMWRITE_ERROR_ENCODE, "failed to decode GIF", err)
        })?;
        for frame in decoder.into_frames() {
            frames.push(frame.map_err(|err| {
                imwrite_error_with_source(&IMWRITE_ERROR_ENCODE, "failed to decode GIF frame", err)
            })?);
        }
    }
    frames.push(gif_frame_from_image(image, invocation.options.delay_time)?);

    let mut bytes = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut bytes);
        let repeat = if let Some(loop_count) = invocation.options.loop_count {
            Some(loop_count_to_repeat(loop_count)?)
        } else {
            existing_repeat
        };
        if let Some(repeat) = repeat {
            encoder.set_repeat(repeat).map_err(|err| {
                imwrite_error_with_source(&IMWRITE_ERROR_ENCODE, "failed to set GIF repeat", err)
            })?;
        }
        for frame in frames {
            encoder.encode_frame(frame).map_err(|err| {
                imwrite_error_with_source(&IMWRITE_ERROR_ENCODE, "failed to encode GIF frame", err)
            })?;
        }
    }
    Ok(bytes)
}

fn gif_repeat_from_bytes(bytes: &[u8]) -> Option<Repeat> {
    const APP_EXT_PREFIX: &[u8] = b"\x21\xFF\x0BNETSCAPE2.0\x03\x01";
    bytes.windows(APP_EXT_PREFIX.len() + 3).find_map(|window| {
        if !window.starts_with(APP_EXT_PREFIX) || window[APP_EXT_PREFIX.len() + 2] != 0 {
            return None;
        }
        let lo = window[APP_EXT_PREFIX.len()];
        let hi = window[APP_EXT_PREFIX.len() + 1];
        let count = u16::from_le_bytes([lo, hi]);
        if count == 0 {
            Some(Repeat::Infinite)
        } else {
            Some(Repeat::Finite(count))
        }
    })
}

fn loop_count_to_repeat(loop_count: f64) -> BuiltinResult<Repeat> {
    options::validate_numeric(ImwriteOption::LoopCount, loop_count)?;
    if loop_count.is_infinite() {
        return Ok(Repeat::Infinite);
    }
    Ok(Repeat::Finite(loop_count.round() as u16))
}

fn gif_frame_from_image(
    image: &MaterializedImage,
    delay_time: Option<f64>,
) -> BuiltinResult<Frame> {
    let width = u32::try_from(image.cols).map_err(|_| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image width is too large")
    })?;
    let height = u32::try_from(image.rows).map_err(|_| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image height is too large")
    })?;

    let mut rgba = vec![0u8; image.rows * image.cols * 4];
    let data = image_data_as_u8(image);
    for pixel in 0..image.rows * image.cols {
        let dst = pixel * 4;
        match image.channels {
            1 => {
                let gray = data[pixel];
                rgba[dst] = gray;
                rgba[dst + 1] = gray;
                rgba[dst + 2] = gray;
                rgba[dst + 3] = 255;
            }
            3 => {
                let src = pixel * 3;
                rgba[dst..dst + 3].copy_from_slice(&data[src..src + 3]);
                rgba[dst + 3] = 255;
            }
            4 => {
                let src = pixel * 4;
                rgba[dst..dst + 4].copy_from_slice(&data[src..src + 4]);
            }
            _ => unreachable!(),
        }
    }
    let image = RgbaImage::from_raw(width, height, rgba).ok_or_else(|| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "invalid GIF frame buffer")
    })?;
    let delay = delay_time
        .map(|seconds| Delay::from_saturating_duration(Duration::from_secs_f64(seconds)))
        .unwrap_or_else(|| Delay::from_numer_denom_ms(0, 1));
    Ok(Frame::from_parts(image, 0, 0, delay))
}

fn image_data_as_u8(image: &MaterializedImage) -> Vec<u8> {
    match &image.data {
        PixelData::U8(data) => data.clone(),
        PixelData::U16(data) => data
            .iter()
            .map(|value| ((*value as f64) / 257.0).round().clamp(0.0, 255.0) as u8)
            .collect(),
    }
}

fn image_as_8bit(image: &MaterializedImage) -> MaterializedImage {
    MaterializedImage {
        rows: image.rows,
        cols: image.cols,
        channels: image.channels,
        data: PixelData::U8(image_data_as_u8(image)),
        alpha_applied: image.alpha_applied,
        indexed_source: image.indexed_source,
    }
}
