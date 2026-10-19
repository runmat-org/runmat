use super::*;

pub(super) async fn encode_image(
    image: MaterializedImage,
    invocation: &Invocation,
    cooperation: &mut ImageCooperation,
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
        ImageFormat::Gif => encode_gif(&image, invocation, cooperation).await,
        ImageFormat::Jpeg => write_dynamic_image(
            image_to_dynamic(image_as_8bit(image, cooperation).await?, false, cooperation).await?,
            ImageOutputFormat::Jpeg(invocation.options.quality),
        ),
        ImageFormat::Bmp => write_dynamic_image(
            image_to_dynamic(image_as_8bit(image, cooperation).await?, false, cooperation).await?,
            ImageOutputFormat::Bmp,
        ),
        ImageFormat::Png => write_dynamic_image(
            image_to_dynamic(image, true, cooperation).await?,
            ImageOutputFormat::Png,
        ),
        ImageFormat::Tiff => write_dynamic_image(
            image_to_dynamic(image, true, cooperation).await?,
            ImageOutputFormat::Tiff,
        ),
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_FORMAT,
            "unsupported image format",
        )),
    }
}

async fn image_to_dynamic(
    image: MaterializedImage,
    keep_alpha: bool,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<DynamicImage> {
    cooperation.completed(0).await?;
    let width = u32::try_from(image.cols).map_err(|_| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image width is too large")
    })?;
    let height = u32::try_from(image.rows).map_err(|_| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image height is too large")
    })?;

    match image.channels {
        1 => match image.data {
            PixelData::U8(data) => ImageBuffer::<Luma<u8>, _>::from_raw(width, height, data)
                .map(DynamicImage::ImageLuma8)
                .ok_or_else(|| {
                    imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_IMAGE,
                        "invalid grayscale image buffer",
                    )
                }),
            PixelData::U16(data) => ImageBuffer::<Luma<u16>, _>::from_raw(width, height, data)
                .map(DynamicImage::ImageLuma16)
                .ok_or_else(|| {
                    imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_IMAGE,
                        "invalid grayscale image buffer",
                    )
                }),
        },
        3 => match image.data {
            PixelData::U8(data) => ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, data)
                .map(DynamicImage::ImageRgb8)
                .ok_or_else(|| {
                    imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_IMAGE,
                        "invalid RGB image buffer",
                    )
                }),
            PixelData::U16(data) => ImageBuffer::<Rgb<u16>, _>::from_raw(width, height, data)
                .map(DynamicImage::ImageRgb16)
                .ok_or_else(|| {
                    imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_IMAGE,
                        "invalid RGB image buffer",
                    )
                }),
        },
        4 if keep_alpha => match image.data {
            PixelData::U8(data) => ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data)
                .map(DynamicImage::ImageRgba8)
                .ok_or_else(|| {
                    imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_IMAGE,
                        "invalid RGBA image buffer",
                    )
                }),
            PixelData::U16(data) => ImageBuffer::<Rgba<u16>, _>::from_raw(width, height, data)
                .map(DynamicImage::ImageRgba16)
                .ok_or_else(|| {
                    imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_IMAGE,
                        "invalid RGBA image buffer",
                    )
                }),
        },
        4 => match image.data {
            PixelData::U8(data) => {
                let mut rgb =
                    geometry::reserve::<u8>(geometry::samples(image.rows, image.cols, 3)?)?;
                let mut pending = 0;
                for chunk in data.chunks_exact(4) {
                    rgb.extend_from_slice(&chunk[..3]);
                    pending += 3;
                    if pending >= IMAGE_WORK_CHUNK {
                        cooperation.completed(pending).await?;
                        pending = 0;
                    }
                }
                cooperation.completed(pending).await?;
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
                let mut rgb =
                    geometry::reserve::<u16>(geometry::samples(image.rows, image.cols, 3)?)?;
                let mut pending = 0;
                for chunk in data.chunks_exact(4) {
                    rgb.extend_from_slice(&chunk[..3]);
                    pending += 3;
                    if pending >= IMAGE_WORK_CHUNK {
                        cooperation.completed(pending).await?;
                        pending = 0;
                    }
                }
                cooperation.completed(pending).await?;
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

async fn encode_gif(
    image: &MaterializedImage,
    invocation: &Invocation,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<Vec<u8>> {
    // Validate and construct the new frame before an append reads existing
    // filesystem data. The old frames and final encoded bytes still need a
    // separate bounded codec/resource contract.
    let new_frame = gif_frame_from_image(image, invocation.options.delay_time, cooperation).await?;
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
        cooperation.completed(0).await?;
        existing_repeat = gif_repeat_from_bytes(&existing);
        let decoder = GifDecoder::new(Cursor::new(existing)).map_err(|err| {
            imwrite_error_with_source(&IMWRITE_ERROR_ENCODE, "failed to decode GIF", err)
        })?;
        for frame in decoder.into_frames() {
            cooperation.completed(0).await?;
            frames.push(frame.map_err(|err| {
                imwrite_error_with_source(&IMWRITE_ERROR_ENCODE, "failed to decode GIF frame", err)
            })?);
        }
    }
    frames.push(new_frame);

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
            cooperation.completed(0).await?;
            encoder.encode_frame(frame).map_err(|err| {
                imwrite_error_with_source(&IMWRITE_ERROR_ENCODE, "failed to encode GIF frame", err)
            })?;
        }
    }
    Ok(bytes)
}

pub(super) fn gif_repeat_from_bytes(bytes: &[u8]) -> Option<Repeat> {
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

async fn gif_frame_from_image(
    image: &MaterializedImage,
    delay_time: Option<f64>,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<Frame> {
    cooperation.completed(0).await?;
    let width = u32::try_from(image.cols).map_err(|_| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image width is too large")
    })?;
    let height = u32::try_from(image.rows).map_err(|_| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image height is too large")
    })?;

    let pixels = geometry::pixels(image.rows, image.cols)?;
    let rgba_samples = geometry::samples(image.rows, image.cols, 4)?;
    let mut rgba = geometry::reserve::<u8>(rgba_samples)?;
    cooperation.fill(&mut rgba, rgba_samples).await?;
    let data = image_data_as_u8(image, cooperation).await?;
    let mut pending = 0;
    for pixel in 0..pixels {
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
        pending += 4;
        if pending >= IMAGE_WORK_CHUNK {
            cooperation.completed(pending).await?;
            pending = 0;
        }
    }
    cooperation.completed(pending).await?;
    let image = RgbaImage::from_raw(width, height, rgba).ok_or_else(|| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "invalid GIF frame buffer")
    })?;
    let delay = delay_time
        .map(|seconds| Delay::from_saturating_duration(Duration::from_secs_f64(seconds)))
        .unwrap_or_else(|| Delay::from_numer_denom_ms(0, 1));
    Ok(Frame::from_parts(image, 0, 0, delay))
}

async fn image_data_as_u8<'a>(
    image: &'a MaterializedImage,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<std::borrow::Cow<'a, [u8]>> {
    cooperation.completed(0).await?;
    match &image.data {
        PixelData::U8(data) => Ok(std::borrow::Cow::Borrowed(data)),
        PixelData::U16(data) => {
            let mut converted = geometry::reserve::<u8>(data.len())?;
            for chunk in data.chunks(IMAGE_WORK_CHUNK) {
                converted.extend(
                    chunk
                        .iter()
                        .map(|value| ((*value as f64) / 257.0).round().clamp(0.0, 255.0) as u8),
                );
                cooperation.completed(chunk.len()).await?;
            }
            Ok(std::borrow::Cow::Owned(converted))
        }
    }
}

async fn image_as_8bit(
    image: MaterializedImage,
    cooperation: &mut ImageCooperation,
) -> BuiltinResult<MaterializedImage> {
    cooperation.completed(0).await?;
    Ok(MaterializedImage {
        rows: image.rows,
        cols: image.cols,
        channels: image.channels,
        data: PixelData::U8(match image.data {
            PixelData::U8(data) => data,
            PixelData::U16(data) => {
                let mut converted = geometry::reserve::<u8>(data.len())?;
                for chunk in data.chunks(IMAGE_WORK_CHUNK) {
                    converted.extend(
                        chunk
                            .iter()
                            .map(|value| ((*value as f64) / 257.0).round().clamp(0.0, 255.0) as u8),
                    );
                    cooperation.completed(chunk.len()).await?;
                }
                converted
            }
        }),
        alpha_applied: image.alpha_applied,
        indexed_source: image.indexed_source,
    })
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    #[test]
    fn native_pixels_move_into_dynamic_image_without_a_second_buffer() {
        let pixels = vec![1_u8, 2, 3, 4, 5, 6];
        let original = pixels.as_ptr();
        let image = MaterializedImage {
            rows: 1,
            cols: 2,
            channels: 3,
            data: PixelData::U8(pixels),
            alpha_applied: false,
            indexed_source: false,
        };
        let dynamic = futures::executor::block_on(image_to_dynamic(
            image,
            true,
            &mut ImageCooperation::new(None, BUILTIN_NAME),
        ))
        .unwrap();
        let DynamicImage::ImageRgb8(buffer) = dynamic else {
            panic!("expected native RGB8 image")
        };
        assert_eq!(buffer.as_raw().as_ptr(), original);
        assert_eq!(buffer.as_raw(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn eight_bit_conversion_reuses_native_pixels() {
        let pixels = vec![7_u8, 8, 9];
        let original = pixels.as_ptr();
        let image = MaterializedImage {
            rows: 1,
            cols: 1,
            channels: 3,
            data: PixelData::U8(pixels),
            alpha_applied: false,
            indexed_source: false,
        };
        let converted = futures::executor::block_on(image_as_8bit(
            image,
            &mut ImageCooperation::new(None, BUILTIN_NAME),
        ))
        .unwrap();
        let PixelData::U8(data) = converted.data else {
            panic!("expected eight-bit pixels")
        };
        assert_eq!(data.as_ptr(), original);
    }

    #[test]
    fn explicit_sixteen_bit_to_byte_conversion_keeps_scaling() {
        let image = MaterializedImage {
            rows: 1,
            cols: 3,
            channels: 1,
            data: PixelData::U16(vec![0, 257, u16::MAX]),
            alpha_applied: false,
            indexed_source: false,
        };
        let converted = futures::executor::block_on(image_as_8bit(
            image,
            &mut ImageCooperation::new(None, BUILTIN_NAME),
        ))
        .unwrap();
        let PixelData::U8(data) = converted.data else {
            panic!("expected eight-bit pixels")
        };
        assert_eq!(data, vec![0, 1, u8::MAX]);
    }

    #[test]
    fn gif_append_rejects_invalid_new_frame_before_reading_old_file() {
        let image = MaterializedImage {
            rows: usize::MAX,
            cols: 2,
            channels: 1,
            data: PixelData::U8(Vec::new()),
            alpha_applied: false,
            indexed_source: false,
        };
        let invocation = Invocation {
            image: Value::Num(0.0),
            map: None,
            alpha: None,
            path: PathBuf::from("nonexistent-gif-append-before-read.gif"),
            format: ImageFormat::Gif,
            options: ImwriteOptions {
                write_mode: WriteMode::Append,
                ..ImwriteOptions::default()
            },
        };
        let error = futures::executor::block_on(encode_gif(
            &image,
            &invocation,
            &mut ImageCooperation::new(None, BUILTIN_NAME),
        ))
        .expect_err("invalid new frame must precede missing append source");
        assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
    }
}
