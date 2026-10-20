use runmat_value::IntValue;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Duration;

use image::codecs::gif::{GifDecoder, GifEncoder, Repeat};
use image::{AnimationDecoder, Delay, DynamicImage, Frame, ImageFormat, ImageOutputFormat};
use image::{ImageBuffer, Luma, Rgb, Rgba, RgbaImage};
use runmat_builtins::{
    BuiltinErrorDescriptor, ImwriteOption, ImwriteWriteMode as WriteMode, IMWRITE_ERROR_ENCODE,
    IMWRITE_ERROR_INVALID_ARGUMENT, IMWRITE_ERROR_INVALID_COLORMAP, IMWRITE_ERROR_INVALID_FILENAME,
    IMWRITE_ERROR_INVALID_FORMAT, IMWRITE_ERROR_INVALID_IMAGE, IMWRITE_ERROR_INVALID_OPTION,
    IMWRITE_ERROR_IO, IMWRITE_SINGLE_GIF_TIFF_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntegerStorage, LogicalArray, NumericDType, NumericScalar, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::image::io_cooperation::{ImageCooperation, IMAGE_WORK_CHUNK};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod admission;
mod encode;
mod geometry;
mod materialize;
mod options;
mod parse;
mod preflight;

use self::encode::encode_image;
use self::materialize::{
    ensure_documented_alpha_class, ensure_documented_image_class, ensure_double_colormap,
    image_dimensions, image_dimensions_from_shape, materialize_image, tensor_from_numeric_like,
};
use self::parse::{
    infer_format_from_path, is_string_like, numeric_scalar, parse_format_hint, parse_invocation,
    string_arg,
};

const BUILTIN_NAME: &str = "imwrite";

#[runtime_builtin(
    name = "imwrite",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::imwrite"
)]
async fn imwrite_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if let Some(n) = crate::output_count::current_output_count() {
        if n > 0 {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_ARGUMENT,
                "imwrite does not return output arguments",
            ));
        }
    }

    if args.len() < 2 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_ARGUMENT,
            "expected image data and filename",
        ));
    }
    // A path is host text, never a numeric provider operand. Check the
    // identifiable path slot before any image/map readback or filesystem I/O.
    let filename_index = if is_string_like(&args[1]) { 1 } else { 2 };
    let filename = args.get(filename_index).ok_or_else(|| {
        imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_ARGUMENT,
            "indexed images require X, map, and filename",
        )
    })?;
    let filename = string_arg("filename", filename, &IMWRITE_ERROR_INVALID_FILENAME)?;
    if filename.trim().is_empty() {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_FILENAME,
            "filename must not be empty",
        ));
    }
    preflight::preflight_resident_argument_roles(args.view())?;
    let format = preflight::preflight_option_text(
        args.view(),
        filename_index + 1,
        &filename,
        filename_index == 2,
    )?;
    if matches!(format, ImageFormat::Gif | ImageFormat::Tiff)
        && preflight::image_is_single(&args[0])
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMWRITE_SINGLE_GIF_TIFF_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let mut cooperation = ImageCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    preflight::preflight_known_host_pixels(args.view(), &mut cooperation).await?;
    cooperation.completed(0).await?;
    let mut host_args = Vec::with_capacity(args.len());
    for arg in &args {
        host_args.push(preflight::gather_imwrite_argument(arg).await?);
        cooperation.completed(0).await?;
    }

    let invocation = parse_invocation(&host_args, &mut cooperation).await?;
    if matches!(invocation.format, ImageFormat::Gif | ImageFormat::Tiff)
        && value_numeric_dtype(&invocation.image) == Some(NumericDType::F32)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMWRITE_SINGLE_GIF_TIFF_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let image = materialize_image(
        &invocation.image,
        invocation.map.as_ref(),
        invocation.alpha.as_ref(),
        &mut cooperation,
    )
    .await?;
    let bytes = encode_image(image, &invocation, &mut cooperation).await?;
    cooperation.completed(0).await?;
    runmat_filesystem::write_async(&invocation.path, &bytes)
        .await
        .map_err(|err| {
            imwrite_error_with_source(
                &IMWRITE_ERROR_IO,
                format!("failed to write '{}'", invocation.path.display()),
                err,
            )
        })?;

    Ok(Value::output_list(Vec::new()))
}

fn value_numeric_dtype(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Num(_) => Some(NumericDType::F64),
        Value::Tensor(tensor) => Some(tensor.numeric_dtype()),
        _ => None,
    }
}

#[derive(Debug)]
struct ImwriteOptions {
    quality: u8,
    delay_time: Option<f64>,
    loop_count: Option<f64>,
    write_mode: WriteMode,
}

impl Default for ImwriteOptions {
    fn default() -> Self {
        Self {
            quality: 75,
            delay_time: None,
            loop_count: None,
            write_mode: WriteMode::Overwrite,
        }
    }
}

#[derive(Debug)]
struct Invocation {
    image: Value,
    map: Option<Value>,
    alpha: Option<Tensor>,
    path: PathBuf,
    format: ImageFormat,
    options: ImwriteOptions,
}

#[derive(Clone)]
struct MaterializedImage {
    rows: usize,
    cols: usize,
    channels: usize,
    data: PixelData,
    alpha_applied: bool,
    indexed_source: bool,
}

#[derive(Clone)]
enum PixelData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

fn imwrite_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn imwrite_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    context: impl AsRef<str>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    build_runtime_error(format!("{}: {}: {source}", error.message, context.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(error.identifier.unwrap_or("RunMat:imwrite:Internal"))
        .with_source(source)
        .build()
}

#[cfg(test)]
mod tests;
