//! MATLAB-compatible `imfinfo` metadata reader.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::io::Reader as ImageReader;
use runmat_builtins::{
    IMFINFO_DESCRIPTOR, IMFINFO_ERROR_ARGUMENT, IMFINFO_ERROR_DECODE, IMFINFO_ERROR_FORMAT,
    IMFINFO_ERROR_IO, IMFINFO_ERROR_OUTPUTS,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, StructValue, Value};

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn info_error(
    error: &'static runmat_builtins::BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    build_runtime_error(format!("{}: {detail}", error.message))
        .with_builtin("imfinfo")
        .with_identifier(error.identifier.unwrap_or("RunMat:imfinfo:Internal"))
        .build()
}

fn info_error_with_source<E>(
    error: &'static runmat_builtins::BuiltinErrorDescriptor,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    build_runtime_error(format!("{}: {source}", error.message))
        .with_builtin("imfinfo")
        .with_identifier(error.identifier.unwrap_or("RunMat:imfinfo:Internal"))
        .with_source(source)
        .build()
}

fn scalar_text(value: &Value, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(info_error(
            &IMFINFO_ERROR_ARGUMENT,
            format!("{arg} must be a string scalar or character vector"),
        )),
    }
}

fn char_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[runtime_builtin(
    name = "imfinfo",
    binding_variant = "default",
    builtin_path = "crate::builtins::image::imfinfo"
)]
async fn imfinfo_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &IMFINFO_DESCRIPTOR,
        &IMFINFO_ERROR_OUTPUTS,
        "imfinfo",
    )?;
    if args.is_empty() || args.len() > 2 {
        return Err(info_error(
            &IMFINFO_ERROR_ARGUMENT,
            "filename and optional format are expected",
        ));
    }
    let filename = scalar_text(&args[0], "filename")?;
    if filename.trim().is_empty() {
        return Err(info_error(
            &IMFINFO_ERROR_ARGUMENT,
            "filename must not be empty",
        ));
    }
    let explicit_format = args
        .get(1)
        .map(|value| {
            let hint = scalar_text(value, "fmt")?;
            let normalized = hint.trim().trim_start_matches('.').to_ascii_lowercase();
            image::ImageFormat::from_extension(&normalized)
                .ok_or_else(|| info_error(&IMFINFO_ERROR_FORMAT, "unsupported image format hint"))
        })
        .transpose()?;
    let path = PathBuf::from(
        expand_user_path(filename.trim(), "imfinfo")
            .map_err(|error| info_error(&IMFINFO_ERROR_ARGUMENT, error))?,
    );
    let bytes = vfs::read_async(&path)
        .await
        .map_err(|err| info_error_with_source(&IMFINFO_ERROR_IO, err))?;
    let mut reader = ImageReader::new(Cursor::new(bytes.as_slice()))
        .with_guessed_format()
        .map_err(|err| info_error_with_source(&IMFINFO_ERROR_DECODE, err))?;
    if let Some(format) = explicit_format {
        reader.set_format(format);
    }
    let format = reader.format();
    let image = reader
        .decode()
        .map_err(|err| info_error_with_source(&IMFINFO_ERROR_DECODE, err))?;
    let color = image.color();
    let mut st = StructValue::new();
    st.insert("Filename", char_value(&path_to_string(&path)));
    st.insert("FileModDate", char_value(""));
    st.insert("FileSize", Value::Num(bytes.len() as f64));
    st.insert("Format", char_value(format_name(format, &path)));
    st.insert("FormatVersion", char_value(""));
    st.insert("Width", Value::Num(image.width() as f64));
    st.insert("Height", Value::Num(image.height() as f64));
    st.insert("BitDepth", Value::Num(color.bits_per_pixel() as f64));
    st.insert("ColorType", char_value(color_type_name(color)));
    Ok(Value::Struct(st))
}

fn format_name(format: Option<image::ImageFormat>, path: &Path) -> &str {
    match format {
        Some(image::ImageFormat::Png) => "png",
        Some(image::ImageFormat::Jpeg) => "jpg",
        Some(image::ImageFormat::Gif) => "gif",
        Some(image::ImageFormat::Tiff) => "tif",
        Some(image::ImageFormat::Bmp) => "bmp",
        Some(image::ImageFormat::Ico) => "ico",
        Some(image::ImageFormat::Pnm) => "pnm",
        Some(image::ImageFormat::Tga) => "tga",
        Some(image::ImageFormat::Dds) => "dds",
        Some(image::ImageFormat::WebP) => "webp",
        Some(image::ImageFormat::Avif) => "avif",
        _ => path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default(),
    }
}

fn color_type_name(color: image::ColorType) -> &'static str {
    match color {
        image::ColorType::L8
        | image::ColorType::L16
        | image::ColorType::La8
        | image::ColorType::La16 => "grayscale",
        image::ColorType::Rgb8 | image::ColorType::Rgb16 | image::ColorType::Rgb32F => "truecolor",
        image::ColorType::Rgba8 | image::ColorType::Rgba16 | image::ColorType::Rgba32F => {
            "truecoloralpha"
        }
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn imfinfo_reports_png_dimensions() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let root = tempfile::TempDir::new().unwrap();
        let path = root.path().join("image.png");
        let image = image::RgbImage::from_pixel(3, 2, image::Rgb([1, 2, 3]));
        image.save(&path).unwrap();
        let value = run(imfinfo_builtin(vec![Value::String(path_to_string(&path))])).unwrap();
        match value {
            Value::Struct(st) => {
                assert_eq!(st.fields.get("Width"), Some(&Value::Num(3.0)));
                assert_eq!(st.fields.get("Height"), Some(&Value::Num(2.0)));
                assert_eq!(st.fields.get("Format"), Some(&char_value("png")));
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn invalid_output_and_format_precede_missing_file_read() {
        let root = tempfile::TempDir::new().unwrap();
        let missing = Value::String(path_to_string(&root.path().join("missing.png")));
        let _outputs = crate::output_count::push_output_count(Some(2));
        let error = run(imfinfo_builtin(vec![missing.clone()])).unwrap_err();
        assert_eq!(error.identifier(), IMFINFO_ERROR_OUTPUTS.identifier);
        drop(_outputs);
        let error = run(imfinfo_builtin(vec![
            missing,
            Value::String("not_an_image_format".into()),
        ]))
        .unwrap_err();
        assert_eq!(error.identifier(), IMFINFO_ERROR_FORMAT.identifier);
    }

    #[test]
    fn nontext_path_rejects_before_filesystem_access() {
        let error = run(imfinfo_builtin(vec![Value::Num(42.0)])).unwrap_err();
        assert_eq!(error.identifier(), IMFINFO_ERROR_ARGUMENT.identifier);
    }
}
