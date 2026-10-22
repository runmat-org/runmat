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
use runmat_value::{CharArray, Value};

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
    super::checked_text::scalar_text(value)
        .map_err(|source| {
            let mut error = info_error(&IMFINFO_ERROR_ARGUMENT, source.to_string());
            error.source = Some(Box::new(source));
            error
        })?
        .ok_or_else(|| {
            info_error(
                &IMFINFO_ERROR_ARGUMENT,
                format!("{arg} must be a string scalar or character vector"),
            )
        })
}

fn char_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[runtime_builtin(
    name = "imfinfo",
    binding_variant = "default",
    builtin_path = "crate::builtins::image::imfinfo"
)]
async fn imfinfo_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
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
            let extension = hint.trim().trim_start_matches('.');
            image::ImageFormat::from_extension(extension.to_ascii_lowercase())
                .map(|format| (format, extension.to_string()))
                .ok_or_else(|| info_error(&IMFINFO_ERROR_FORMAT, "unsupported image format hint"))
        })
        .transpose()?;
    let requested_path = PathBuf::from(
        expand_user_path(filename.trim(), "imfinfo")
            .map_err(|error| info_error(&IMFINFO_ERROR_ARGUMENT, error))?,
    );
    let (path, bytes) = match vfs::read_async(&requested_path).await {
        Ok(bytes) => (requested_path, bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && explicit_format.is_some() => {
            let suffix = &explicit_format.as_ref().expect("checked above").1;
            let mut candidate = requested_path.as_os_str().to_os_string();
            candidate.push(".");
            candidate.push(suffix);
            let candidate = PathBuf::from(candidate);
            let bytes = vfs::read_async(&candidate)
                .await
                .map_err(|err| info_error_with_source(&IMFINFO_ERROR_IO, err))?;
            (candidate, bytes)
        }
        Err(error) => return Err(info_error_with_source(&IMFINFO_ERROR_IO, error)),
    };
    let mut reader = ImageReader::new(Cursor::new(bytes.as_slice()))
        .with_guessed_format()
        .map_err(|err| info_error_with_source(&IMFINFO_ERROR_DECODE, err))?;
    if let Some((format, _)) = explicit_format {
        reader.set_format(format);
    }
    let format = reader.format();
    let image = reader
        .decode()
        .map_err(|err| info_error_with_source(&IMFINFO_ERROR_DECODE, err))?;
    let color = image.color();
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.insert("Filename", char_value(&path_to_string(&path)));
    st.insert("FileModDate", char_value(""));
    st.insert("FileSize", Value::Num(bytes.len() as f64));
    st.insert("Format", char_value(format_name(format, &path)));
    st.insert("FormatVersion", char_value(""));
    st.insert("Width", Value::Num(image.width() as f64));
    st.insert("Height", Value::Num(image.height() as f64));
    st.insert("BitDepth", Value::Num(color.bits_per_pixel() as f64));
    st.insert("ColorType", char_value(color_type_name(color)));
    Ok(Value::Struct(st.finish()))
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

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn imfinfo_reports_png_dimensions() {
        let _guard = runmat_filesystem::provider_override_lock();
        let root = tempfile::TempDir::new().unwrap();
        let path = root.path().join("image.png");
        let image = image::RgbImage::from_pixel(3, 2, image::Rgb([1, 2, 3]));
        image.save(&path).unwrap();
        let value = run(imfinfo_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                path_to_string(&path).into(),
            )]),
        ))
        .unwrap();
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
    fn supported_format_adds_extension_only_when_original_path_is_missing() {
        let _guard = runmat_filesystem::provider_override_lock();
        let root = tempfile::TempDir::new().unwrap();
        let bare = root.path().join("sample");
        let with_extension = root.path().join("sample.png");
        let image = image::RgbImage::from_pixel(3, 2, image::Rgb([1, 2, 3]));
        image.save(&with_extension).unwrap();
        let info = run(imfinfo_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(path_to_string(&bare).into()),
                Value::String("png".into()),
            ]),
        ))
        .unwrap();
        let Value::Struct(fields) = info else {
            panic!("expected metadata struct")
        };
        assert_eq!(
            fields.fields.get("Filename"),
            Some(&char_value(&path_to_string(&with_extension)))
        );
        assert_eq!(fields.fields.get("Width"), Some(&Value::Num(3.0)));

        std::fs::copy(&with_extension, &bare).unwrap();
        let info = run(imfinfo_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(path_to_string(&bare).into()),
                Value::String("png".into()),
            ]),
        ))
        .unwrap();
        let Value::Struct(fields) = info else {
            panic!("expected metadata struct")
        };
        assert_eq!(
            fields.fields.get("Filename"),
            Some(&char_value(&path_to_string(&bare)))
        );

        let upper = root.path().join("upper.PNG");
        std::fs::copy(&with_extension, &upper).unwrap();
        let info = run(imfinfo_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(path_to_string(&root.path().join("upper")).into()),
                Value::String("PNG".into()),
            ]),
        ))
        .unwrap();
        let Value::Struct(fields) = info else {
            panic!("expected metadata struct")
        };
        assert_eq!(
            fields.fields.get("Filename"),
            Some(&char_value(&path_to_string(&upper)))
        );
    }

    #[test]
    fn invalid_output_and_format_precede_missing_file_read() {
        let root = tempfile::TempDir::new().unwrap();
        let missing = Value::String(path_to_string(&root.path().join("missing.png")).into());
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = run(imfinfo_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![missing.clone()]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), IMFINFO_ERROR_OUTPUTS.identifier);
        drop(_outputs);
        let error = run(imfinfo_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                missing,
                Value::String("not_an_image_format".into()),
            ]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), IMFINFO_ERROR_FORMAT.identifier);
    }

    #[test]
    fn nontext_path_rejects_before_filesystem_access() {
        let error = run(imfinfo_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(42.0)]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), IMFINFO_ERROR_ARGUMENT.identifier);
    }
}
