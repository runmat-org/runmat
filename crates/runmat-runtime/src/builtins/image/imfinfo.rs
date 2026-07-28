//! MATLAB-compatible `imfinfo` metadata reader.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::io::Reader as ImageReader;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, CharArray, StructValue,
    Value,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

fn info_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("imfinfo").build()
}

const IMFINFO_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Image file to inspect.",
    },
    BuiltinParamDescriptor {
        name: "fmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional image format hint.",
    },
];
const IMFINFO_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Image metadata struct.",
}];
const IMFINFO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = imfinfo(filename, fmt)",
    inputs: &IMFINFO_INPUTS,
    outputs: &IMFINFO_OUTPUTS,
}];
pub const IMFINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMFINFO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

async fn gather_args(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(|err| info_error(format!("imfinfo: {}", err.message())))?,
        );
    }
    Ok(out)
}

fn scalar_text(value: &Value, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(info_error(format!(
            "imfinfo: {arg} must be a string scalar or character vector"
        ))),
    }
}

fn char_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[runtime_builtin(
    name = "imfinfo",
    category = "image",
    summary = "Read image file metadata.",
    keywords = "imfinfo,image,metadata,width,height,format",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::image::imfinfo::IMFINFO_DESCRIPTOR),
    builtin_path = "crate::builtins::image::imfinfo"
)]
async fn imfinfo_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args(&args).await?;
    if args.is_empty() || args.len() > 2 {
        return Err(info_error(
            "imfinfo: filename and optional format are expected",
        ));
    }
    let filename = scalar_text(&args[0], "filename")?;
    let path = PathBuf::from(expand_user_path(filename.trim(), "imfinfo").map_err(info_error)?);
    let bytes = vfs::read_async(&path)
        .await
        .map_err(|err| info_error(format!("imfinfo: {err}")))?;
    let mut reader = ImageReader::new(Cursor::new(bytes.as_slice()))
        .with_guessed_format()
        .map_err(|err| info_error(format!("imfinfo: {err}")))?;
    if let Some(fmt) = args.get(1) {
        let hint = scalar_text(fmt, "fmt")?;
        if let Some(format) = image::ImageFormat::from_extension(hint.trim_start_matches('.')) {
            reader.set_format(format);
        }
    }
    let format = reader.format();
    let image = reader
        .decode()
        .map_err(|err| info_error(format!("imfinfo: {err}")))?;
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
        let path = std::env::temp_dir().join("runmat_imfinfo_test.png");
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
        let _ = std::fs::remove_file(path);
    }
}
