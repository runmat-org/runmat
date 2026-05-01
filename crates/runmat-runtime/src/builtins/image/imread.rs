use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Duration;

use image::io::Reader as ImageReader;
use image::{DynamicImage, ImageFormat};
use runmat_builtins::{NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;
use url::Url;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::{map_control_flow_with_builtin, tensor};
use crate::builtins::image::type_resolvers::imread_type;
use crate::builtins::io::http::transport::{
    self, HttpMethod, HttpRequest, TransportError, TransportErrorKind,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "imread";
const DEFAULT_TIMEOUT_SECONDS: f64 = 60.0;
const DEFAULT_USER_AGENT: &str = "RunMat imread/0.0";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::imread")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imread",
    op_kind: GpuOpKind::Custom("image-read"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only image I/O and CPU decoding. Decoded tensors are host-resident; use gpuArray after import for GPU work.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::imread")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imread",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; image loading performs file or network I/O and CPU decoding.",
};

fn imread_error(identifier: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(identifier)
        .build()
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "imread",
    category = "image/io",
    summary = "Read an image from a file path or HTTP(S) URL into a MATLAB-compatible array.",
    keywords = "imread,image,read,file,jpeg,jpg,png,bmp,gif,tiff,webp,url",
    accel = "sink",
    type_resolver(imread_type),
    builtin_path = "crate::builtins::image::imread"
)]
async fn imread_builtin(source: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let source = gather_if_needed_async(&source).await.map_err(map_flow)?;
    let mut gathered_rest = Vec::with_capacity(rest.len());
    for arg in &rest {
        gathered_rest.push(gather_if_needed_async(arg).await.map_err(map_flow)?);
    }

    let source = string_arg("filename", &source)?;
    if source.is_empty() {
        return Err(imread_error(
            "RunMat:imread:InvalidFilename",
            "imread: filename must not be empty",
        ));
    }

    let format_hint = match gathered_rest.as_slice() {
        [] => None,
        [format] => Some(parse_format_hint(&string_arg("format", format)?)?),
        _ => {
            return Err(imread_error(
                "RunMat:imread:TooManyInputs",
                "imread: too many input arguments",
            ))
        }
    };

    let bytes = read_source_bytes(&source).await?;
    let decoded = decode_image_bytes(&bytes, format_hint)?;
    let materialized = materialize_image(decoded)?;

    match crate::output_count::current_output_count() {
        None => Ok(Value::Tensor(materialized.image)),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![Value::Tensor(materialized.image)])),
        Some(2) => Ok(Value::OutputList(vec![
            Value::Tensor(materialized.image),
            empty_tensor_value()?,
        ])),
        Some(3) => Ok(Value::OutputList(vec![
            Value::Tensor(materialized.image),
            empty_tensor_value()?,
            materialized
                .alpha
                .map(Value::Tensor)
                .unwrap_or(empty_tensor_value()?),
        ])),
        Some(_) => Err(imread_error(
            "RunMat:imread:TooManyOutputs",
            "imread: too many output arguments",
        )),
    }
}

fn string_arg(label: &str, value: &Value) -> BuiltinResult<String> {
    tensor::value_to_string(value).ok_or_else(|| {
        imread_error(
            "RunMat:imread:InvalidArgument",
            format!("imread: {label} must be a string scalar or character vector"),
        )
    })
}

fn parse_format_hint(value: &str) -> BuiltinResult<ImageFormat> {
    let label = value.trim().trim_start_matches('.').to_ascii_lowercase();
    if label.is_empty() {
        return Err(imread_error(
            "RunMat:imread:InvalidFormat",
            "imread: format hint must not be empty",
        ));
    }
    let format = match label.as_str() {
        "jpg" | "jpeg" | "jpe" => ImageFormat::Jpeg,
        "tif" | "tiff" => ImageFormat::Tiff,
        "png" => ImageFormat::Png,
        "bmp" => ImageFormat::Bmp,
        "gif" => ImageFormat::Gif,
        "webp" => ImageFormat::WebP,
        "ico" => ImageFormat::Ico,
        other => ImageFormat::from_extension(other).ok_or_else(|| {
            imread_error(
                "RunMat:imread:UnsupportedFormat",
                format!("imread: unsupported image format '{other}'"),
            )
        })?,
    };
    Ok(format)
}

async fn read_source_bytes(source: &str) -> BuiltinResult<Vec<u8>> {
    if let Ok(url) = Url::parse(source) {
        let scheme = url.scheme();
        // A single-letter scheme is a Windows drive letter (e.g. "C:/..."), not a URL.
        if scheme.len() > 1 {
            return match scheme {
                "http" | "https" => read_url_bytes(url).await,
                "file" => {
                    let path = file_url_to_path(&url)?;
                    read_local_path(&path).await
                }
                _ => Err(imread_error(
                    "RunMat:imread:UnsupportedScheme",
                    format!("imread: unsupported URL scheme '{scheme}'"),
                )),
            };
        }
    }

    read_local_path(Path::new(source)).await
}

async fn read_local_path(path: &Path) -> BuiltinResult<Vec<u8>> {
    runmat_filesystem::read_async(path).await.map_err(|err| {
        imread_error(
            "RunMat:imread:FileReadError",
            format!("imread: unable to read '{}': {err}", path.display()),
        )
    })
}

fn file_url_to_path(url: &Url) -> BuiltinResult<PathBuf> {
    if let Some(host) = url.host_str() {
        if !host.is_empty() && !host.eq_ignore_ascii_case("localhost") {
            return Err(imread_error(
                "RunMat:imread:InvalidFileUrl",
                format!("imread: file URL host '{host}' is not local"),
            ));
        }
    }

    let decoded = percent_decode_url_path(url.path())?;

    #[cfg(windows)]
    {
        let path =
            if decoded.len() >= 3 && decoded.as_bytes()[0] == b'/' && decoded.as_bytes()[2] == b':'
            {
                &decoded[1..]
            } else {
                decoded.as_str()
            };
        return Ok(PathBuf::from(path));
    }

    #[cfg(not(windows))]
    {
        Ok(PathBuf::from(decoded))
    }
}

fn percent_decode_url_path(input: &str) -> BuiltinResult<String> {
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return Err(imread_error(
                    "RunMat:imread:InvalidFileUrl",
                    "imread: invalid percent escape in file URL",
                ));
            }
            let hi = hex_value(bytes[index + 1]).ok_or_else(|| {
                imread_error(
                    "RunMat:imread:InvalidFileUrl",
                    "imread: invalid percent escape in file URL",
                )
            })?;
            let lo = hex_value(bytes[index + 2]).ok_or_else(|| {
                imread_error(
                    "RunMat:imread:InvalidFileUrl",
                    "imread: invalid percent escape in file URL",
                )
            })?;
            output.push((hi << 4) | lo);
            index += 3;
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }

    String::from_utf8(output).map_err(|err| {
        imread_error(
            "RunMat:imread:InvalidFileUrl",
            format!("imread: file URL path is not valid UTF-8: {err}"),
        )
    })
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

async fn read_url_bytes(url: Url) -> BuiltinResult<Vec<u8>> {
    let request = HttpRequest {
        url,
        method: HttpMethod::Get,
        headers: vec![(
            "Accept".to_string(),
            "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8".to_string(),
        )],
        body: None,
        timeout: Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS),
        user_agent: DEFAULT_USER_AGENT.to_string(),
    };
    let response = transport::send_request(&request).map_err(imread_transport_error)?;
    Ok(response.body)
}

fn imread_transport_error(err: TransportError) -> RuntimeError {
    let identifier = match &err.kind {
        TransportErrorKind::Timeout => "RunMat:imread:Timeout",
        TransportErrorKind::Connect => "RunMat:imread:NetworkError",
        TransportErrorKind::Status(_) => "RunMat:imread:HttpStatus",
        TransportErrorKind::InvalidHeader(_) => "RunMat:imread:InvalidHeader",
        TransportErrorKind::Other => "RunMat:imread:NetworkError",
    };
    let message = err.message_with_prefix(BUILTIN_NAME);
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(identifier)
        .with_source(err)
        .build()
}

fn decode_image_bytes(bytes: &[u8], format: Option<ImageFormat>) -> BuiltinResult<DynamicImage> {
    let reader = if let Some(format) = format {
        ImageReader::with_format(Cursor::new(bytes), format)
    } else {
        ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|err| {
                imread_error(
                    "RunMat:imread:DecodeError",
                    format!("imread: unable to detect image format: {err}"),
                )
            })?
    };
    reader.decode().map_err(|err| {
        imread_error(
            "RunMat:imread:DecodeError",
            format!("imread: unable to decode image: {err}"),
        )
    })
}

struct MaterializedImage {
    image: Tensor,
    alpha: Option<Tensor>,
}

fn materialize_image(image: DynamicImage) -> BuiltinResult<MaterializedImage> {
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
    Tensor::new_with_dtype(data, shape, dtype)
        .map_err(|err| imread_error("RunMat:imread:ShapeError", format!("imread: {err}")))
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
    Tensor::new_with_dtype(data, vec![rows, cols], dtype)
        .map_err(|err| imread_error("RunMat:imread:ShapeError", format!("imread: {err}")))
}

fn empty_tensor_value() -> BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|err| imread_error("RunMat:imread:ShapeError", format!("imread: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, ImageOutputFormat, Luma, Rgb, RgbImage, Rgba, RgbaImage};
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Arc;

    fn encode_image(image: DynamicImage, format: ImageOutputFormat) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        image.write_to(&mut cursor, format).expect("encode image");
        cursor.into_inner()
    }

    fn rgb_png() -> Vec<u8> {
        let image = RgbImage::from_fn(2, 2, |x, y| match (x, y) {
            (0, 0) => Rgb([10, 20, 30]),
            (1, 0) => Rgb([40, 50, 60]),
            (0, 1) => Rgb([70, 80, 90]),
            (1, 1) => Rgb([100, 110, 120]),
            _ => unreachable!(),
        });
        encode_image(DynamicImage::ImageRgb8(image), ImageOutputFormat::Png)
    }

    fn rgba_png() -> Vec<u8> {
        let image = RgbaImage::from_fn(2, 1, |x, _| match x {
            0 => Rgba([1, 2, 3, 4]),
            1 => Rgba([5, 6, 7, 8]),
            _ => unreachable!(),
        });
        encode_image(DynamicImage::ImageRgba8(image), ImageOutputFormat::Png)
    }

    fn run_imread(bytes: &[u8], extension: &str, rest: Vec<Value>) -> Value {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(format!("image.{extension}"));
        std::fs::write(&path, bytes).expect("write image");
        futures::executor::block_on(imread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            rest,
        ))
        .expect("imread")
    }

    #[test]
    fn imread_decodes_rgb_png_as_column_major_truecolor_uint8() {
        let result = run_imread(&rgb_png(), "png", Vec::new());
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor, got {result:?}");
        };
        assert_eq!(tensor.shape, vec![2, 2, 3]);
        assert_eq!(tensor.dtype, NumericDType::U8);
        assert_eq!(
            tensor.data,
            vec![10.0, 70.0, 40.0, 100.0, 20.0, 80.0, 50.0, 110.0, 30.0, 90.0, 60.0, 120.0]
        );
    }

    #[test]
    fn imread_returns_alpha_as_third_output_for_rgba_png() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("alpha.png");
        std::fs::write(&path, rgba_png()).expect("write image");
        let _guard = crate::output_count::push_output_count(Some(3));
        let result = futures::executor::block_on(imread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            Vec::new(),
        ))
        .expect("imread");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list, got {result:?}");
        };
        assert_eq!(outputs.len(), 3);
        match &outputs[0] {
            Value::Tensor(rgb) => {
                assert_eq!(rgb.shape, vec![1, 2, 3]);
                assert_eq!(rgb.dtype, NumericDType::U8);
                assert_eq!(rgb.data, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
            }
            other => panic!("expected rgb tensor, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(map) => assert_eq!(map.shape, vec![0, 0]),
            other => panic!("expected empty map tensor, got {other:?}"),
        }
        match &outputs[2] {
            Value::Tensor(alpha) => {
                assert_eq!(alpha.shape, vec![1, 2]);
                assert_eq!(alpha.dtype, NumericDType::U8);
                assert_eq!(alpha.data, vec![4.0, 8.0]);
            }
            other => panic!("expected alpha tensor, got {other:?}"),
        }
    }

    #[test]
    fn imread_reads_local_file_path() {
        let result = run_imread(&rgb_png(), "png", Vec::new());
        assert!(matches!(result, Value::Tensor(_)));
    }

    #[test]
    fn imread_windows_drive_letter_path_is_not_treated_as_url_scheme() {
        // Url::parse("C:/foo.png") succeeds with scheme "c", which must not be
        // treated as an unsupported URL scheme — it should fall through to the
        // local path reader and produce a file-not-found error, not a scheme error.
        let err = futures::executor::block_on(imread_builtin(
            Value::from("C:/nonexistent/photo.png"),
            Vec::new(),
        ))
        .expect_err("expected error for missing file");
        assert_ne!(
            err.identifier(),
            Some("RunMat:imread:UnsupportedScheme"),
            "drive-letter path incorrectly rejected as unsupported URL scheme"
        );
    }

    #[test]
    fn imread_respects_explicit_format_hint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("image-no-extension");
        std::fs::write(&path, rgb_png()).expect("write image");
        let result = futures::executor::block_on(imread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::from("png")],
        ))
        .expect("imread");
        assert!(matches!(result, Value::Tensor(_)));
    }

    #[test]
    fn imread_rejects_unknown_format_hint() {
        let err = futures::executor::block_on(imread_builtin(
            Value::from("missing"),
            vec![Value::from("not-a-format")],
        ))
        .expect_err("expected error");
        assert_eq!(err.identifier(), Some("RunMat:imread:UnsupportedFormat"));
    }

    #[test]
    fn imread_dispatcher_reports_builtin_error_directly() {
        let url = spawn_repeating_server(
            2,
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_vec(),
        );
        let err = crate::call_builtin("imread", &[Value::from(format!("{url}/missing.jpg"))])
            .expect_err("expected 404");
        assert_eq!(err.identifier(), Some("RunMat:imread:HttpStatus"));
        assert!(err.message().contains("HTTP status 404"));
        assert!(!err.message().contains("No matching overload"));
    }

    #[test]
    fn imread_materializes_multi_outputs_with_empty_colormap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rgb.png");
        std::fs::write(&path, rgb_png()).expect("write image");
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = futures::executor::block_on(imread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            Vec::new(),
        ))
        .expect("imread");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list, got {result:?}");
        };
        assert_eq!(outputs.len(), 2);
        assert!(matches!(&outputs[0], Value::Tensor(_)));
        match &outputs[1] {
            Value::Tensor(map) => assert_eq!(map.shape, vec![0, 0]),
            other => panic!("expected map tensor, got {other:?}"),
        }
    }

    #[test]
    fn imread_decodes_16_bit_grayscale() {
        let image: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_fn(2, 2, |x, y| {
            let value = match (x, y) {
                (0, 0) => 1,
                (1, 0) => 2,
                (0, 1) => 300,
                (1, 1) => 65535,
                _ => unreachable!(),
            };
            Luma([value])
        });
        let bytes = encode_image(DynamicImage::ImageLuma16(image), ImageOutputFormat::Png);
        let result = run_imread(&bytes, "png", Vec::new());
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor, got {result:?}");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, NumericDType::U16);
        assert_eq!(tensor.data, vec![1.0, 300.0, 2.0, 65535.0]);
    }

    #[test]
    fn imread_fetches_http_url() {
        let body = rgb_png();
        let response = http_response(200, "OK", "image/png", &body);
        let url = spawn_server(response);
        let result = futures::executor::block_on(imread_builtin(
            Value::from(format!("{url}/image.png")),
            Vec::new(),
        ))
        .expect("imread");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor, got {result:?}");
        };
        assert_eq!(tensor.shape, vec![2, 2, 3]);
        assert_eq!(tensor.dtype, NumericDType::U8);
    }

    fn http_response(status: u16, reason: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(body);
        response
    }

    fn spawn_server(response: Vec<u8>) -> String {
        spawn_repeating_server(1, response)
    }

    fn spawn_repeating_server(limit: usize, response: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let response = Arc::new(response);
        std::thread::spawn(move || {
            for stream in listener.incoming().take(limit) {
                let Ok(mut stream) = stream else {
                    continue;
                };
                write_response(&mut stream, &response);
            }
        });
        format!("http://{addr}")
    }

    fn write_response(stream: &mut TcpStream, response: &[u8]) {
        let mut buffer = [0u8; 1024];
        let _ = stream.read(&mut buffer);
        stream.write_all(response).expect("write response");
        stream.flush().expect("flush response");
    }
}
