use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Duration;

use image::codecs::gif::{GifDecoder, GifEncoder, Repeat};
use image::{AnimationDecoder, Delay, DynamicImage, Frame, ImageFormat, ImageOutputFormat};
use image::{ImageBuffer, Luma, Rgb, Rgba, RgbaImage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LogicalArray, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::image::type_resolvers::imwrite_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "imwrite";

const IMWRITE_INPUTS_IMAGE_FILENAME: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grayscale, truecolor, or RGBA image data.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output image path.",
    },
];

const IMWRITE_INPUTS_INDEXED: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image data.",
    },
    BuiltinParamDescriptor {
        name: "map",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nx3 colormap.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output image path.",
    },
];

const IMWRITE_INPUTS_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Image data.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output image path.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option value.",
    },
];

const IMWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "imwrite(A, filename)",
        inputs: &IMWRITE_INPUTS_IMAGE_FILENAME,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "imwrite(A, filename, fmt)",
        inputs: &IMWRITE_INPUTS_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "imwrite(A, filename, name, value, ...)",
        inputs: &IMWRITE_INPUTS_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "imwrite(X, map, filename, ...)",
        inputs: &IMWRITE_INPUTS_INDEXED,
        outputs: &[],
    },
];

const IMWRITE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_ARGUMENT",
    identifier: Some("RunMat:imwrite:InvalidArgument"),
    when: "Arguments do not match a supported imwrite form.",
    message: "imwrite: invalid argument",
};
const IMWRITE_ERROR_INVALID_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_FILENAME",
    identifier: Some("RunMat:imwrite:InvalidFilename"),
    when: "Filename is missing or empty.",
    message: "imwrite: invalid filename",
};
const IMWRITE_ERROR_INVALID_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_FORMAT",
    identifier: Some("RunMat:imwrite:InvalidFormat"),
    when: "Image format cannot be inferred or is unsupported.",
    message: "imwrite: invalid image format",
};
const IMWRITE_ERROR_INVALID_IMAGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_IMAGE",
    identifier: Some("RunMat:imwrite:InvalidImage"),
    when: "Image data has unsupported type, shape, or values.",
    message: "imwrite: invalid image data",
};
const IMWRITE_ERROR_INVALID_COLORMAP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_COLORMAP",
    identifier: Some("RunMat:imwrite:InvalidColormap"),
    when: "Indexed-image colormap is not an Nx3 numeric array.",
    message: "imwrite: invalid colormap",
};
const IMWRITE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_OPTION",
    identifier: Some("RunMat:imwrite:InvalidOption"),
    when: "Name-value option is malformed or unsupported for the requested format.",
    message: "imwrite: invalid option",
};
const IMWRITE_ERROR_ENCODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.ENCODE",
    identifier: Some("RunMat:imwrite:EncodeError"),
    when: "Image data cannot be encoded.",
    message: "imwrite: encode error",
};
const IMWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.IO",
    identifier: Some("RunMat:imwrite:Io"),
    when: "Image file cannot be read for append or written.",
    message: "imwrite: file I/O error",
};

const IMWRITE_ERRORS: [BuiltinErrorDescriptor; 8] = [
    IMWRITE_ERROR_INVALID_ARGUMENT,
    IMWRITE_ERROR_INVALID_FILENAME,
    IMWRITE_ERROR_INVALID_FORMAT,
    IMWRITE_ERROR_INVALID_IMAGE,
    IMWRITE_ERROR_INVALID_COLORMAP,
    IMWRITE_ERROR_INVALID_OPTION,
    IMWRITE_ERROR_ENCODE,
    IMWRITE_ERROR_IO,
];

pub const IMWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMWRITE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::imwrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imwrite",
    op_kind: GpuOpKind::Custom("image-imwrite"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host image encoder sink; gpuArray inputs are gathered before writing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::imwrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imwrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is not eligible for fusion.",
};

#[runtime_builtin(
    name = "imwrite",
    category = "image/io",
    summary = "Write image data to a file.",
    keywords = "image,write,imwrite,png,jpeg,gif,bmp,tiff",
    sink = true,
    suppress_auto_output = true,
    type_resolver(imwrite_type),
    descriptor(crate::builtins::image::imwrite::IMWRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::image::imwrite"
)]
async fn imwrite_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if let Some(n) = crate::output_count::current_output_count() {
        if n > 0 {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_ARGUMENT,
                "imwrite does not return output arguments",
            ));
        }
    }

    let mut host_args = Vec::with_capacity(args.len());
    for arg in &args {
        host_args.push(gather_if_needed_async(arg).await?);
    }

    let invocation = parse_invocation(&host_args)?;
    let image = materialize_image(
        &invocation.image,
        invocation.map.as_ref(),
        invocation.alpha.as_ref(),
    )?;
    let bytes = encode_image(&image, &invocation).await?;
    runmat_filesystem::write_async(&invocation.path, &bytes)
        .await
        .map_err(|err| {
            imwrite_error_with_detail(
                &IMWRITE_ERROR_IO,
                format!("failed to write '{}': {err}", invocation.path.display()),
            )
        })?;

    Ok(Value::OutputList(Vec::new()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriteMode {
    Overwrite,
    Append,
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
}

#[derive(Clone)]
enum PixelData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

fn parse_invocation(args: &[Value]) -> BuiltinResult<Invocation> {
    if args.len() < 2 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_ARGUMENT,
            "expected image data and filename",
        ));
    }

    let (image, map, filename_index) = if is_string_like(&args[1]) {
        (args[0].clone(), None, 1usize)
    } else {
        if args.len() < 3 {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_ARGUMENT,
                "indexed images require X, map, and filename",
            ));
        }
        (args[0].clone(), Some(args[1].clone()), 2usize)
    };

    let filename = string_arg(
        "filename",
        &args[filename_index],
        &IMWRITE_ERROR_INVALID_FILENAME,
    )?;
    if filename.trim().is_empty() {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_FILENAME,
            "filename must not be empty",
        ));
    }
    let path = PathBuf::from(filename);
    let mut idx = filename_index + 1;

    let mut explicit_format = None;
    if idx < args.len() {
        if let Some(text) = tensor::value_to_string(&args[idx]) {
            if !is_option_name(&text) {
                explicit_format = Some(parse_format_hint(&text)?);
                idx += 1;
            }
        }
    }

    let mut options = ImwriteOptions::default();
    let mut alpha = None;
    while idx < args.len() {
        let name = string_arg("option name", &args[idx], &IMWRITE_ERROR_INVALID_OPTION)?;
        idx += 1;
        if idx >= args.len() {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_OPTION,
                format!("option '{name}' requires a value"),
            ));
        }
        let value = &args[idx];
        idx += 1;

        match canonical_option_name(&name).as_str() {
            "alpha" => alpha = Some(tensor_from_numeric_like(value, "Alpha")?),
            "quality" => {
                let q = numeric_scalar(value, "Quality")?;
                if !q.is_finite() || !(0.0..=100.0).contains(&q) {
                    return Err(imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_OPTION,
                        "Quality must be a scalar from 0 to 100",
                    ));
                }
                options.quality = q.round() as u8;
            }
            "writemode" => {
                let mode = string_arg("WriteMode", value, &IMWRITE_ERROR_INVALID_OPTION)?;
                options.write_mode = match mode.trim().to_ascii_lowercase().as_str() {
                    "overwrite" => WriteMode::Overwrite,
                    "append" => WriteMode::Append,
                    _ => {
                        return Err(imwrite_error_with_detail(
                            &IMWRITE_ERROR_INVALID_OPTION,
                            "WriteMode must be 'overwrite' or 'append'",
                        ))
                    }
                };
            }
            "delaytime" => {
                let delay = numeric_scalar(value, "DelayTime")?;
                if !delay.is_finite() || delay < 0.0 {
                    return Err(imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_OPTION,
                        "DelayTime must be a finite non-negative scalar in seconds",
                    ));
                }
                options.delay_time = Some(delay);
            }
            "loopcount" => {
                let count = numeric_scalar(value, "LoopCount")?;
                if count.is_nan() || count < 0.0 {
                    return Err(imwrite_error_with_detail(
                        &IMWRITE_ERROR_INVALID_OPTION,
                        "LoopCount must be non-negative or Inf",
                    ));
                }
                options.loop_count = Some(count);
            }
            "compression" | "bitdepth" | "mode" | "disposalmethod" | "backgroundcolor"
            | "comment" | "transparentcolor" => {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    format!("option '{name}' is not supported yet"),
                ));
            }
            _ => {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    format!("unsupported option '{name}'"),
                ))
            }
        }
    }

    let format = match explicit_format {
        Some(format) => format,
        None => infer_format_from_path(&path)?,
    };

    Ok(Invocation {
        image,
        map,
        alpha,
        path,
        format,
        options,
    })
}

fn is_string_like(value: &Value) -> bool {
    tensor::value_to_string(value).is_some()
}

fn string_arg(
    label: &str,
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    tensor::value_to_string(value).ok_or_else(|| {
        imwrite_error_with_detail(
            error,
            format!("{label} must be a string scalar or char vector"),
        )
    })
}

fn numeric_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::LogicalArray(a) if a.data.len() == 1 => Ok(if a.data[0] != 0 { 1.0 } else { 0.0 }),
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            format!("{label} must be a numeric scalar"),
        )),
    }
}

fn canonical_option_name(name: &str) -> String {
    name.chars()
        .filter(|ch| !ch.is_whitespace() && *ch != '_' && *ch != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_option_name(name: &str) -> bool {
    matches!(
        canonical_option_name(name).as_str(),
        "alpha"
            | "quality"
            | "writemode"
            | "delaytime"
            | "loopcount"
            | "compression"
            | "bitdepth"
            | "mode"
            | "disposalmethod"
            | "backgroundcolor"
            | "comment"
            | "transparentcolor"
    )
}

fn parse_format_hint(value: &str) -> BuiltinResult<ImageFormat> {
    let label = value.trim().trim_start_matches('.').to_ascii_lowercase();
    if label.is_empty() {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_FORMAT,
            "format hint must not be empty",
        ));
    }
    match label.as_str() {
        "jpg" | "jpeg" | "jpe" => Ok(ImageFormat::Jpeg),
        "png" => Ok(ImageFormat::Png),
        "bmp" => Ok(ImageFormat::Bmp),
        "gif" => Ok(ImageFormat::Gif),
        "tif" | "tiff" => Ok(ImageFormat::Tiff),
        other => ImageFormat::from_extension(other)
            .filter(is_supported_format)
            .ok_or_else(|| {
                imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_FORMAT,
                    format!("unsupported image format '{other}'"),
                )
            }),
    }
}

fn infer_format_from_path(path: &Path) -> BuiltinResult<ImageFormat> {
    ImageFormat::from_path(path)
        .ok()
        .filter(is_supported_format)
        .ok_or_else(|| {
            imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_FORMAT,
                format!(
                    "could not infer supported image format from '{}'",
                    path.display()
                ),
            )
        })
}

fn is_supported_format(format: &ImageFormat) -> bool {
    matches!(
        format,
        ImageFormat::Png
            | ImageFormat::Jpeg
            | ImageFormat::Bmp
            | ImageFormat::Gif
            | ImageFormat::Tiff
    )
}

fn tensor_from_numeric_like(value: &Value, label: &str) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t.clone()),
        Value::LogicalArray(a) => logical_to_tensor(a),
        Value::Num(n) => Tensor::new(vec![*n], vec![1, 1]).map_err(|err| {
            imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, format!("{label}: {err}"))
        }),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|err| {
            imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, format!("{label}: {err}"))
        }),
        Value::Bool(b) => {
            Tensor::new(vec![if *b { 1.0 } else { 0.0 }], vec![1, 1]).map_err(|err| {
                imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, format!("{label}: {err}"))
            })
        }
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("{label} must be numeric or logical"),
        )),
    }
}

fn logical_to_tensor(value: &LogicalArray) -> BuiltinResult<Tensor> {
    let data = value
        .data
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    Tensor::new(data, value.shape.clone())
        .map_err(|err| imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, err))
}

fn materialize_image(
    image: &Value,
    map: Option<&Value>,
    alpha: Option<&Tensor>,
) -> BuiltinResult<MaterializedImage> {
    let tensor = tensor_from_numeric_like(image, "image")?;
    let mut out = if let Some(map_value) = map {
        materialize_indexed_image(&tensor, &tensor_from_numeric_like(map_value, "map")?)?
    } else {
        materialize_direct_image(&tensor)?
    };

    if let Some(alpha) = alpha {
        apply_alpha(&mut out, alpha)?;
    }
    Ok(out)
}

fn image_dimensions(tensor: &Tensor) -> BuiltinResult<(usize, usize, usize)> {
    match tensor.shape.len() {
        0 => Ok((1, 1, 1)),
        1 => Ok((1, tensor.shape[0], 1)),
        2 => Ok((tensor.shape[0], tensor.shape[1], 1)),
        3 if matches!(tensor.shape[2], 1 | 3 | 4) => {
            Ok((tensor.shape[0], tensor.shape[1], tensor.shape[2]))
        }
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "image must be MxN, MxNx3, or MxNx4",
        )),
    }
}

fn materialize_direct_image(tensor: &Tensor) -> BuiltinResult<MaterializedImage> {
    let (rows, cols, channels) = image_dimensions(tensor)?;
    let pixels = rows.checked_mul(cols).ok_or_else(|| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image dimensions overflow")
    })?;
    if tensor.data.len() != pixels * channels {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "image data length does not match shape",
        ));
    }

    let mut data = if tensor.dtype == NumericDType::U16 {
        PixelData::U16(vec![0u16; pixels * channels])
    } else {
        PixelData::U8(vec![0u8; pixels * channels])
    };
    for row in 0..rows {
        for col in 0..cols {
            for channel in 0..channels {
                let src = row + rows * col + pixels * channel;
                let dst = (row * cols + col) * channels + channel;
                match &mut data {
                    PixelData::U8(data) => data[dst] = value_to_u8(tensor.data[src], tensor.dtype),
                    PixelData::U16(data) => {
                        data[dst] = value_to_u16(tensor.data[src], tensor.dtype)
                    }
                }
            }
        }
    }
    Ok(MaterializedImage {
        rows,
        cols,
        channels,
        data,
    })
}

fn materialize_indexed_image(indexed: &Tensor, map: &Tensor) -> BuiltinResult<MaterializedImage> {
    let (rows, cols, channels) = image_dimensions(indexed)?;
    if channels != 1 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image X must be a 2-D array",
        ));
    }
    if map.shape.len() != 2 || map.shape[1] != 3 || map.shape[0] == 0 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_COLORMAP,
            "map must be an Nx3 colormap",
        ));
    }

    let pixels = rows.checked_mul(cols).ok_or_else(|| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image dimensions overflow")
    })?;
    let byte_len = pixels.checked_mul(3).ok_or_else(|| {
        imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, "image dimensions overflow")
    })?;
    let mut data = vec![0u8; byte_len];
    for row in 0..rows {
        for col in 0..cols {
            let pixel = row + rows * col;
            let map_idx = map_index(indexed.data[pixel], indexed.dtype, map.shape[0])?;
            let dst = (row * cols + col) * 3;
            for channel in 0..3 {
                let src = map_idx + map.shape[0] * channel;
                data[dst + channel] = value_to_u8(map.data[src], map.dtype);
            }
        }
    }
    Ok(MaterializedImage {
        rows,
        cols,
        channels: 3,
        data: PixelData::U8(data),
    })
}

fn map_index(value: f64, dtype: NumericDType, map_rows: usize) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            "indexed image values must be finite",
        ));
    }
    let index = if matches!(dtype, NumericDType::U8 | NumericDType::U16) {
        value.round() as isize
    } else {
        value.round() as isize - 1
    };
    if index < 0 || index as usize >= map_rows {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            format!("indexed image value {value} is outside the colormap"),
        ));
    }
    Ok(index as usize)
}

fn value_to_u8(value: f64, dtype: NumericDType) -> u8 {
    let scaled = match dtype {
        NumericDType::U8 => value,
        NumericDType::U16 => value / 257.0,
        NumericDType::F64 | NumericDType::F32 => value.clamp(0.0, 1.0) * 255.0,
    };
    if scaled.is_nan() {
        0
    } else {
        scaled.round().clamp(0.0, 255.0) as u8
    }
}

fn value_to_u16(value: f64, dtype: NumericDType) -> u16 {
    let scaled = match dtype {
        NumericDType::U16 => value,
        NumericDType::U8 => value * 257.0,
        NumericDType::F64 | NumericDType::F32 => value.clamp(0.0, 1.0) * 65535.0,
    };
    if scaled.is_nan() {
        0
    } else {
        scaled.round().clamp(0.0, 65535.0) as u16
    }
}

fn apply_alpha(image: &mut MaterializedImage, alpha: &Tensor) -> BuiltinResult<()> {
    if alpha.shape.len() != 2 || alpha.shape[0] != image.rows || alpha.shape[1] != image.cols {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "Alpha must be an MxN array matching the image dimensions",
        ));
    }
    if alpha.data.len() != image.rows * image.cols {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "Alpha data length does not match shape",
        ));
    }

    let pixels = image.rows * image.cols;
    image.data = match &image.data {
        PixelData::U8(data) => {
            let mut rgba = vec![0u8; pixels * 4];
            for row in 0..image.rows {
                for col in 0..image.cols {
                    let pixel = row * image.cols + col;
                    let alpha_idx = row + image.rows * col;
                    let dst = pixel * 4;
                    match image.channels {
                        1 => {
                            let gray = data[pixel];
                            rgba[dst] = gray;
                            rgba[dst + 1] = gray;
                            rgba[dst + 2] = gray;
                        }
                        3 | 4 => {
                            let src = pixel * image.channels;
                            rgba[dst] = data[src];
                            rgba[dst + 1] = data[src + 1];
                            rgba[dst + 2] = data[src + 2];
                        }
                        _ => unreachable!(),
                    }
                    rgba[dst + 3] = value_to_u8(alpha.data[alpha_idx], alpha.dtype);
                }
            }
            PixelData::U8(rgba)
        }
        PixelData::U16(data) => {
            let mut rgba = vec![0u16; pixels * 4];
            for row in 0..image.rows {
                for col in 0..image.cols {
                    let pixel = row * image.cols + col;
                    let alpha_idx = row + image.rows * col;
                    let dst = pixel * 4;
                    match image.channels {
                        1 => {
                            let gray = data[pixel];
                            rgba[dst] = gray;
                            rgba[dst + 1] = gray;
                            rgba[dst + 2] = gray;
                        }
                        3 | 4 => {
                            let src = pixel * image.channels;
                            rgba[dst] = data[src];
                            rgba[dst + 1] = data[src + 1];
                            rgba[dst + 2] = data[src + 2];
                        }
                        _ => unreachable!(),
                    }
                    rgba[dst + 3] = value_to_u16(alpha.data[alpha_idx], alpha.dtype);
                }
            }
            PixelData::U16(rgba)
        }
    };
    image.channels = 4;
    Ok(())
}

async fn encode_image(
    image: &MaterializedImage,
    invocation: &Invocation,
) -> BuiltinResult<Vec<u8>> {
    if invocation.options.write_mode == WriteMode::Append && invocation.format != ImageFormat::Gif {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "WriteMode 'append' is supported for GIF files only",
        ));
    }

    match invocation.format {
        ImageFormat::Gif => encode_gif(image, invocation).await,
        ImageFormat::Jpeg => {
            if image.channels == 4 {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    "JPEG does not support alpha channels",
                ));
            }
            write_dynamic_image(
                image_to_dynamic(&image_as_8bit(image), false)?,
                ImageOutputFormat::Jpeg(invocation.options.quality),
            )
        }
        ImageFormat::Bmp => {
            if image.channels == 4 {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    "BMP alpha output is not supported",
                ));
            }
            write_dynamic_image(
                image_to_dynamic(&image_as_8bit(image), false)?,
                ImageOutputFormat::Bmp,
            )
        }
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
        imwrite_error_with_detail(
            &IMWRITE_ERROR_ENCODE,
            format!("unable to encode image: {err}"),
        )
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
                imwrite_error_with_detail(
                    &IMWRITE_ERROR_IO,
                    format!(
                        "failed to read GIF for append '{}': {err}",
                        invocation.path.display()
                    ),
                )
            })?;
        existing_repeat = gif_repeat_from_bytes(&existing);
        let decoder = GifDecoder::new(Cursor::new(existing)).map_err(|err| {
            imwrite_error_with_detail(
                &IMWRITE_ERROR_ENCODE,
                format!("failed to decode GIF: {err}"),
            )
        })?;
        for frame in decoder.into_frames() {
            frames.push(frame.map_err(|err| {
                imwrite_error_with_detail(
                    &IMWRITE_ERROR_ENCODE,
                    format!("failed to decode GIF frame: {err}"),
                )
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
                imwrite_error_with_detail(
                    &IMWRITE_ERROR_ENCODE,
                    format!("failed to set GIF repeat: {err}"),
                )
            })?;
        }
        for frame in frames {
            encoder.encode_frame(frame).map_err(|err| {
                imwrite_error_with_detail(
                    &IMWRITE_ERROR_ENCODE,
                    format!("failed to encode GIF frame: {err}"),
                )
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
    if loop_count.is_infinite() {
        return Ok(Repeat::Infinite);
    }
    let rounded = loop_count.round();
    if (rounded - loop_count).abs() > 1e-6 || rounded > u16::MAX as f64 {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "LoopCount must be an integer between 0 and 65535, or Inf",
        ));
    }
    Ok(Repeat::Finite(rounded as u16))
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
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use image::io::Reader as ImageReader;
    use std::fs;
    use tempfile::tempdir;

    fn tensor(data: Vec<f64>, shape: Vec<usize>, dtype: NumericDType) -> Tensor {
        Tensor::new_with_dtype(data, shape, dtype).expect("tensor")
    }

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(imwrite_builtin(args))
    }

    #[test]
    fn writes_png_rgb_and_round_trips_layout() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rgb.png");
        let rgb = tensor(
            vec![255.0, 0.0, 0.0, 0.0, 0.0, 255.0],
            vec![1, 2, 3],
            NumericDType::U8,
        );

        call(vec![
            Value::Tensor(rgb),
            Value::from(path.to_string_lossy().as_ref()),
        ])
        .unwrap();

        let decoded = ImageReader::open(&path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(decoded.get_pixel(1, 0).0, [0, 0, 255]);
    }

    #[test]
    fn writes_png_alpha_option() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("alpha.png");
        let image = tensor(vec![1.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::F64);
        let alpha = tensor(vec![0.5], vec![1, 1], NumericDType::F64);

        call(vec![
            Value::Tensor(image),
            Value::from(path.to_string_lossy().as_ref()),
            Value::from("Alpha"),
            Value::Tensor(alpha),
        ])
        .unwrap();

        let decoded = ImageReader::open(&path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0, 128]);
    }

    #[test]
    fn writes_uint16_png_without_downcasting() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("gray16.png");
        let image = tensor(
            vec![0.0, 65535.0, 12345.0, 40000.0],
            vec![2, 2],
            NumericDType::U16,
        );

        call(vec![
            Value::Tensor(image),
            Value::from(path.to_string_lossy().as_ref()),
        ])
        .unwrap();

        let decoded = ImageReader::open(&path).unwrap().decode().unwrap();
        let gray = decoded.as_luma16().expect("expected 16-bit grayscale PNG");
        assert_eq!(gray.dimensions(), (2, 2));
        assert_eq!(gray.get_pixel(0, 0).0, [0]);
        assert_eq!(gray.get_pixel(0, 1).0, [65535]);
        assert_eq!(gray.get_pixel(1, 0).0, [12345]);
        assert_eq!(gray.get_pixel(1, 1).0, [40000]);
    }

    #[test]
    fn writes_indexed_gif_with_zero_based_uint8_indices() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("indexed.gif");
        let x = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::U8);
        let map = tensor(
            vec![255.0, 0.0, 0.0, 0.0, 0.0, 255.0],
            vec![2, 3],
            NumericDType::U8,
        );

        call(vec![
            Value::Tensor(x),
            Value::Tensor(map),
            Value::from(path.to_string_lossy().as_ref()),
        ])
        .unwrap();

        let decoded = ImageReader::open(&path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(decoded.get_pixel(1, 0).0, [0, 0, 255]);
    }

    #[test]
    fn appends_gif_frame() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("animated.gif");
        let first = tensor(vec![1.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::F64);
        let second = tensor(vec![0.0, 1.0, 0.0], vec![1, 1, 3], NumericDType::F64);

        call(vec![
            Value::Tensor(first),
            Value::from(path.to_string_lossy().as_ref()),
            Value::from("LoopCount"),
            Value::Num(f64::INFINITY),
            Value::from("DelayTime"),
            Value::Num(0.25),
        ])
        .unwrap();
        call(vec![
            Value::Tensor(second),
            Value::from(path.to_string_lossy().as_ref()),
            Value::from("WriteMode"),
            Value::from("append"),
            Value::from("DelayTime"),
            Value::Num(0.25),
        ])
        .unwrap();

        let bytes = fs::read(&path).unwrap();
        assert!(matches!(
            gif_repeat_from_bytes(&bytes),
            Some(Repeat::Infinite)
        ));
        let decoder = GifDecoder::new(Cursor::new(bytes)).unwrap();
        let frames = decoder.into_frames().collect_frames().unwrap();
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn rejects_alpha_for_jpeg() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.jpg");
        let image = tensor(vec![1.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::F64);
        let alpha = tensor(vec![1.0], vec![1, 1], NumericDType::F64);

        let err = call(vec![
            Value::Tensor(image),
            Value::from(path.to_string_lossy().as_ref()),
            Value::from("Alpha"),
            Value::Tensor(alpha),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:imwrite:InvalidOption"));
    }

    #[test]
    fn descriptor_has_stable_errors() {
        let codes: Vec<&str> = IMWRITE_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.IMWRITE.INVALID_IMAGE"));
        assert!(codes.contains(&"RM.IMWRITE.ENCODE"));
    }
}
