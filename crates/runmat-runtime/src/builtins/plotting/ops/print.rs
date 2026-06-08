//! MATLAB-compatible `print` builtin for exporting figures.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Type,
    Value,
};
use runmat_filesystem::OpenOptions;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::op_common::handles::handle_from_scalar;
use super::state::{current_figure_handle, FigureHandle};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::print")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "print",
    op_kind: GpuOpKind::Custom("figure-export"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Exports the current figure through the plotting renderer. String/options arguments are gathered; figure content may still render through the shared WGPU plotting path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::print")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "print",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "print performs figure export I/O and terminates fusion graphs.",
};

const BUILTIN_NAME: &str = "print";
const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;
const DEFAULT_DPI: u32 = 150;
const MAX_EXPORT_DIMENSION: u32 = 8192;
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

const PRINT_OUTPUT_OK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the export completed.",
}];
const PRINT_INPUTS_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional figure handle, filename, device token such as '-dpng', and resolution token such as '-r300'.",
}];
const PRINT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "ok = print(filename, '-dpng')",
        inputs: &PRINT_INPUTS_ARGS,
        outputs: &PRINT_OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = print(filename, '-dpng', '-r300')",
        inputs: &PRINT_INPUTS_ARGS,
        outputs: &PRINT_OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = print(fig, filename, '-dpng')",
        inputs: &PRINT_INPUTS_ARGS,
        outputs: &PRINT_OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = print('-dpng', filename)",
        inputs: &PRINT_INPUTS_ARGS,
        outputs: &PRINT_OUTPUT_OK,
    },
];

const PRINT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.INVALID_INPUT",
    identifier: Some("RunMat:print:InvalidInput"),
    when: "Arguments are missing, malformed, or cannot be interpreted as a supported print form.",
    message: "print: invalid input arguments",
};
const PRINT_ERROR_UNSUPPORTED_DEVICE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.UNSUPPORTED_DEVICE",
    identifier: Some("RunMat:print:UnsupportedDevice"),
    when: "The requested output device is not supported by the active exporter.",
    message: "print: unsupported output device",
};
const PRINT_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.INVALID_OPTION",
    identifier: Some("RunMat:print:InvalidOption"),
    when: "A print option token is unsupported or invalid.",
    message: "print: invalid option",
};
const PRINT_ERROR_RENDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.RENDER",
    identifier: Some("RunMat:print:RenderFailed"),
    when: "The figure renderer fails while serializing the figure.",
    message: "print: figure export failed",
};
const PRINT_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.IO",
    identifier: Some("RunMat:print:IoFailure"),
    when: "The exported bytes cannot be written to the target file.",
    message: "print: file I/O failed",
};
const PRINT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion fails.",
    message: "print: internal error",
};
const PRINT_ERRORS: [BuiltinErrorDescriptor; 6] = [
    PRINT_ERROR_INVALID_INPUT,
    PRINT_ERROR_UNSUPPORTED_DEVICE,
    PRINT_ERROR_INVALID_OPTION,
    PRINT_ERROR_RENDER,
    PRINT_ERROR_IO,
    PRINT_ERROR_INTERNAL,
];

pub const PRINT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PRINT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PRINT_ERRORS,
};

pub fn print_type(_args: &[Type], _context: &runmat_builtins::ResolveContext) -> Type {
    Type::Bool
}

#[runtime_builtin(
    name = "print",
    category = "plotting",
    summary = "Export a figure to an image file.",
    keywords = "print,plotting,figure export,png,save figure",
    sink = true,
    suppress_auto_output = true,
    accel = "metadata",
    type_resolver(print_type),
    descriptor(crate::builtins::plotting::print::PRINT_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::print"
)]
pub async fn print_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    let args = gather_values(&args).await?;
    let request = parse_print_args(&args)?;
    let path = request.output_path()?;
    let bytes = render_png(request.figure, request.width, request.height).await?;
    write_bytes(&path, &bytes).await?;
    Ok(true)
}

async fn gather_values(values: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrintDevice {
    Png,
}

#[derive(Debug, Clone)]
struct PrintRequest {
    figure: FigureHandle,
    filename: String,
    device: PrintDevice,
    width: u32,
    height: u32,
}

impl PrintRequest {
    fn output_path(&self) -> BuiltinResult<PathBuf> {
        let path = Path::new(&self.filename);
        match self.device {
            PrintDevice::Png if has_extension(path, "png") => Ok(path.to_path_buf()),
            PrintDevice::Png => {
                let Some(file_name) = path.file_name() else {
                    return Err(print_error_with_detail(
                        &PRINT_ERROR_INVALID_INPUT,
                        "filename must include a file name before appending '.png'",
                    ));
                };
                let mut file_name = OsString::from(file_name);
                file_name.push(".png");
                Ok(path.with_file_name(file_name))
            }
        }
    }
}

fn parse_print_args(args: &[Value]) -> BuiltinResult<PrintRequest> {
    if args.is_empty() {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename is required",
        ));
    }

    let mut idx = 0usize;
    let mut figure = current_figure_handle();
    if let Some(handle) = figure_handle_arg(&args[0])? {
        figure = handle;
        idx = 1;
    }

    let mut filename: Option<String> = None;
    let mut device: Option<PrintDevice> = None;
    let mut dpi = DEFAULT_DPI;

    while idx < args.len() {
        let token = string_arg(&args[idx])?;
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_INPUT,
                "empty string arguments are not valid print options",
            ));
        }

        if let Some(option) = trimmed.strip_prefix('-') {
            parse_option(option, &mut device, &mut dpi)?;
        } else if filename.is_none() {
            filename = Some(trimmed.to_string());
        } else {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_INPUT,
                format!("unexpected extra filename argument '{trimmed}'"),
            ));
        }
        idx += 1;
    }

    let filename = filename.ok_or_else(|| {
        print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename is required for RunMat figure export",
        )
    })?;
    if filename.contains('\0') {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename must not contain NUL bytes",
        ));
    }

    let inferred = infer_device_from_filename(&filename);
    let device = match (device, inferred) {
        (Some(device), _) => device,
        (None, Some(device)) => device,
        (None, None) => PrintDevice::Png,
    };
    let (width, height) = dimensions_for_dpi(dpi)?;
    Ok(PrintRequest {
        figure,
        filename,
        device,
        width,
        height,
    })
}

fn figure_handle_arg(value: &Value) -> BuiltinResult<Option<FigureHandle>> {
    match value {
        Value::Num(v) => Ok(Some(handle_from_scalar(*v, BUILTIN_NAME)?)),
        Value::Int(i) => Ok(Some(handle_from_scalar(i.to_f64(), BUILTIN_NAME)?)),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            Ok(Some(handle_from_scalar(tensor.data[0], BUILTIN_NAME)?))
        }
        _ => Ok(None),
    }
}

fn parse_option(
    option: &str,
    device: &mut Option<PrintDevice>,
    dpi: &mut u32,
) -> BuiltinResult<()> {
    let lower = option.to_ascii_lowercase();
    if let Some(device_name) = lower.strip_prefix('d') {
        let parsed = parse_device(device_name)?;
        if device.replace(parsed).is_some() {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_OPTION,
                "multiple output devices were specified",
            ));
        }
        return Ok(());
    }

    if let Some(resolution) = lower.strip_prefix('r') {
        *dpi = parse_resolution(resolution)?;
        return Ok(());
    }

    Err(print_error_with_detail(
        &PRINT_ERROR_INVALID_OPTION,
        format!("unsupported option '-{option}'"),
    ))
}

fn parse_device(device: &str) -> BuiltinResult<PrintDevice> {
    match device {
        "png" => Ok(PrintDevice::Png),
        "pdf" | "eps" | "epsc" | "eps2" | "svg" => Err(print_error_with_detail(
            &PRINT_ERROR_UNSUPPORTED_DEVICE,
            format!("'-d{device}' is not available yet; RunMat currently supports '-dpng'"),
        )),
        "" => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_OPTION,
            "device option must name a format, for example '-dpng'",
        )),
        other => Err(print_error_with_detail(
            &PRINT_ERROR_UNSUPPORTED_DEVICE,
            format!("unsupported output device '-d{other}'"),
        )),
    }
}

fn parse_resolution(value: &str) -> BuiltinResult<u32> {
    if value == "0" {
        return Ok(DEFAULT_DPI);
    }
    let dpi = value.parse::<u32>().map_err(|err| {
        print_error_with_source(
            format!(
                "{}: expected '-r<N>' with a positive integer DPI",
                PRINT_ERROR_INVALID_OPTION.message
            ),
            &PRINT_ERROR_INVALID_OPTION,
            err,
        )
    })?;
    if dpi == 0 {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_OPTION,
            "resolution must be positive, or use '-r0' for default screen resolution",
        ));
    }
    Ok(dpi)
}

fn infer_device_from_filename(filename: &str) -> Option<PrintDevice> {
    let ext = Path::new(filename)
        .extension()?
        .to_str()?
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some(PrintDevice::Png),
        _ => None,
    }
}

fn dimensions_for_dpi(dpi: u32) -> BuiltinResult<(u32, u32)> {
    let width = scaled_dimension(DEFAULT_WIDTH, dpi)?;
    let height = scaled_dimension(DEFAULT_HEIGHT, dpi)?;
    Ok((width, height))
}

fn scaled_dimension(base: u32, dpi: u32) -> BuiltinResult<u32> {
    let scaled = (u64::from(base) * u64::from(dpi)).div_ceil(u64::from(DEFAULT_DPI));
    if scaled == 0 || scaled > u64::from(MAX_EXPORT_DIMENSION) {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_OPTION,
            format!(
                "resolution '-r{dpi}' would produce an unsupported image dimension ({scaled}px)"
            ),
        ));
    }
    Ok(scaled as u32)
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

fn string_arg(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::CharArray(_) => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "character vector arguments must be 1-by-N",
        )),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::StringArray(_) => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "string array arguments must be scalar",
        )),
        other => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            format!("expected string option or filename, got {other:?}"),
        )),
    }
}

#[cfg(feature = "plot-core")]
async fn render_png(handle: FigureHandle, width: u32, height: u32) -> BuiltinResult<Vec<u8>> {
    super::render_figure_snapshot(handle, width, height, None)
        .await
        .map_err(|err| {
            print_error_with_source(
                format!("{}: {}", PRINT_ERROR_RENDER.message, err.message()),
                &PRINT_ERROR_RENDER,
                err,
            )
        })
}

#[cfg(not(feature = "plot-core"))]
async fn render_png(_handle: FigureHandle, _width: u32, _height: u32) -> BuiltinResult<Vec<u8>> {
    Err(print_error_with_detail(
        &PRINT_ERROR_RENDER,
        "plot-core support is not enabled in this build",
    ))
}

async fn write_bytes(path: &Path, payload: &[u8]) -> BuiltinResult<()> {
    let temp_path = temporary_output_path(path);
    let mut created_temp = false;

    let result: BuiltinResult<()> = async {
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        let mut file = options.open_async(&temp_path).await.map_err(|err| {
            print_error_with_source(
                format!(
                    "{}: unable to open temporary file '{}': {}",
                    PRINT_ERROR_IO.message,
                    temp_path.display(),
                    err
                ),
                &PRINT_ERROR_IO,
                err,
            )
        })?;
        created_temp = true;
        file.write_all(payload).map_err(|err| {
            print_error_with_source(
                format!(
                    "{}: unable to write temporary file '{}': {}",
                    PRINT_ERROR_IO.message,
                    temp_path.display(),
                    err
                ),
                &PRINT_ERROR_IO,
                err,
            )
        })?;
        file.flush_async().await.map_err(|err| {
            print_error_with_source(
                format!(
                    "{}: unable to flush temporary file '{}': {}",
                    PRINT_ERROR_IO.message,
                    temp_path.display(),
                    err
                ),
                &PRINT_ERROR_IO,
                err,
            )
        })?;
        file.sync_all_async().await.map_err(|err| {
            print_error_with_source(
                format!(
                    "{}: unable to sync temporary file '{}': {}",
                    PRINT_ERROR_IO.message,
                    temp_path.display(),
                    err
                ),
                &PRINT_ERROR_IO,
                err,
            )
        })?;
        drop(file);

        runmat_filesystem::rename_async(&temp_path, path)
            .await
            .map_err(|err| {
                print_error_with_source(
                    format!(
                        "{}: unable to replace '{}' with temporary file '{}': {}",
                        PRINT_ERROR_IO.message,
                        path.display(),
                        temp_path.display(),
                        err
                    ),
                    &PRINT_ERROR_IO,
                    err,
                )
            })?;
        Ok(())
    }
    .await;

    if result.is_err() && created_temp {
        let _ = runmat_filesystem::remove_file_async(&temp_path).await;
    }

    result
}

fn temporary_output_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .map(OsString::from)
        .unwrap_or_else(|| OsString::from("print-output"));
    let mut temp_name = OsString::from(".");
    temp_name.push(file_name);
    temp_name.push(format!(
        ".{}.tmp",
        TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));

    path.with_file_name(temp_name)
}

fn print_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    print_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn print_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn print_error_with_source(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    print_error_with_source(
        format!("{}: {}", PRINT_ERROR_INTERNAL.message, err.message()),
        &PRINT_ERROR_INTERNAL,
        err,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::state::{clear_figure, reset_hold_state_for_run};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use futures::executor::block_on;
    use runmat_builtins::{NumericDType, Tensor};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: NumericDType::F64,
        }
    }

    fn unique_temp_path(stem: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("thread");
        let safe_thread_name: String = thread_name
            .chars()
            .map(|ch| match ch {
                '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
                ch if ch.is_control() => '_',
                ch => ch,
            })
            .collect();
        path.push(format!(
            "runmat_print_test_{}_{}_{}",
            stem,
            std::process::id(),
            safe_thread_name
        ));
        path
    }

    #[test]
    fn print_descriptor_signatures_cover_png_forms() {
        let labels: Vec<&str> = PRINT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"ok = print(filename, '-dpng')"));
        assert!(labels.contains(&"ok = print(fig, filename, '-dpng')"));
        assert!(labels.contains(&"ok = print('-dpng', filename)"));
    }

    #[test]
    fn print_rejects_unsupported_device() {
        let _guard = setup();
        let err = block_on(print_builtin(vec![
            Value::from("plot"),
            Value::from("-dsvg"),
        ]))
        .expect_err("svg is not supported yet");
        assert!(err.message().contains("currently supports '-dpng'"));
    }

    #[test]
    fn print_adds_png_extension_and_writes_png() {
        let _guard = setup();
        block_on(plot_builtin(vec![
            Value::Tensor(tensor(&[0.0, 1.0, 2.0, 3.0])),
            Value::Tensor(tensor(&[0.0, 1.0, 4.0, 9.0])),
        ]))
        .expect("plot");

        let stem = unique_temp_path("basic");
        let output = stem.with_extension("png");
        let _ = std::fs::remove_file(&output);
        block_on(print_builtin(vec![
            Value::String(stem.to_string_lossy().into_owned()),
            Value::from("-dpng"),
            Value::from("-r75"),
        ]))
        .expect("print");

        let bytes = std::fs::read(&output).expect("read exported png");
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(bytes.len() > 1000);
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn command_style_order_accepts_device_then_filename() {
        let _guard = setup();
        let request = parse_print_args(&[Value::from("-dpng"), Value::from("command_style_name")])
            .expect("parse");
        assert_eq!(
            request.output_path().expect("output path"),
            PathBuf::from("command_style_name.png")
        );
        assert_eq!(request.device, PrintDevice::Png);
    }

    #[test]
    fn print_appends_png_to_filename_component_only() {
        let request = parse_print_args(&[Value::from("exports/report"), Value::from("-dpng")])
            .expect("parse");
        assert_eq!(
            request.output_path().expect("output path"),
            PathBuf::from("exports/report.png")
        );
    }

    #[test]
    fn print_rejects_directory_like_png_output_path() {
        let request = parse_print_args(&[Value::from("/"), Value::from("-dpng")]).expect("parse");
        let err = request
            .output_path()
            .expect_err("root path has no filename component");
        assert!(err
            .message()
            .contains("filename must include a file name before appending '.png'"));
    }
}
