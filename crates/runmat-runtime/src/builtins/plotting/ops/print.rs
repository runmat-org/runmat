//! MATLAB-compatible `print` builtin for exporting figures.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[cfg(test)]
use runmat_builtins::PRINT_DESCRIPTOR;
use runmat_builtins::{
    print_dimensions, BuiltinErrorDescriptor, PrintControl, PrintControlError, PrintDevice,
    PRINT_DEFAULT_DPI, PRINT_ERROR_INVALID_INPUT, PRINT_ERROR_INVALID_OPTION, PRINT_ERROR_IO,
    PRINT_ERROR_RENDER, PRINT_ERROR_TOO_MANY_OUTPUTS, PRINT_ERROR_UNSUPPORTED_DEVICE,
    PRINT_INTEGER_FIGURE_EXTENSION, PRINT_STATUS_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::op_common::handles::{handle_from_integer, handle_from_scalar};
#[cfg(feature = "plot-core")]
use super::state::clone_figure;
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

#[path = "print/export.rs"]
mod export;

#[runtime_builtin(
    name = "print",
    category = "plotting",
    summary = "Export a figure to an image file.",
    keywords = "print,plotting,figure export,png,save figure",
    sink = true,
    suppress_auto_output = true,
    accel = "metadata",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::print"
)]
pub async fn print_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<bool> {
    let requested_outputs = if let Some(runtime) = runtime.as_ref() {
        runtime
            .scope(async { crate::output_count::current_output_count() })
            .await
    } else {
        crate::output_count::current_output_count()
    };
    match requested_outputs {
        Some(count) if count > 1 => {
            return Err(print_error_with_detail(
                &PRINT_ERROR_TOO_MANY_OUTPUTS,
                "print returns at most one RunMat status result",
            ));
        }
        Some(1) => {
            if let Some(runtime) = runtime.as_ref() {
                crate::compatibility::ensure_builtin_extension_enabled_in_context(
                    runtime,
                    &PRINT_STATUS_OUTPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &PRINT_STATUS_OUTPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
        }
        _ => {}
    }
    let selected_figure = current_figure_handle();
    preflight_host_arguments(&args)?;
    if args
        .first()
        .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        if let Some(runtime) = runtime.as_ref() {
            crate::compatibility::ensure_builtin_extension_enabled_in_context(
                runtime,
                &PRINT_INTEGER_FIGURE_EXTENSION,
                BUILTIN_NAME,
            )?;
        } else {
            crate::compatibility::ensure_builtin_extension_enabled(
                &PRINT_INTEGER_FIGURE_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    #[cfg(feature = "plot-core")]
    let early_handle = match args.first() {
        Some(value) if is_host_text(value) => Some(selected_figure),
        Some(value) => figure_handle_arg(value)?,
        None => None,
    };
    // The ordinary host-known scene is cloned before any provider or renderer
    // await. A resident handle is resolved and cloned after materialization.
    #[cfg(feature = "plot-core")]
    let early_scene = early_handle.and_then(clone_figure);
    let owner = export::PrintFsOwner::capture().map_err(export::map_io)?;
    if let Some(first) = args.first() {
        crate::builtins::common::validation::reject_typed_complex_integer(first, "print")?;
        if crate::builtins::common::validation::value_has_native_integer_class(first)
            && !export::await_effect(
                runtime.as_ref(),
                crate::builtins::common::validation::native_integer_value_is_exact_f64_async(first),
            )
            .await??
        {
            return Err(build_runtime_error(
                "print: integer figure handle values must be exactly representable as double",
            )
            .with_builtin(BUILTIN_NAME)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
    }
    // Host-known invalid controls are rejected before any resident download or
    // filesystem service capture. Dynamic controls receive the same parser
    // after materialization.
    let args = gather_values(runtime.as_ref(), &args).await?;
    let request = parse_print_args(&args, selected_figure)?;
    let path = request.output_path()?;
    let selected = owner.path(path).map_err(export::map_io)?;
    #[cfg(feature = "plot-core")]
    let scene = if early_handle == Some(request.figure) {
        early_scene
    } else {
        clone_figure(request.figure)
    }
    .ok_or_else(|| {
        print_error_with_detail(
            &PRINT_ERROR_RENDER,
            format!("figure handle {} does not exist", request.figure.as_u32()),
        )
    })?;
    #[cfg(feature = "plot-core")]
    let bytes = export::await_effect(
        runtime.as_ref(),
        render_png(scene, request.width, request.height),
    )
    .await??;
    #[cfg(not(feature = "plot-core"))]
    let bytes = export::await_effect(
        runtime.as_ref(),
        render_png(request.figure, request.width, request.height),
    )
    .await??;
    export::replace_png(runtime.as_ref(), &selected, &bytes).await?;
    Ok(true)
}

#[cfg(feature = "plot-core")]
fn is_host_text(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_)
    )
}

async fn gather_values(
    runtime: Option<&crate::context::RuntimeContext>,
    values: &[Value],
) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            export::await_effect(runtime, gather_if_needed_async(value))
                .await?
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
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

fn parse_print_args(args: &[Value], selected_figure: FigureHandle) -> BuiltinResult<PrintRequest> {
    if args.is_empty() {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename is required",
        ));
    }

    let mut idx = 0usize;
    let mut figure = selected_figure;
    if let Some(handle) = figure_handle_arg(&args[0])? {
        figure = handle;
        idx = 1;
    }

    let mut filename: Option<String> = None;
    let mut device: Option<PrintDevice> = None;
    let mut dpi = PRINT_DEFAULT_DPI;

    while idx < args.len() {
        let token = string_arg(&args[idx])?;
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_INPUT,
                "empty string arguments are not valid print options",
            ));
        }

        if trimmed.starts_with('-') {
            parse_option(trimmed, &mut device, &mut dpi)?;
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
    let (width, height) = print_dimensions(dpi).map_err(|error| {
        print_error_with_detail(
            &PRINT_ERROR_INVALID_OPTION,
            format!("unsupported resolution: {error:?}"),
        )
    })?;
    Ok(PrintRequest {
        figure,
        filename,
        device,
        width,
        height,
    })
}

fn preflight_host_arguments(args: &[Value]) -> BuiltinResult<()> {
    if args.is_empty() {
        return Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_INPUT,
            "filename is required",
        ));
    }
    let mut filenames = 0usize;
    let mut devices = 0usize;
    let mut dpi = PRINT_DEFAULT_DPI;
    for (index, value) in args.iter().enumerate() {
        if index == 0 && figure_handle_arg(value)?.is_some() {
            continue;
        }
        let token = match value {
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => string_arg(value)?,
            // Resident controls are validated after their admitted gather.
            _ => continue,
        };
        let token = token.trim();
        if token.is_empty() {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_INPUT,
                "empty string arguments are not valid print options",
            ));
        }
        if token.starts_with('-') {
            let mut device = None;
            parse_option(token, &mut device, &mut dpi)?;
            if device.is_some() {
                devices += 1;
            }
        } else {
            filenames += 1;
            if token.contains('\0') {
                return Err(print_error_with_detail(
                    &PRINT_ERROR_INVALID_INPUT,
                    "filename must not contain NUL bytes",
                ));
            }
        }
        if filenames > 1 || devices > 1 {
            return Err(print_error_with_detail(
                &PRINT_ERROR_INVALID_INPUT,
                "too many filenames or device options",
            ));
        }
    }
    Ok(())
}

fn figure_handle_arg(value: &Value) -> BuiltinResult<Option<FigureHandle>> {
    match value {
        Value::Num(v) => Ok(Some(handle_from_scalar(*v, BUILTIN_NAME)?)),
        Value::Int(i) => Ok(Some(handle_from_integer(i, BUILTIN_NAME)?)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return Ok(Some(handle_from_integer(
                    &storage.value_at(0).expect("one-element integer storage"),
                    BUILTIN_NAME,
                )?));
            }
            Ok(Some(handle_from_scalar(
                tensor_utils::tensor_value_f64(tensor, 0),
                BUILTIN_NAME,
            )?))
        }
        _ => Ok(None),
    }
}

fn parse_option(token: &str, device: &mut Option<PrintDevice>, dpi: &mut u32) -> BuiltinResult<()> {
    match PrintControl::decode(token) {
        Ok(PrintControl::Device(parsed)) => {
            if device.replace(parsed).is_some() {
                return Err(print_error_with_detail(
                    &PRINT_ERROR_INVALID_OPTION,
                    "multiple output devices were specified",
                ));
            }
            Ok(())
        }
        Ok(PrintControl::Resolution(parsed)) => {
            *dpi = parsed;
            Ok(())
        }
        Err(PrintControlError::UnsupportedDevice) => Err(print_error_with_detail(
            &PRINT_ERROR_UNSUPPORTED_DEVICE,
            format!("'{token}' is not available yet; RunMat currently supports '-dpng'"),
        )),
        Err(PrintControlError::InvalidOption) => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_OPTION,
            format!("unsupported option '{token}'"),
        )),
        Err(PrintControlError::ResolutionParse(source)) => Err(print_error_with_source(
            format!(
                "{}: expected '-r<N>' with a nonnegative integer DPI",
                PRINT_ERROR_INVALID_OPTION.message
            ),
            &PRINT_ERROR_INVALID_OPTION,
            source,
        )),
        Err(PrintControlError::UnsupportedResolution(scaled)) => Err(print_error_with_detail(
            &PRINT_ERROR_INVALID_OPTION,
            format!(
                "resolution '{token}' would produce an unsupported image dimension ({scaled}px)"
            ),
        )),
    }
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
async fn render_png(
    scene: runmat_plot::plots::Figure,
    width: u32,
    height: u32,
) -> BuiltinResult<Vec<u8>> {
    super::render_figure_png_bytes(scene, width, height)
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
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
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
    let identifier = err.identifier().map(str::to_owned);
    let mut builder = build_runtime_error(err.message().to_owned())
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::state::{clear_figure, reset_hold_state_for_run};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use futures::executor::block_on;
    use runmat_value::Tensor;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("print test vector")
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
    fn print_figure_handle_arg_reads_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::U32(vec![5]), vec![1, 1]).unwrap();

        assert_eq!(
            figure_handle_arg(&Value::Tensor(tensor)).unwrap(),
            Some(FigureHandle::from(5))
        );
    }

    #[test]
    fn print_descriptor_signatures_cover_png_forms() {
        let labels: Vec<&str> = PRINT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"print(filename)"));
        assert!(labels.contains(&"print(fig, filename, ___)"));
        assert!(labels.contains(&"ok = print(filename, ___)"));
    }

    #[test]
    fn print_rejects_unsupported_device() {
        let _guard = setup();
        let err = block_on(print_builtin(
            None,
            vec![Value::from("plot"), Value::from("-dsvg")],
        ))
        .expect_err("svg is not supported yet");
        assert!(err.message().contains("currently supports '-dpng'"));
    }

    #[test]
    fn output_admission_precedes_render_and_file_effects() {
        let _guard = setup();
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let one = crate::output_count::push_output_count(Some(1));
        let error = block_on(print_builtin(None, vec![Value::from("never-created.png")]))
            .expect_err("status output is an extension");
        assert_eq!(
            error.identifier(),
            PRINT_STATUS_OUTPUT_EXTENSION.error_identifier
        );
        drop(one);
        let _two = crate::output_count::push_output_count(Some(2));
        let error = block_on(print_builtin(None, vec![Value::from("never-created.png")]))
            .expect_err("excess outputs");
        assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
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
        let _statement = crate::output_count::push_output_count(Some(0));
        block_on(print_builtin(
            None,
            vec![
                Value::String(stem.to_string_lossy().into_owned()),
                Value::from("-dpng"),
                Value::from("-r75"),
            ],
        ))
        .expect("print");

        let bytes = std::fs::read(&output).expect("read exported png");
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(bytes.len() > 1000);
        let _ = std::fs::remove_file(&output);
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn captured_scene_renders_after_ambient_figure_is_cleared() {
        let _guard = setup();
        block_on(plot_builtin(vec![
            Value::Tensor(tensor(&[0.0, 1.0, 2.0])),
            Value::Tensor(tensor(&[1.0, 3.0, 2.0])),
        ]))
        .expect("plot");
        let selected = current_figure_handle();
        let scene = clone_figure(selected).expect("captured plotted scene");
        clear_figure(None).expect("clear ambient figure");
        let bytes = block_on(render_png(scene, 800, 600)).expect("captured scene export");
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(bytes.len() > 1000);
    }

    #[test]
    fn command_style_order_accepts_device_then_filename() {
        let _guard = setup();
        let request = parse_print_args(
            &[Value::from("-dpng"), Value::from("command_style_name")],
            current_figure_handle(),
        )
        .expect("parse");
        assert_eq!(
            request.output_path().expect("output path"),
            PathBuf::from("command_style_name.png")
        );
        assert_eq!(request.device, PrintDevice::Png);
    }

    #[test]
    fn print_appends_png_to_filename_component_only() {
        let request = parse_print_args(
            &[Value::from("exports/report"), Value::from("-dpng")],
            current_figure_handle(),
        )
        .expect("parse");
        assert_eq!(
            request.output_path().expect("output path"),
            PathBuf::from("exports/report.png")
        );
    }

    #[test]
    fn print_rejects_directory_like_png_output_path() {
        let request = parse_print_args(
            &[Value::from("/"), Value::from("-dpng")],
            current_figure_handle(),
        )
        .expect("parse");
        let err = request
            .output_path()
            .expect_err("root path has no filename component");
        assert!(err
            .message()
            .contains("filename must include a file name before appending '.png'"));
    }
}
