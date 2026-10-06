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
#[path = "print/arguments.rs"]
mod arguments;

use arguments::{figure_handle_arg, parse_print_args, preflight_host_arguments};

#[cfg(all(test, feature = "plot-core"))]
#[path = "print/tests/pending.rs"]
mod pending_tests;

#[runtime_builtin(
    name = "print",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::print"
)]
pub async fn print_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    mut args: Vec<Value>,
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
    if let Some(first) = args.first() {
        crate::builtins::common::validation::reject_typed_complex_integer(first, "print")?;
    }
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
    if let Some(first) = args.first() {
        if crate::builtins::common::validation::value_has_native_integer_class(first)
            && !matches!(first, Value::GpuTensor(_))
            && !crate::builtins::common::validation::native_integer_value_is_exact_f64(first)
        {
            return Err(build_runtime_error(
                "print: integer figure handle values must be exactly representable as double",
            )
            .with_builtin(BUILTIN_NAME)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
    }
    let known_filename = preflight_host_arguments(&args)?;
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
    let owner = export::PrintFsOwner::capture(known_filename.as_deref()).map_err(export::map_io)?;
    // A wide resident integer handle needs one authenticated provider readback
    // to establish binary64 exactness. Retain and materialize that payload so
    // the ordinary argument gather never repeats the download.
    if let Some(first @ Value::GpuTensor(_)) = args.first() {
        if crate::builtins::common::validation::value_has_native_integer_class(first) {
            let mut cooperation = runtime
                .as_ref()
                .map(crate::context::RuntimeContext::cpu_cooperation);
            let prepared = export::await_effect(
                runtime.as_ref(),
                crate::builtins::common::validation::prepare_direct_integer_f64_boundary(
                    first,
                    cooperation.as_mut(),
                ),
            )
            .await?
            .map_err(|error| error.into_runtime_error())?;
            if !prepared.is_exact() {
                return Err(build_runtime_error(
                    "print: integer figure handle values must be exactly representable as double",
                )
                .with_builtin(BUILTIN_NAME)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build());
            }
            if let Some(payload) = prepared.into_resident_payload() {
                args[0] = payload.into_value().map_err(|detail| {
                    build_runtime_error(format!(
                        "print: validated integer figure handle could not be materialized: {detail}"
                    ))
                    .with_identifier("RunMat:Internal")
                    .with_builtin(BUILTIN_NAME)
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
                })?;
            }
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
#[path = "print/tests.rs"]
mod tests;
