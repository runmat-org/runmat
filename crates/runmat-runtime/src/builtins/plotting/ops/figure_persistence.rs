//! MATLAB-compatible figure save/load helpers.

use std::collections::HashSet;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, NumericDType, StructValue, Tensor, Type, Value,
};
use runmat_filesystem::OpenOptions;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::replay::limits::ReplayLimits;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::properties::{get_properties, resolve_plot_handle, set_properties, PlotHandle};
use super::state::{current_figure_handle, set_figure_visible, FigureHandle};

const SAVEAS: &str = "saveas";
const SAVEFIG: &str = "savefig";
const OPENFIG: &str = "openfig";
const HGSAVE: &str = "hgsave";
const HGLOAD: &str = "hgload";

const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;
const FIG_MAGIC: &[u8] = b"RUNMAT-FIGURE-BUNDLE-V1\n";
const MAX_FIGURE_BUNDLE_COUNT: usize = 4096;
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

const OUTPUT_OK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the save operation completed.",
}];

const OUTPUT_FIG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure handle or array of figure handles.",
}];

const OUTPUT_HGLOAD_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Top-level graphics handles loaded from the file.",
    },
    BuiltinParamDescriptor {
        name: "old_prop_values",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array of previous top-level property values overridden by hgload.",
    },
];

const INPUT_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "MATLAB-compatible figure persistence arguments.",
}];

const SAVEAS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "ok = saveas(fig, filename)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = saveas(fig, filename, formattype)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_OK,
    },
];

const SAVEFIG_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "ok = savefig(filename)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = savefig(fig, filename)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = savefig(fig, filename, version)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_OK,
    },
];

const OPENFIG_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "fig = openfig(filename)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_FIG,
    },
    BuiltinSignatureDescriptor {
        label: "fig = openfig(filename, copies)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_FIG,
    },
    BuiltinSignatureDescriptor {
        label: "fig = openfig(filename, visibility)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_FIG,
    },
    BuiltinSignatureDescriptor {
        label: "fig = openfig(filename, copies, visibility)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_FIG,
    },
];

const HGSAVE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "ok = hgsave(filename)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = hgsave(h, filename)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = hgsave(..., version)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_OK,
    },
];

const HGLOAD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "h = hgload(filename)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_FIG,
    },
    BuiltinSignatureDescriptor {
        label: "[h, old_prop_values] = hgload(filename, property_structure)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_HGLOAD_TWO,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.INVALID_ARGUMENT",
    identifier: Some("RunMat:figurePersistence:InvalidArgument"),
    when: "Arguments are missing, malformed, or incompatible with the requested figure persistence operation.",
    message: "figure persistence: invalid argument",
};
const ERROR_UNSUPPORTED_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.UNSUPPORTED_FORMAT",
    identifier: Some("RunMat:figurePersistence:UnsupportedFormat"),
    when: "The requested save/load format is recognized but not supported by RunMat's active plotting exporter.",
    message: "figure persistence: unsupported format",
};
const ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.IO",
    identifier: Some("RunMat:figurePersistence:IoFailure"),
    when: "The target file cannot be read, written, flushed, synced, or atomically replaced.",
    message: "figure persistence: file I/O failed",
};
const ERROR_RENDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.RENDER",
    identifier: Some("RunMat:figurePersistence:RenderFailed"),
    when: "The plotting renderer or scene exporter cannot serialize the requested figure.",
    message: "figure persistence: export failed",
};
const ERROR_IMPORT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.IMPORT",
    identifier: Some("RunMat:figurePersistence:ImportFailed"),
    when: "The figure file is missing, corrupt, unsupported, or incompatible with RunMat's scene importer.",
    message: "figure persistence: import failed",
};
const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.INTERNAL",
    identifier: Some("RunMat:figurePersistence:Internal"),
    when: "Internal runtime control-flow or conversion fails.",
    message: "figure persistence: internal operation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_UNSUPPORTED_FORMAT,
    ERROR_IO,
    ERROR_RENDER,
    ERROR_IMPORT,
    ERROR_INTERNAL,
];

pub const SAVEAS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SAVEAS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const SAVEFIG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SAVEFIG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const OPENFIG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OPENFIG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const HGSAVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HGSAVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const HGLOAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HGLOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::figure_persistence")]
pub const SAVEAS_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "saveas",
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
    notes: "saveas gathers only handle/path/format arguments. PNG output uses the shared plotting renderer; FIG output serializes the figure scene replay payload.",
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::figure_persistence")]
pub const SAVEFIG_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "savefig",
    op_kind: GpuOpKind::Custom("figure-scene-export"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "savefig is an I/O sink and serializes RunMat figure scene payloads without forcing numeric plot data through scalar CPU loops.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "figure_persistence",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Figure persistence performs plotting I/O and terminates fusion graphs.",
};

fn bool_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Bool
}

fn handle_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Num
}

#[runtime_builtin(
    name = "saveas",
    category = "plotting",
    summary = "Save a figure to FIG or PNG format.",
    keywords = "saveas,figure save,plotting,png,fig",
    sink = true,
    suppress_auto_output = true,
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::figure_persistence::SAVEAS_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn saveas_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    let args = gather_values(&args, SAVEAS).await?;
    let request = parse_saveas_args(&args)?;
    match request.format {
        SaveFormat::Fig => save_figures(&request.output_path, &[request.figure], SAVEAS).await?,
        SaveFormat::Png => {
            let bytes = render_png(request.figure, DEFAULT_WIDTH, DEFAULT_HEIGHT, SAVEAS).await?;
            write_bytes(&request.output_path, &bytes, SAVEAS).await?;
        }
    }
    Ok(true)
}

#[runtime_builtin(
    name = "savefig",
    category = "plotting",
    summary = "Save figures as RunMat FIG files.",
    keywords = "savefig,figure save,plotting,fig",
    sink = true,
    suppress_auto_output = true,
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::figure_persistence::SAVEFIG_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn savefig_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    let args = gather_values(&args, SAVEFIG).await?;
    let request = parse_savefig_args(&args, SAVEFIG)?;
    save_figures(&request.output_path, &request.figures, SAVEFIG).await?;
    Ok(true)
}

#[runtime_builtin(
    name = "openfig",
    category = "plotting",
    summary = "Open figures saved as RunMat FIG files.",
    keywords = "openfig,figure open,plotting,fig",
    suppress_auto_output = true,
    type_resolver(handle_type),
    descriptor(crate::builtins::plotting::figure_persistence::OPENFIG_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn openfig_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_values(&args, OPENFIG).await?;
    let request = parse_openfig_args(&args, OPENFIG)?;
    let handles = open_figures(&request, OPENFIG).await?;
    Ok(handles_value(&handles))
}

#[runtime_builtin(
    name = "hgsave",
    category = "plotting",
    summary = "Save figures or parent figures of graphics handles to RunMat FIG files.",
    keywords = "hgsave,figure save,graphics,fig",
    sink = true,
    suppress_auto_output = true,
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::figure_persistence::HGSAVE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn hgsave_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    let args = gather_values(&args, HGSAVE).await?;
    let request = parse_savefig_args(&args, HGSAVE)?;
    save_figures(&request.output_path, &request.figures, HGSAVE).await?;
    Ok(true)
}

#[runtime_builtin(
    name = "hgload",
    category = "plotting",
    summary = "Load graphics object hierarchies from RunMat FIG files.",
    keywords = "hgload,figure load,graphics,fig",
    type_resolver(handle_type),
    descriptor(crate::builtins::plotting::figure_persistence::HGLOAD_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn hgload_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_values(&args, HGLOAD).await?;
    let (request, overrides) = parse_hgload_args(&args)?;
    let handles = open_figures(&request, HGLOAD).await?;
    let old_values = apply_hgload_overrides(&handles, overrides.as_ref())?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let outputs = vec![handles_value(&handles), old_values];
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
    Ok(handles_value(&handles))
}

async fn gather_values(values: &[Value], builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather_if_needed_async(value).await.map_err(|err| {
            let message = err.message().to_string();
            persistence_error_with_source(&ERROR_INTERNAL, builtin, message, err)
        })?);
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SaveFormat {
    Fig,
    Png,
}

impl SaveFormat {
    fn standard_extension(self) -> &'static str {
        match self {
            SaveFormat::Fig => "fig",
            SaveFormat::Png => "png",
        }
    }

    fn label(self) -> &'static str {
        match self {
            SaveFormat::Fig => "fig",
            SaveFormat::Png => "png",
        }
    }
}

#[derive(Debug)]
struct SaveAsRequest {
    figure: FigureHandle,
    output_path: PathBuf,
    format: SaveFormat,
}

#[derive(Debug)]
struct SaveFigRequest {
    figures: Vec<FigureHandle>,
    output_path: PathBuf,
}

#[derive(Debug)]
struct OpenFigRequest {
    path: PathBuf,
    visibility: Option<bool>,
}

fn parse_saveas_args(args: &[Value]) -> BuiltinResult<SaveAsRequest> {
    if args.len() < 2 || args.len() > 3 {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            SAVEAS,
            "expected saveas(fig, filename) or saveas(fig, filename, formattype)",
        ));
    }
    let figure = parent_figure_from_value(&args[0], SAVEAS)?;
    let filename = scalar_string(&args[1], SAVEAS)?;
    validate_filename(&filename, SAVEAS)?;
    let explicit_format = args
        .get(2)
        .map(|value| parse_save_format(&scalar_string(value, SAVEAS)?, SAVEAS))
        .transpose()?;
    let inferred = infer_save_format(&filename, SAVEAS)?;
    let format = match explicit_format {
        Some(explicit) => {
            if let Some(inferred) = inferred {
                if inferred != explicit {
                    return Err(persistence_error(
                        &ERROR_UNSUPPORTED_FORMAT,
                        SAVEAS,
                        format!(
                            "saveas: filename extension '.{}' does not match requested format '{}'",
                            inferred.standard_extension(),
                            explicit.label()
                        ),
                    ));
                }
            }
            explicit
        }
        None => inferred.unwrap_or(SaveFormat::Fig),
    };
    let output_path = output_path_for_format(&filename, format, SAVEAS)?;
    Ok(SaveAsRequest {
        figure,
        output_path,
        format,
    })
}

fn parse_savefig_args(args: &[Value], builtin: &'static str) -> BuiltinResult<SaveFigRequest> {
    if args.is_empty() || args.len() > 3 {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected filename, optional handles, and optional version"),
        ));
    }

    let mut idx = 0usize;
    let figures =
        if args.len() >= 2 && !is_version_token(&args[0]) && !is_probable_filename(&args[0]) {
            idx = 1;
            top_level_figures_from_value(&args[0], builtin)?
        } else {
            vec![current_figure_handle()]
        };

    if idx >= args.len() {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: filename is required"),
        ));
    }
    let filename = scalar_string(&args[idx], builtin)?;
    validate_filename(&filename, builtin)?;
    idx += 1;

    while idx < args.len() {
        validate_version_token(&args[idx], builtin)?;
        idx += 1;
    }

    let output_path = fig_path(&filename, true, builtin)?;
    Ok(SaveFigRequest {
        figures,
        output_path,
    })
}

fn parse_openfig_args(args: &[Value], builtin: &'static str) -> BuiltinResult<OpenFigRequest> {
    if args.is_empty() || args.len() > 3 {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected filename, optional copies, and optional visibility"),
        ));
    }
    let filename = scalar_string(&args[0], builtin)?;
    validate_filename(&filename, builtin)?;
    let mut visibility = None;
    let mut saw_copies = false;
    for value in &args[1..] {
        let token = scalar_string(value, builtin)?;
        match token.trim().to_ascii_lowercase().as_str() {
            "new" if !saw_copies => saw_copies = true,
            "reuse" if !saw_copies => {
                return Err(persistence_error(
                    &ERROR_UNSUPPORTED_FORMAT,
                    builtin,
                    format!(
                        "{builtin}: openfig reuse mode requires figure-file reuse tracking, which is not available yet"
                    ),
                ));
            }
            "visible" if visibility.replace(true).is_none() => {}
            "invisible" if visibility.replace(false).is_none() => {}
            "on" if visibility.replace(true).is_none() => {}
            "off" if visibility.replace(false).is_none() => {}
            other => {
                return Err(persistence_error(
                    &ERROR_INVALID_ARGUMENT,
                    builtin,
                    format!("{builtin}: unsupported option '{other}'"),
                ));
            }
        }
    }
    Ok(OpenFigRequest {
        path: fig_path(&filename, false, builtin)?,
        visibility,
    })
}

fn parse_hgload_args(args: &[Value]) -> BuiltinResult<(OpenFigRequest, Option<StructValue>)> {
    if args.is_empty() || args.len() > 2 {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            HGLOAD,
            "hgload: expected filename and optional property structure",
        ));
    }
    let request = parse_openfig_args(&args[..1], HGLOAD)?;
    let overrides = match args.get(1) {
        Some(Value::Struct(st)) => Some(st.clone()),
        Some(_) => {
            return Err(persistence_error(
                &ERROR_INVALID_ARGUMENT,
                HGLOAD,
                "hgload: property_structure must be a scalar struct",
            ));
        }
        None => None,
    };
    Ok((request, overrides))
}

fn parent_figure_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<FigureHandle> {
    match resolve_plot_handle(value, builtin)? {
        PlotHandle::Figure(handle) => Ok(handle),
        PlotHandle::Axes(handle, _) => Ok(handle),
        PlotHandle::Ruler(handle, _, _) => Ok(handle),
        PlotHandle::Text(handle, _, _) => Ok(handle),
        PlotHandle::Legend(handle, _) => Ok(handle),
        PlotHandle::PlotChild(_, state) => Ok(state.figure_axes().0),
        PlotHandle::Root => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: root handle cannot be saved as a figure"),
        )),
    }
}

fn top_level_figures_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Vec<FigureHandle>> {
    match value {
        Value::Tensor(tensor) => {
            let values = tensor_utils::tensor_values_f64_cow(tensor);
            if values.is_empty() {
                return Err(persistence_error(
                    &ERROR_INVALID_ARGUMENT,
                    builtin,
                    format!("{builtin}: figure handle array cannot be empty"),
                ));
            }
            let mut seen = HashSet::with_capacity(values.len());
            let mut figures = Vec::with_capacity(values.len());
            for &handle in values.iter() {
                let value = Value::Num(handle);
                let figure = parent_figure_from_value(&value, builtin)?;
                if seen.insert(figure) {
                    figures.push(figure);
                }
            }
            Ok(figures)
        }
        _ => Ok(vec![parent_figure_from_value(value, builtin)?]),
    }
}

fn is_probable_filename(value: &Value) -> bool {
    scalar_string(value, SAVEFIG).is_ok()
}

fn is_version_token(value: &Value) -> bool {
    scalar_string(value, SAVEFIG)
        .map(|text| {
            matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "-v6" | "v6" | "-v7" | "v7" | "-v7.3" | "v7.3" | "compact"
            )
        })
        .unwrap_or(false)
}

fn validate_version_token(value: &Value, builtin: &'static str) -> BuiltinResult<()> {
    let token = scalar_string(value, builtin)?;
    match token.trim().to_ascii_lowercase().as_str() {
        "-v6" | "v6" | "-v7" | "v7" | "-v7.3" | "v7.3" | "compact" => Ok(()),
        other => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: unsupported FIG version option '{other}'"),
        )),
    }
}

fn parse_save_format(raw: &str, builtin: &'static str) -> BuiltinResult<SaveFormat> {
    let lowered = raw
        .trim()
        .trim_start_matches("-d")
        .trim_start_matches('.')
        .to_ascii_lowercase();
    match lowered.as_str() {
        "fig" => Ok(SaveFormat::Fig),
        "png" => Ok(SaveFormat::Png),
        "m" | "mfig" | "jpeg" | "jpg" | "tiff" | "tif" | "pdf" | "eps" | "epsc" | "svg" => {
            Err(persistence_error(
                &ERROR_UNSUPPORTED_FORMAT,
                builtin,
                format!("{builtin}: format '{lowered}' is not available yet; RunMat currently supports FIG and PNG"),
            ))
        }
        "" => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: format must not be empty"),
        )),
        other => Err(persistence_error(
            &ERROR_UNSUPPORTED_FORMAT,
            builtin,
            format!("{builtin}: unsupported output format '{other}'"),
        )),
    }
}

fn infer_save_format(filename: &str, builtin: &'static str) -> BuiltinResult<Option<SaveFormat>> {
    match Path::new(filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("fig") => Ok(Some(SaveFormat::Fig)),
        Some("png") => Ok(Some(SaveFormat::Png)),
        Some(ext) => parse_save_format(ext, builtin).map(Some),
        None => Ok(None),
    }
}

fn output_path_for_format(
    filename: &str,
    format: SaveFormat,
    builtin: &'static str,
) -> BuiltinResult<PathBuf> {
    let path = Path::new(filename);
    if path.extension().is_some() {
        return Ok(path.to_path_buf());
    }
    append_extension(path, format.standard_extension(), builtin)
}

fn fig_path(filename: &str, reject_non_fig: bool, builtin: &'static str) -> BuiltinResult<PathBuf> {
    let path = Path::new(filename);
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("fig") => Ok(path.to_path_buf()),
        Some(ext) if reject_non_fig => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected a .fig filename, got extension '.{ext}'"),
        )),
        Some(_) => Ok(path.to_path_buf()),
        None => append_extension(path, "fig", builtin),
    }
}

fn append_extension(path: &Path, ext: &str, builtin: &'static str) -> BuiltinResult<PathBuf> {
    let Some(file_name) = path.file_name() else {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: filename must include a file name before appending '.{ext}'"),
        ));
    };
    let mut file_name = OsString::from(file_name);
    file_name.push(".");
    file_name.push(ext);
    Ok(path.with_file_name(file_name))
}

async fn save_figures(
    path: &Path,
    figures: &[FigureHandle],
    builtin: &'static str,
) -> BuiltinResult<()> {
    if figures.is_empty() {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: at least one figure is required"),
        ));
    }
    let mut payloads = Vec::with_capacity(figures.len());
    for handle in figures {
        let bytes = export_figure(*handle, builtin).await?;
        payloads.push(bytes);
    }
    let bytes = encode_figure_bundle(&payloads, builtin)?;
    write_bytes(path, &bytes, builtin).await
}

async fn open_figures(
    request: &OpenFigRequest,
    builtin: &'static str,
) -> BuiltinResult<Vec<FigureHandle>> {
    let bytes = runmat_filesystem::read_async(&request.path)
        .await
        .map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to read figure file '{}'",
                    request.path.display()
                ),
                err,
            )
        })?;
    let payloads = decode_figure_bundle(&bytes, builtin)?;
    let mut handles = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let handle = import_figure(payload, builtin).await?;
        if let Some(visible) = request.visibility {
            set_figure_visible(handle, visible).map_err(|err| {
                persistence_error_with_source(&ERROR_IMPORT, builtin, err.to_string(), err)
            })?;
        }
        handles.push(handle);
    }
    Ok(handles)
}

#[cfg(feature = "plot-core")]
async fn export_figure(handle: FigureHandle, builtin: &'static str) -> BuiltinResult<Vec<u8>> {
    super::export_figure_scene(handle)
        .await
        .map_err(|err| {
            persistence_error_with_source(
                &ERROR_RENDER,
                builtin,
                format!("{builtin}: unable to export figure scene"),
                err,
            )
        })?
        .ok_or_else(|| {
            persistence_error(
                &ERROR_RENDER,
                builtin,
                format!("{builtin}: plot-core exporter did not produce a figure scene"),
            )
        })
}

#[cfg(not(feature = "plot-core"))]
async fn export_figure(_handle: FigureHandle, builtin: &'static str) -> BuiltinResult<Vec<u8>> {
    Err(persistence_error(
        &ERROR_RENDER,
        builtin,
        format!("{builtin}: plot-core support is not enabled in this build"),
    ))
}

#[cfg(feature = "plot-core")]
async fn import_figure(bytes: &[u8], builtin: &'static str) -> BuiltinResult<FigureHandle> {
    super::import_figure_scene_async(bytes)
        .await
        .map_err(|err| {
            persistence_error_with_source(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: unable to import figure scene"),
                err,
            )
        })?
        .ok_or_else(|| {
            persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: figure file did not contain an importable figure"),
            )
        })
}

#[cfg(not(feature = "plot-core"))]
async fn import_figure(_bytes: &[u8], builtin: &'static str) -> BuiltinResult<FigureHandle> {
    Err(persistence_error(
        &ERROR_IMPORT,
        builtin,
        format!("{builtin}: plot-core support is not enabled in this build"),
    ))
}

#[cfg(feature = "plot-core")]
async fn render_png(
    handle: FigureHandle,
    width: u32,
    height: u32,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    super::render_figure_snapshot(handle, width, height, None)
        .await
        .map_err(|err| {
            persistence_error_with_source(
                &ERROR_RENDER,
                builtin,
                format!("{builtin}: unable to render PNG"),
                err,
            )
        })
}

#[cfg(not(feature = "plot-core"))]
async fn render_png(
    _handle: FigureHandle,
    _width: u32,
    _height: u32,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    Err(persistence_error(
        &ERROR_RENDER,
        builtin,
        format!("{builtin}: plot-core support is not enabled in this build"),
    ))
}

fn encode_figure_bundle(payloads: &[Vec<u8>], builtin: &'static str) -> BuiltinResult<Vec<u8>> {
    if payloads.len() == 1 {
        return Ok(payloads[0].clone());
    }
    let count = u32::try_from(payloads.len()).map_err(|_| {
        persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: too many figures to save in one FIG file"),
        )
    })?;
    let mut out = Vec::new();
    out.extend_from_slice(FIG_MAGIC);
    out.extend_from_slice(&count.to_le_bytes());
    for payload in payloads {
        let len = u64::try_from(payload.len()).map_err(|_| {
            persistence_error(
                &ERROR_INVALID_ARGUMENT,
                builtin,
                format!("{builtin}: figure payload is too large"),
            )
        })?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(payload);
    }
    Ok(out)
}

fn decode_figure_bundle<'a>(
    bytes: &'a [u8],
    builtin: &'static str,
) -> BuiltinResult<Vec<&'a [u8]>> {
    if !bytes.starts_with(FIG_MAGIC) {
        return Ok(vec![bytes]);
    }
    let mut cursor = FIG_MAGIC.len();
    let count = usize::try_from(read_u32(bytes, &mut cursor, builtin)?).map_err(|_| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle count does not fit this platform"),
        )
    })?;
    if count == 0 {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle contains no figures"),
        ));
    }
    if count > MAX_FIGURE_BUNDLE_COUNT {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle contains too many figures"),
        ));
    }
    let remaining = bytes.len().saturating_sub(cursor);
    if count > remaining / 8 {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: truncated FIG bundle"),
        ));
    }
    let mut payloads = Vec::with_capacity(count);
    for _ in 0..count {
        let len = usize::try_from(read_u64(bytes, &mut cursor, builtin)?).map_err(|_| {
            persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: FIG payload length does not fit this platform"),
            )
        })?;
        if len > ReplayLimits::default().max_scene_payload_bytes {
            return Err(persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: FIG payload exceeds maximum scene payload size"),
            ));
        }
        let end = cursor.checked_add(len).ok_or_else(|| {
            persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: FIG bundle size overflow"),
            )
        })?;
        if end > bytes.len() {
            return Err(persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: truncated FIG bundle"),
            ));
        }
        payloads.push(&bytes[cursor..end]);
        cursor = end;
    }
    if cursor != bytes.len() {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle has trailing bytes"),
        ));
    }
    Ok(payloads)
}

fn read_u32(bytes: &[u8], cursor: &mut usize, builtin: &'static str) -> BuiltinResult<u32> {
    let end = cursor.checked_add(4).ok_or_else(|| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: corrupt FIG bundle"),
        )
    })?;
    let slice = bytes.get(*cursor..end).ok_or_else(|| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: truncated FIG bundle"),
        )
    })?;
    *cursor = end;
    Ok(u32::from_le_bytes(
        slice.try_into().expect("u32 slice length"),
    ))
}

fn read_u64(bytes: &[u8], cursor: &mut usize, builtin: &'static str) -> BuiltinResult<u64> {
    let end = cursor.checked_add(8).ok_or_else(|| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: corrupt FIG bundle"),
        )
    })?;
    let slice = bytes.get(*cursor..end).ok_or_else(|| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: truncated FIG bundle"),
        )
    })?;
    *cursor = end;
    Ok(u64::from_le_bytes(
        slice.try_into().expect("u64 slice length"),
    ))
}

fn apply_hgload_overrides(
    handles: &[FigureHandle],
    overrides: Option<&StructValue>,
) -> BuiltinResult<Value> {
    let mut cells = Vec::with_capacity(handles.len());
    for handle in handles {
        let mut old = StructValue::new();
        if let Some(overrides) = overrides {
            let mut set_args = Vec::new();
            for (name, value) in &overrides.fields {
                if let Ok(previous) =
                    get_properties(PlotHandle::Figure(*handle), Some(name), HGLOAD)
                {
                    old.insert(name.clone(), previous);
                    set_args.push(Value::String(name.clone()));
                    set_args.push(value.clone());
                }
            }
            if !set_args.is_empty() {
                set_properties(PlotHandle::Figure(*handle), &set_args, HGLOAD)?;
            }
        }
        cells.push(Value::Struct(old));
    }
    CellArray::new(cells, 1, handles.len())
        .map(Value::Cell)
        .map_err(|err| persistence_error(&ERROR_INTERNAL, HGLOAD, &err))
}

fn handles_value(handles: &[FigureHandle]) -> Value {
    if handles.len() == 1 {
        Value::Num(handles[0].as_u32() as f64)
    } else {
        Value::Tensor(Tensor {
            data: handles.iter().map(|h| h.as_u32() as f64).collect(),
            integer_data: None,
            shape: vec![1, handles.len()],
            rows: 1,
            cols: handles.len(),
            dtype: NumericDType::F64,
        })
    }
}

fn scalar_string(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(_) => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: character vector arguments must be 1-by-N"),
        )),
        Value::StringArray(_) => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: string array arguments must be scalar"),
        )),
        other => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected a string argument, got {other:?}"),
        )),
    }
}

fn validate_filename(filename: &str, builtin: &'static str) -> BuiltinResult<()> {
    if filename.trim().is_empty() {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: filename must not be empty"),
        ));
    }
    if filename.contains('\0') {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: filename must not contain NUL bytes"),
        ));
    }
    Ok(())
}

async fn write_bytes(path: &Path, payload: &[u8], builtin: &'static str) -> BuiltinResult<()> {
    let temp_path = temporary_output_path(path);
    let mut created_temp = false;
    let result: BuiltinResult<()> = async {
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        let mut file = options.open_async(&temp_path).await.map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to open temporary file '{}'",
                    temp_path.display()
                ),
                err,
            )
        })?;
        created_temp = true;
        file.write_all(payload).map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to write temporary file '{}'",
                    temp_path.display()
                ),
                err,
            )
        })?;
        file.flush_async().await.map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to flush temporary file '{}'",
                    temp_path.display()
                ),
                err,
            )
        })?;
        file.sync_all_async().await.map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to sync temporary file '{}'",
                    temp_path.display()
                ),
                err,
            )
        })?;
        drop(file);
        runmat_filesystem::rename_async(&temp_path, path)
            .await
            .map_err(|err| {
                persistence_error_with_source(
                    &ERROR_IO,
                    builtin,
                    format!(
                        "{builtin}: unable to replace '{}' with temporary file '{}'",
                        path.display(),
                        temp_path.display()
                    ),
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
        .unwrap_or_else(|| OsString::from("runmat-figure"));
    let mut temp_name = OsString::from(".");
    temp_name.push(file_name);
    temp_name.push(format!(
        ".{}.tmp",
        TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    path.with_file_name(temp_name)
}

fn persistence_error(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn persistence_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::figure::figure_builtin;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, figure_handles, reset_hold_state_for_run, reset_plot_state,
    };
    use futures::executor::block_on;

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
            integer_data: None,
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: NumericDType::F64,
        }
    }

    fn temp_path(stem: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_figure_persistence_{stem}_{}.fig",
            std::process::id()
        ));
        path
    }

    #[test]
    fn saveas_descriptor_covers_core_forms() {
        let labels = SAVEAS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"ok = saveas(fig, filename)"));
        assert!(labels.contains(&"ok = saveas(fig, filename, formattype)"));
    }

    #[test]
    fn savefig_round_trips_single_figure() {
        let _guard = setup();
        block_on(plot_builtin(vec![
            Value::Tensor(tensor(&[1.0, 2.0, 3.0])),
            Value::Tensor(tensor(&[1.0, 4.0, 9.0])),
        ]))
        .expect("plot");

        let path = temp_path("single");
        let _ = std::fs::remove_file(&path);
        block_on(savefig_builtin(vec![Value::String(
            path.to_string_lossy().into_owned(),
        )]))
        .expect("savefig");
        reset_plot_state();
        let value = block_on(openfig_builtin(vec![Value::String(
            path.to_string_lossy().into_owned(),
        )]))
        .expect("openfig");
        assert!(matches!(value, Value::Num(_)));
        assert_eq!(figure_handles().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn savefig_round_trips_multiple_figures() {
        let _guard = setup();
        let first = figure_builtin(vec![]).expect("figure one");
        block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))])).expect("plot one");
        let second = figure_builtin(vec![Value::String("next".into())]).expect("figure two");
        block_on(plot_builtin(vec![Value::Tensor(tensor(&[3.0, 2.0, 1.0]))])).expect("plot two");
        let handles = Value::Tensor(Tensor {
            data: vec![first, second],
            integer_data: None,
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: NumericDType::F64,
        });

        let path = temp_path("multi");
        let _ = std::fs::remove_file(&path);
        block_on(savefig_builtin(vec![
            handles,
            Value::String(path.to_string_lossy().into_owned()),
            Value::String("-v7.3".into()),
        ]))
        .expect("savefig multi");
        clear_figure(None).expect("clear");
        let value = block_on(openfig_builtin(vec![
            Value::String(path.to_string_lossy().into_owned()),
            Value::String("new".into()),
            Value::String("invisible".into()),
        ]))
        .expect("openfig");
        let Value::Tensor(tensor) = value else {
            panic!("expected handle tensor");
        };
        assert_eq!(tensor.data.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn savefig_reads_typed_integer_handle_arrays_exactly() {
        let _guard = setup();
        let first = figure_builtin(vec![]).expect("figure one");
        block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))])).expect("plot one");
        let second = figure_builtin(vec![Value::String("next".into())]).expect("figure two");
        block_on(plot_builtin(vec![Value::Tensor(tensor(&[3.0, 2.0, 1.0]))])).expect("plot two");
        let mut handles = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I64(vec![first as i64, second as i64]),
            vec![1, 2],
        )
        .expect("handle tensor");
        handles.data.clear();

        let path = temp_path("integer_handles");
        let _ = std::fs::remove_file(&path);
        block_on(savefig_builtin(vec![
            Value::Tensor(handles),
            Value::String(path.to_string_lossy().into_owned()),
        ]))
        .expect("savefig integer handles");
        clear_figure(None).expect("clear");
        let value = block_on(openfig_builtin(vec![Value::String(
            path.to_string_lossy().into_owned(),
        )]))
        .expect("openfig");
        let Value::Tensor(tensor) = value else {
            panic!("expected handle tensor");
        };
        assert_eq!(tensor.data.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hgsave_deduplicates_handles_resolving_to_same_figure() {
        let _guard = setup();
        let fig = figure_builtin(vec![]).expect("figure");
        block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))])).expect("plot");
        let handles = Value::Tensor(Tensor {
            data: vec![fig, fig],
            integer_data: None,
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: NumericDType::F64,
        });

        let path = temp_path("dedupe");
        let _ = std::fs::remove_file(&path);
        block_on(hgsave_builtin(vec![
            handles,
            Value::String(path.to_string_lossy().into_owned()),
        ]))
        .expect("hgsave duplicate handles");
        reset_plot_state();
        let value = block_on(hgload_builtin(vec![Value::String(
            path.to_string_lossy().into_owned(),
        )]))
        .expect("hgload");
        assert!(matches!(value, Value::Num(_)));
        assert_eq!(figure_handles().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn saveas_png_uses_renderer_fast_path() {
        let _guard = setup();
        block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))])).expect("plot");
        let mut path = std::env::temp_dir();
        path.push(format!("runmat_saveas_png_{}.png", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let handle = current_figure_handle().as_u32() as f64;
        block_on(saveas_builtin(vec![
            Value::Num(handle),
            Value::String(path.to_string_lossy().into_owned()),
            Value::String("png".into()),
        ]))
        .expect("saveas png");
        let bytes = std::fs::read(&path).expect("read png");
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn saveas_rejects_unsupported_or_mismatched_extensions() {
        let _guard = setup();
        let fig = figure_builtin(vec![]).expect("figure");
        let jpg_err = block_on(saveas_builtin(vec![
            Value::Num(fig),
            Value::String("plot.jpg".into()),
        ]))
        .expect_err("jpg is unsupported");
        assert!(jpg_err.message().contains("not available yet"));

        let mismatch_err = block_on(saveas_builtin(vec![
            Value::Num(fig),
            Value::String("plot.fig".into()),
            Value::String("png".into()),
        ]))
        .expect_err("extension/format mismatch");
        assert!(mismatch_err.message().contains("does not match"));
    }

    #[test]
    fn openfig_rejects_reuse_mode_until_tracking_exists() {
        let _guard = setup();
        let err = block_on(openfig_builtin(vec![
            Value::String("missing.fig".into()),
            Value::String("reuse".into()),
        ]))
        .expect_err("reuse is not implemented");
        assert!(err.message().contains("reuse mode"));
    }

    #[test]
    fn decode_figure_bundle_rejects_hostile_counts_before_allocating() {
        let mut bytes = FIG_MAGIC.to_vec();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        let err = decode_figure_bundle(&bytes, OPENFIG).expect_err("hostile count");
        assert!(err.message().contains("too many figures"));
    }

    #[test]
    fn decode_figure_bundle_rejects_oversized_payload_length() {
        let mut bytes = FIG_MAGIC.to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(
            &(ReplayLimits::default().max_scene_payload_bytes as u64 + 1).to_le_bytes(),
        );
        let err = decode_figure_bundle(&bytes, OPENFIG).expect_err("oversized payload");
        assert!(err.message().contains("exceeds maximum scene payload size"));
    }

    #[test]
    fn hgload_applies_property_overrides_and_returns_old_values() {
        let _guard = setup();
        let fig = figure_builtin(vec![
            Value::String("Name".into()),
            Value::String("Before".into()),
        ])
        .expect("figure");
        let path = temp_path("hgload");
        let _ = std::fs::remove_file(&path);
        block_on(hgsave_builtin(vec![
            Value::Num(fig),
            Value::String(path.to_string_lossy().into_owned()),
        ]))
        .expect("hgsave");
        clear_figure(None).expect("clear");
        let mut overrides = StructValue::new();
        overrides.insert("Name", Value::String("After".into()));
        overrides.insert("NotAProperty", Value::Num(1.0));
        let _outputs = crate::output_count::push_output_count(Some(2));
        let result = block_on(hgload_builtin(vec![
            Value::String(path.to_string_lossy().into_owned()),
            Value::Struct(overrides),
        ]))
        .expect("hgload");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);
        let Value::Cell(cell) = &outputs[1] else {
            panic!("expected old values cell");
        };
        let Value::Struct(old) = &cell.data[0] else {
            panic!("expected old values struct");
        };
        assert_eq!(
            old.fields.get("Name"),
            Some(&Value::String("Before".into()))
        );
        assert!(!old.fields.contains_key("NotAProperty"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn savefig_rejects_non_fig_extension() {
        let _guard = setup();
        let err = block_on(savefig_builtin(vec![Value::String("plot.png".into())]))
            .expect_err("savefig requires fig");
        assert!(err.message().contains("expected a .fig filename"));
    }
}
