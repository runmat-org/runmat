//! MATLAB-compatible `scatter3` builtin.

use glam::{Vec3, Vec4};
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
#[cfg(test)]
use runmat_builtins::Tensor;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::scatter2::{ScatterAttributeBuffer, ScatterColorBuffer};
use runmat_plot::gpu::scatter3::{Scatter3GpuInputs, Scatter3GpuParams};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::scatter::MarkerStyle;
use runmat_plot::plots::scatter3::Scatter3GpuStyle;
use runmat_plot::plots::surface::ColorMap;
use runmat_plot::plots::LineStyle;
use runmat_plot::plots::Scatter3Plot;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use std::convert::TryFrom;

use super::common::{gather_tensor_from_gpu, numeric_triplet};
use super::gpu_helpers::axis_bounds;
use super::op_common::line_inputs::NumericInput as ScatterInput;
use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::perf::scatter3_lod_stride;
use super::plotting_error;
use super::point::{
    convert_rgb_color_matrix, convert_scalar_color_values, convert_size_vector,
    default_marker_diameter_px, map_scalar_values_to_colors, marker_area_points2_to_diameter_px,
    validate_gpu_color_matrix, validate_gpu_vector_length, PointArgs, PointColorArg, PointGpuColor,
    PointSizeArg,
};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{LineStyleParseOptions, MarkerColor};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const BUILTIN_NAME: &str = "scatter3";

const SCATTER3_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered 3-D scatter plot.",
}];

const SCATTER3_INPUTS_X_Y_Z: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
];

const SCATTER3_INPUTS_X_Y_Z_S: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker area specification.",
    },
];

const SCATTER3_INPUTS_X_Y_Z_S_C: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker area specification.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Color specification (uniform, per-point scalar, or RGB matrix).",
    },
];

const SCATTER3_INPUTS_X_Y_Z_STYLE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "lineSpec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker/line style shorthand.",
    },
];

const SCATTER3_INPUTS_X_Y_Z_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value marker style properties.",
    },
];

const SCATTER3_INPUTS_AX_X_Y_Z: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
];

const SCATTER3_INPUTS_AX_X_Y_Z_S: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker area specification.",
    },
];

const SCATTER3_INPUTS_AX_X_Y_Z_S_C: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker area specification.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Color specification (uniform, per-point scalar, or RGB matrix).",
    },
];

const SCATTER3_INPUTS_AX_X_Y_Z_STYLE: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "lineSpec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker/line style shorthand.",
    },
];

const SCATTER3_INPUTS_AX_X_Y_Z_PROPS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value marker style properties.",
    },
];

const SCATTER3_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z)",
        inputs: &SCATTER3_INPUTS_X_Y_Z,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z, S)",
        inputs: &SCATTER3_INPUTS_X_Y_Z_S,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z, S, C)",
        inputs: &SCATTER3_INPUTS_X_Y_Z_S_C,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z, LineSpec)",
        inputs: &SCATTER3_INPUTS_X_Y_Z_STYLE,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z, Name, Value, ...)",
        inputs: &SCATTER3_INPUTS_X_Y_Z_PROPS,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z, S)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z_S,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z, S, C)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z_S_C,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z, LineSpec)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z_STYLE,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z, Name, Value, ...)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z_PROPS,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
];

pub const SCATTER3_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTER3.INVALID_ARGUMENT",
    identifier: Some("RunMat:scatter3:InvalidArgument"),
    when: "Input data, axes targeting, or marker style arguments are invalid.",
    message: "scatter3: invalid argument",
};

pub const SCATTER3_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTER3.INTERNAL",
    identifier: Some("RunMat:scatter3:Internal"),
    when: "Internal 3-D scatter construction or rendering fails unexpectedly.",
    message: "scatter3: internal operation failed",
};

const SCATTER3_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SCATTER3_ERROR_INVALID_ARGUMENT, SCATTER3_ERROR_INTERNAL];

pub const SCATTER3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SCATTER3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SCATTER3_ERRORS,
};

fn scatter3_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_scatter3_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    scatter3_error_with_detail(&SCATTER3_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_scatter3_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    scatter3_error_with_detail(&SCATTER3_ERROR_INTERNAL, err.message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::scatter3")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "scatter3",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    // Plotting is a sink, but can consume gpuArray inputs zero-copy when a shared WGPU context exists.
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Rendering terminates fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::scatter3")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "scatter3",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "scatter3 terminates fusion graphs and performs host/WebGPU rendering.",
};

#[runtime_builtin(
    name = "scatter3",
    category = "plotting",
    summary = "Create 3-D scatter plots from x/y/z point data.",
    keywords = "scatter3,plotting,3d,pointcloud",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::scatter3::SCATTER3_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::scatter3"
)]
pub async fn scatter3_builtin(
    x: Value,
    y: Value,
    z: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<f64> {
    let mut args = vec![x, y, z];
    args.extend(rest);
    let (axes_target, mut args) =
        split_leading_axes_handle(args, BUILTIN_NAME).map_err(map_scatter3_invalid_argument)?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_scatter3_invalid_argument)?;
    if args.len() < 3 {
        return Err(scatter3_error_with_detail(
            &SCATTER3_ERROR_INVALID_ARGUMENT,
            "expected X, Y, and Z data after axes handle",
        ));
    }
    let x = args.remove(0);
    let y = args.remove(0);
    let z = args.remove(0);
    let rest = args;
    let style_args = PointArgs::parse(rest, LineStyleParseOptions::scatter3())
        .map_err(map_scatter3_invalid_argument)?;
    let mut x_input =
        Some(ScatterInput::from_value(x, BUILTIN_NAME).map_err(map_scatter3_invalid_argument)?);
    let mut y_input =
        Some(ScatterInput::from_value(y, BUILTIN_NAME).map_err(map_scatter3_invalid_argument)?);
    let mut z_input =
        Some(ScatterInput::from_value(z, BUILTIN_NAME).map_err(map_scatter3_invalid_argument)?);
    let opts = PlotRenderOptions {
        title: "3-D Scatter",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let style_args = style_args.clone();
        let point_count = x_input.as_ref().map(|input| input.len()).unwrap_or(0);
        let mut resolved_style = resolve_scatter3_style(point_count, &style_args, "scatter3")
            .map_err(map_scatter3_invalid_argument)?;
        let x_arg = x_input.take().expect("scatter3 x consumed once");
        let y_arg = y_input.take().expect("scatter3 y consumed once");
        let z_arg = z_input.take().expect("scatter3 z consumed once");

        if let (Some(x_gpu), Some(y_gpu), Some(z_gpu)) =
            (x_arg.gpu_handle(), y_arg.gpu_handle(), z_arg.gpu_handle())
        {
            if !resolved_style.requires_cpu {
                match build_scatter3_gpu_plot(x_gpu, y_gpu, z_gpu, &resolved_style) {
                    Ok(plot) => {
                        let plot_index = figure.add_scatter3_plot_on_axes(plot, axes);
                        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
                        return Ok(());
                    }
                    Err(err) => {
                        warn!("scatter3 GPU path unavailable: {err}");
                    }
                }
            }
        }

        let (x_tensor, y_tensor, z_tensor) = (
            x_arg
                .into_tensor("scatter3")
                .map_err(map_scatter3_invalid_argument)?,
            y_arg
                .into_tensor("scatter3")
                .map_err(map_scatter3_invalid_argument)?,
            z_arg
                .into_tensor("scatter3")
                .map_err(map_scatter3_invalid_argument)?,
        );
        let (x_vals, y_vals, z_vals) = numeric_triplet(x_tensor, y_tensor, z_tensor, "scatter3")
            .map_err(map_scatter3_invalid_argument)?;
        let scatter = build_scatter3_plot(x_vals, y_vals, z_vals, &mut resolved_style)
            .map_err(map_scatter3_invalid_argument)?;
        let plot_index = figure.add_scatter3_plot_on_axes(scatter, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_scatter3_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_scatter3_internal(err));
    }
    Ok(handle)
}

const DEFAULT_SCATTER3_LABEL: &str = "Data";

fn scatter3_err(message: impl Into<String>) -> RuntimeError {
    plotting_error(BUILTIN_NAME, message)
}

fn default_color() -> Vec4 {
    Vec4::new(0.1, 0.6, 0.9, 0.9)
}

#[derive(Clone, Debug)]
struct Scatter3ResolvedStyle {
    uniform_color: Vec4,
    edge_color: Vec4,
    edge_thickness: f32,
    marker_style: MarkerStyle,
    point_size: f32,
    filled: bool,
    per_point_sizes: Option<Vec<f32>>,
    per_point_colors: Option<Vec<Vec4>>,
    color_values: Option<Vec<f64>>,
    color_limits: Option<(f64, f64)>,
    gpu_sizes: Option<GpuTensorHandle>,
    gpu_colors: Option<PointGpuColor>,
    colormap: ColorMap,
    marker_face_flat: bool,
    marker_edge_flat: bool,
    requires_cpu: bool,
    label: String,
}

fn resolve_scatter3_style(
    point_count: usize,
    args: &PointArgs,
    context: &'static str,
) -> BuiltinResult<Scatter3ResolvedStyle> {
    let mut style = Scatter3ResolvedStyle {
        uniform_color: default_color(),
        edge_color: default_color(),
        edge_thickness: 1.0,
        marker_style: MarkerStyle::Circle,
        point_size: default_marker_diameter_px(),
        filled: args.filled,
        per_point_sizes: None,
        per_point_colors: None,
        color_values: None,
        color_limits: None,
        gpu_sizes: None,
        gpu_colors: None,
        colormap: ColorMap::Parula,
        marker_face_flat: false,
        marker_edge_flat: false,
        requires_cpu: false,
        label: DEFAULT_SCATTER3_LABEL.to_string(),
    };

    if let Some(label) = args.style.label.clone() {
        style.label = label;
    }

    let appearance = &args.style.appearance;
    style.uniform_color = appearance.color;
    style.edge_color = appearance.color;
    style.edge_thickness = appearance.line_width.max(0.1);

    if let PointColorArg::Uniform(color) = &args.color {
        style.uniform_color = *color;
        style.edge_color = *color;
    }

    if appearance.marker.is_none() {
        style.edge_color = style.uniform_color;
    }

    if let Some(marker) = appearance.marker.as_ref() {
        style.marker_style = marker.kind.to_plot_marker();
        if let Some(size) = marker.size {
            style.point_size = marker_area_points2_to_diameter_px(size as f64);
        }
        if matches!(marker.edge_color, MarkerColor::Flat) {
            style.marker_edge_flat = true;
        } else {
            style.edge_color =
                resolve_marker_color(&marker.edge_color, style.edge_color, style.uniform_color);
        }
        match &marker.face_color {
            MarkerColor::Flat => {
                style.marker_face_flat = true;
                style.filled = true;
            }
            MarkerColor::None => {
                style.filled = false;
            }
            _ => {
                let face_color = resolve_marker_color(
                    &marker.face_color,
                    style.uniform_color,
                    style.uniform_color,
                );
                if matches!(marker.face_color, MarkerColor::Color(_) | MarkerColor::Auto) {
                    style.uniform_color = face_color;
                }
            }
        }
    }

    if let PointSizeArg::Scalar(size) = &args.size {
        style.point_size = marker_area_points2_to_diameter_px(*size as f64);
    }

    if let Some(value) = args.size.value() {
        match value {
            Value::GpuTensor(handle) => {
                validate_gpu_vector_length(handle, point_count, context)?;
                style.gpu_sizes = Some(handle.clone());
            }
            _ => {
                style.per_point_sizes = Some(convert_size_vector(value, point_count, context)?);
            }
        }
    }

    match &args.color {
        PointColorArg::ScalarValues(value) => {
            let scalars = convert_scalar_color_values(value, point_count, context)?;
            let (colors, limits) = map_scalar_values_to_colors(&scalars, style.colormap);
            style.color_values = Some(scalars);
            style.per_point_colors = Some(colors);
            style.color_limits = Some(limits);
        }
        PointColorArg::RgbMatrix(value) => match value {
            Value::GpuTensor(handle) => {
                let components = validate_gpu_color_matrix(handle, point_count, context)?;
                style.gpu_colors = Some(PointGpuColor {
                    handle: handle.clone(),
                    components,
                });
            }
            _ => {
                style.per_point_colors =
                    Some(convert_rgb_color_matrix(value, point_count, context)?);
            }
        },
        _ => {}
    }

    if style.per_point_colors.is_some() || style.color_values.is_some() {
        style.filled = true;
    }

    if style.marker_face_flat {
        if style.per_point_colors.is_none() && style.gpu_colors.is_none() {
            return Err(scatter3_err(format!(
                "{context}: MarkerFaceColor 'flat' requires per-point color data (C argument)"
            )));
        }
        style.filled = true;
    }

    if style.marker_edge_flat && style.per_point_colors.is_none() && style.gpu_colors.is_none() {
        return Err(scatter3_err(format!(
            "{context}: MarkerEdgeColor 'flat' requires per-point color data (C argument)"
        )));
    }

    if args.style.appearance.line_style != LineStyle::Solid && args.style.line_style_explicit {
        style.requires_cpu = true;
    }
    if args.style.line_style_order.is_some() {
        style.requires_cpu = true;
    }
    style.requires_cpu |= args.style.requires_cpu_fallback;

    Ok(style)
}

fn resolve_marker_color(marker_color: &MarkerColor, fallback: Vec4, default_base: Vec4) -> Vec4 {
    match marker_color {
        MarkerColor::Auto => fallback,
        MarkerColor::None => Vec4::new(default_base.x, default_base.y, default_base.z, 0.0),
        MarkerColor::Flat => fallback,
        MarkerColor::Color(color) => *color,
    }
}

fn build_scatter3_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    style: &mut Scatter3ResolvedStyle,
) -> BuiltinResult<Scatter3Plot> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(scatter3_err(
            "scatter3: X, Y, and Z inputs must have identical lengths",
        ));
    }

    ensure_scatter3_host_metadata(style, x.len())?;

    let points: Vec<Vec3> = x
        .iter()
        .zip(y.iter())
        .zip(z.iter())
        .map(|((x, y), z)| Vec3::new(*x as f32, *y as f32, *z as f32))
        .collect();

    let mut scatter = Scatter3Plot::new(points)
        .map_err(|err| scatter3_err(format!("scatter3: {err}")))?
        .with_point_size(style.point_size)
        .with_color(style.uniform_color)
        .with_label(style.label.clone());
    scatter.set_marker_style(style.marker_style);
    scatter.set_filled(style.filled);
    scatter.set_edge_color(style.edge_color);
    scatter.set_edge_thickness(style.edge_thickness);
    scatter.set_edge_color_from_vertex(style.marker_edge_flat);
    if let Some(sizes) = style.per_point_sizes.take() {
        scatter.set_point_sizes(sizes);
    }
    if let Some(colors) = style.per_point_colors.take() {
        scatter = scatter
            .with_colors(colors)
            .map_err(|e| scatter3_err(format!("scatter3: {e}")))?;
    }
    Ok(scatter)
}

fn build_scatter3_gpu_plot(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    z: &GpuTensorHandle,
    style: &Scatter3ResolvedStyle,
) -> BuiltinResult<Scatter3Plot> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(BUILTIN_NAME)?;

    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| scatter3_err("scatter3: unable to export GPU X data"))?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| scatter3_err("scatter3: unable to export GPU Y data"))?;
    let z_ref = runmat_accelerate_api::export_wgpu_buffer(z)
        .ok_or_else(|| scatter3_err("scatter3: unable to export GPU Z data"))?;

    if x_ref.len == 0 {
        return Err(scatter3_err("scatter3: empty input tensor"));
    }
    if x_ref.len != y_ref.len || x_ref.len != z_ref.len {
        return Err(scatter3_err(
            "scatter3: X, Y, and Z inputs must have identical lengths",
        ));
    }
    if x_ref.precision != y_ref.precision || x_ref.precision != z_ref.precision {
        return Err(scatter3_err(
            "scatter3: gpuArray inputs must have matching precision",
        ));
    }
    let point_count = x_ref.len;
    let len_u32 = u32::try_from(point_count)
        .map_err(|_| scatter3_err("scatter3: point count exceeds supported range"))?;
    let scalar = ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64);
    let bounds = build_gpu_bounds(x, y, z)?;
    let extent_hint = scatter3_extent_hint(&bounds);
    let lod_stride = scatter3_lod_stride(len_u32, extent_hint);

    let size_buffer = build_scatter3_size_buffer(style, point_count)?;
    let color_buffer = build_scatter3_color_buffer(style, point_count)?;

    let inputs = Scatter3GpuInputs {
        x_buffer: x_ref.buffer.clone(),
        y_buffer: y_ref.buffer.clone(),
        z_buffer: z_ref.buffer.clone(),
        len: len_u32,
        scalar,
    };
    let params = Scatter3GpuParams {
        color: style.uniform_color,
        point_size: style.point_size,
        sizes: size_buffer,
        colors: color_buffer,
        lod_stride,
    };

    let gpu_vertices = runmat_plot::gpu::scatter3::pack_vertices_from_xyz(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| scatter3_err(format!("scatter3: failed to build GPU vertices: {e}")))?;

    let drawn_points = gpu_vertices.vertex_count;
    let gpu_style = Scatter3GpuStyle {
        color: style.uniform_color,
        edge_color: style.edge_color,
        edge_thickness: style.edge_thickness,
        marker_style: style.marker_style,
        filled: style.filled,
        has_per_point_colors: style.per_point_colors.is_some() || style.gpu_colors.is_some(),
        edge_from_vertex_colors: style.marker_edge_flat,
    };

    Ok(Scatter3Plot::from_gpu_buffer(
        gpu_vertices,
        drawn_points,
        gpu_style,
        style.point_size,
        bounds,
    )
    .with_gpu_source_inputs(inputs)
    .with_label(style.label.clone()))
}

fn build_gpu_bounds(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    z: &GpuTensorHandle,
) -> BuiltinResult<BoundingBox> {
    let (min_x, max_x) = axis_bounds(x, BUILTIN_NAME)?;
    let (min_y, max_y) = axis_bounds(y, BUILTIN_NAME)?;
    let (min_z, max_z) = axis_bounds(z, BUILTIN_NAME)?;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, min_z),
        Vec3::new(max_x, max_y, max_z),
    ))
}

fn scatter3_extent_hint(bounds: &BoundingBox) -> f32 {
    bounds.size().length().max(1.0)
}

fn build_scatter3_size_buffer(
    style: &Scatter3ResolvedStyle,
    point_count: usize,
) -> BuiltinResult<ScatterAttributeBuffer> {
    if let Some(handle) = style.gpu_sizes.as_ref() {
        let exported = runmat_accelerate_api::export_wgpu_buffer(handle)
            .ok_or_else(|| scatter3_err("scatter3: unable to export GPU marker sizes"))?;
        if exported.len != point_count {
            return Err(scatter3_err(format!(
                "scatter3: marker size array must have {point_count} elements (got {})",
                exported.len
            )));
        }
        if exported.precision != ProviderPrecision::F32 {
            return Err(scatter3_err(
                "scatter3: GPU marker sizes must be single-precision (cast before plotting)",
            ));
        }
        return Ok(ScatterAttributeBuffer::Gpu(exported.buffer));
    }
    if let Some(sizes) = style.per_point_sizes.as_ref() {
        if sizes.is_empty() {
            Ok(ScatterAttributeBuffer::None)
        } else {
            Ok(ScatterAttributeBuffer::Host(sizes.clone()))
        }
    } else {
        Ok(ScatterAttributeBuffer::None)
    }
}

fn build_scatter3_color_buffer(
    style: &Scatter3ResolvedStyle,
    point_count: usize,
) -> BuiltinResult<ScatterColorBuffer> {
    if let Some(gpu_color) = style.gpu_colors.as_ref() {
        let exported = runmat_accelerate_api::export_wgpu_buffer(&gpu_color.handle)
            .ok_or_else(|| scatter3_err("scatter3: unable to export GPU color data"))?;
        let expected = point_count * gpu_color.components.stride() as usize;
        if exported.len != expected {
            return Err(scatter3_err(format!(
                "scatter3: color array must contain {} elements (got {})",
                expected, exported.len
            )));
        }
        if exported.precision != ProviderPrecision::F32 {
            return Err(scatter3_err(
                "scatter3: GPU color arrays must be single-precision (cast before plotting)",
            ));
        }
        return Ok(ScatterColorBuffer::Gpu {
            buffer: exported.buffer,
            components: gpu_color.components.stride(),
        });
    }

    if let Some(colors) = style.per_point_colors.as_ref() {
        if colors.is_empty() {
            Ok(ScatterColorBuffer::None)
        } else {
            let data = colors.iter().map(|c| c.to_array()).collect();
            Ok(ScatterColorBuffer::Host(data))
        }
    } else {
        Ok(ScatterColorBuffer::None)
    }
}

fn ensure_scatter3_host_metadata(
    style: &mut Scatter3ResolvedStyle,
    point_count: usize,
) -> BuiltinResult<()> {
    if style.per_point_sizes.is_none() {
        if let Some(handle) = style.gpu_sizes.clone() {
            let tensor = gather_tensor_from_gpu(handle, "scatter3")?;
            let value = Value::Tensor(tensor);
            style.per_point_sizes = Some(convert_size_vector(&value, point_count, "scatter3")?);
        }
    }
    if style.per_point_colors.is_none() {
        if let Some(gpu_color) = style.gpu_colors.as_ref() {
            let tensor = gather_tensor_from_gpu(gpu_color.handle.clone(), "scatter3")?;
            let value = Value::Tensor(tensor);
            style.per_point_colors =
                Some(convert_rgb_color_matrix(&value, point_count, "scatter3")?);
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::style::LineStyleParseOptions;
    use super::*;
    use crate::builtins::plotting::state::current_axes_handle_for_figure;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_builtins::Value;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_plot::plots::PlotElement;

    fn setup_plot_tests() {
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
    }

    fn scatter3_builtin(x: Value, y: Value, z: Value, rest: Vec<Value>) -> BuiltinResult<f64> {
        block_on(super::scatter3_builtin(x, y, z, rest))
    }

    fn test_style() -> Scatter3ResolvedStyle {
        Scatter3ResolvedStyle {
            uniform_color: default_color(),
            edge_color: default_color(),
            edge_thickness: 1.0,
            marker_style: MarkerStyle::Circle,
            point_size: default_marker_diameter_px(),
            filled: false,
            per_point_sizes: None,
            per_point_colors: None,
            color_values: None,
            color_limits: None,
            gpu_sizes: None,
            gpu_colors: None,
            colormap: ColorMap::Parula,
            marker_face_flat: false,
            marker_edge_flat: false,
            requires_cpu: false,
            label: DEFAULT_SCATTER3_LABEL.to_string(),
        }
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    fn assert_plotting_unavailable(err: &RuntimeError) {
        let lower = err.to_string().to_lowercase();
        assert!(
            lower.contains("plotting is unavailable") || lower.contains("non-main thread"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn build_scatter3_requires_equal_lengths() {
        setup_plot_tests();
        assert!(build_scatter3_plot(vec![1.0], vec![], vec![1.0], &mut test_style()).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter3_builtin_emits_result_or_backend_error() {
        setup_plot_tests();
        let out = scatter3_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Vec::new(),
        );
        if let Err(flow) = out {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter3_accepts_per_point_sizes() {
        setup_plot_tests();
        let rest = vec![Value::Tensor(Tensor {
            data: vec![1.0, 2.0],
            shape: vec![2],
            rows: 2,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        })];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter3()).unwrap();
        let style = resolve_scatter3_style(2, &args, "scatter3").expect("style");
        assert!(style.per_point_sizes.is_some());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter3_resolves_marker_style_color_and_size() {
        setup_plot_tests();
        let rest = vec![
            Value::Num(12.0),
            Value::String("g".into()),
            Value::String("filled".into()),
            Value::String("Marker".into()),
            Value::String("s".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter3()).unwrap();
        let style = resolve_scatter3_style(3, &args, "scatter3").expect("style");
        assert_eq!(style.marker_style, MarkerStyle::Square);
        assert!(style.filled);
        assert!(
            (style.point_size - marker_area_points2_to_diameter_px(12.0)).abs() < 1e-5,
            "size was {}",
            style.point_size
        );
        assert!(style.uniform_color.y > style.uniform_color.x);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter3_single_rgb_row_sets_uniform_face_and_edge_color() {
        setup_plot_tests();
        let rest = vec![
            Value::Num(49.0),
            Value::Tensor(Tensor {
                data: vec![0.9, 0.2, 0.2],
                shape: vec![1, 3],
                rows: 1,
                cols: 3,
                dtype: runmat_builtins::NumericDType::F64,
            }),
            Value::String("filled".into()),
            Value::String("Marker".into()),
            Value::String("o".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter3()).unwrap();
        let style = resolve_scatter3_style(4, &args, "scatter3").expect("style");
        assert!((style.uniform_color.x - 0.9).abs() < 1e-6);
        assert!((style.uniform_color.y - 0.2).abs() < 1e-6);
        assert!((style.uniform_color.z - 0.2).abs() < 1e-6);
        assert!((style.edge_color.x - 0.9).abs() < 1e-6);
        assert!((style.edge_color.y - 0.2).abs() < 1e-6);
        assert!((style.edge_color.z - 0.2).abs() < 1e-6);
        assert!(!style.marker_edge_flat);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter3_applies_display_name() {
        setup_plot_tests();
        let rest = vec![
            Value::String("DisplayName".into()),
            Value::String("Cloud A".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter3()).unwrap();
        let mut style = resolve_scatter3_style(2, &args, "scatter3").expect("style");
        let plot = build_scatter3_plot(vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0], &mut style)
            .expect("plot");
        assert_eq!(plot.label.as_deref(), Some("Cloud A"));
    }

    #[test]
    fn scatter3_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(
                &[Type::tensor(), Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new())
            ),
            Type::Num
        );
    }

    #[test]
    fn scatter3_accepts_leading_axes_handle() {
        let _guard = crate::builtins::plotting::tests::lock_plot_registry();
        setup_plot_tests();
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        let _ = scatter3_builtin(
            Value::Num(ax),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
            vec![Value::Tensor(tensor_from(&[2.0, 3.0]))],
        );
        let fig = clone_figure(fig_handle).unwrap();
        assert_eq!(fig.plot_axes_indices(), &[1]);
    }

    #[test]
    fn scatter3_accepts_scalar_point() {
        let _guard = crate::builtins::plotting::tests::lock_plot_registry();
        setup_plot_tests();
        let _ = scatter3_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Vec::new(),
        );
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Scatter3(plot) = fig.plots().next().unwrap() else {
            panic!("expected scatter3")
        };
        assert_eq!(plot.points, vec![glam::Vec3::new(1.0, 2.0, 3.0)]);
    }

    #[test]
    fn scatter3_descriptor_signatures_cover_supported_forms() {
        let labels: Vec<&str> = SCATTER3_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = scatter3(X, Y, Z)"));
        assert!(labels.contains(&"h = scatter3(X, Y, Z, S, C)"));
        assert!(labels.contains(&"h = scatter3(X, Y, Z, Name, Value, ...)"));
        assert!(labels.contains(&"h = scatter3(ax, X, Y, Z)"));
        assert!(labels.contains(&"h = scatter3(ax, X, Y, Z, Name, Value, ...)"));
    }

    #[test]
    fn scatter3_missing_post_axes_input_uses_stable_identifier() {
        let _guard = crate::builtins::plotting::tests::lock_plot_registry();
        setup_plot_tests();
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        let err = scatter3_builtin(
            Value::Num(ax),
            Value::Num(1.0),
            Value::String("filled".into()),
            Vec::new(),
        )
        .expect_err("missing z after axes handle should fail");
        assert_eq!(err.identifier(), SCATTER3_ERROR_INVALID_ARGUMENT.identifier);
    }
}
