use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::axis::{AxisData, OwnedAxisData};
use runmat_plot::plots::QuiverPlot;
use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_line_style_args, value_as_f64, LineStyleParseOptions};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "quiver";
type QuiverArgs = (
    Option<usize>,
    QuiverCoordinateInput,
    QuiverCoordinateInput,
    Value,
    Value,
    Vec<Value>,
);
type QuiverComponents = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

const QUIVER_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered quiver plot.",
}];

const QUIVER_INPUTS_U_V: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
];

const QUIVER_INPUTS_X_Y_U_V: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
];

const QUIVER_INPUTS_X_Y_U_V_STYLE: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
    BuiltinParamDescriptor {
        name: "lineSpec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line style/color shorthand.",
    },
];

const QUIVER_INPUTS_X_Y_U_V_PROPS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value quiver style properties.",
    },
];

const QUIVER_INPUTS_AX_U_V: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
];

const QUIVER_INPUTS_AX_X_Y_U_V: [BuiltinParamDescriptor; 5] = [
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
        description: "X coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
];

const QUIVER_INPUTS_AX_X_Y_U_V_STYLE: [BuiltinParamDescriptor; 6] = [
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
        description: "X coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
    BuiltinParamDescriptor {
        name: "lineSpec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line style/color shorthand.",
    },
];

const QUIVER_INPUTS_AX_X_Y_U_V_PROPS: [BuiltinParamDescriptor; 6] = [
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
        description: "X coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates (vector or matrix).",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value quiver style properties.",
    },
];

const QUIVER_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = quiver(U, V)",
        inputs: &QUIVER_INPUTS_U_V,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(X, Y, U, V)",
        inputs: &QUIVER_INPUTS_X_Y_U_V,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(X, Y, U, V, LineSpec)",
        inputs: &QUIVER_INPUTS_X_Y_U_V_STYLE,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(X, Y, U, V, Name, Value, ...)",
        inputs: &QUIVER_INPUTS_X_Y_U_V_PROPS,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(ax, U, V)",
        inputs: &QUIVER_INPUTS_AX_U_V,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(ax, X, Y, U, V)",
        inputs: &QUIVER_INPUTS_AX_X_Y_U_V,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(ax, X, Y, U, V, LineSpec)",
        inputs: &QUIVER_INPUTS_AX_X_Y_U_V_STYLE,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(ax, X, Y, U, V, Name, Value, ...)",
        inputs: &QUIVER_INPUTS_AX_X_Y_U_V_PROPS,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
];

pub const QUIVER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUIVER.INVALID_ARGUMENT",
    identifier: Some("RunMat:quiver:InvalidArgument"),
    when: "Input data, axes targeting, or quiver style arguments are invalid.",
    message: "quiver: invalid argument",
};

pub const QUIVER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUIVER.INTERNAL",
    identifier: Some("RunMat:quiver:Internal"),
    when: "Internal quiver construction or rendering fails unexpectedly.",
    message: "quiver: internal operation failed",
};

const QUIVER_ERRORS: [BuiltinErrorDescriptor; 2] =
    [QUIVER_ERROR_INVALID_ARGUMENT, QUIVER_ERROR_INTERNAL];

pub const QUIVER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &QUIVER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &QUIVER_ERRORS,
};

fn quiver_error_with_detail(
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

fn map_quiver_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    quiver_error_with_detail(&QUIVER_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_quiver_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    quiver_error_with_detail(&QUIVER_ERROR_INTERNAL, err.message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::quiver")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "quiver",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "quiver is a plotting sink; GPU inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::quiver")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "quiver",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "quiver performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "quiver",
    category = "plotting",
    summary = "Render 2-D quiver vector-field plots.",
    keywords = "quiver,plotting,vector field,arrows",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::quiver::QUIVER_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::quiver"
)]
pub async fn quiver_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target_axes, x, y, u, v, rest) =
        parse_quiver_args(args).map_err(map_quiver_invalid_argument)?;
    let parsed = parse_quiver_style_args(&rest).map_err(map_quiver_invalid_argument)?;
    let mut x_in = Some(x);
    let mut y_in = Some(y);
    let mut u_in =
        Some(NumericInput::from_value(u, BUILTIN_NAME).map_err(map_quiver_invalid_argument)?);
    let mut v_in =
        Some(NumericInput::from_value(v, BUILTIN_NAME).map_err(map_quiver_invalid_argument)?);
    let opts = PlotRenderOptions {
        title: "Quiver",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let axes = target_axes.unwrap_or(axes);
        if let (Some(u_gpu), Some(v_gpu)) = (
            u_in.as_ref().and_then(NumericInput::gpu_handle),
            v_in.as_ref().and_then(NumericInput::gpu_handle),
        ) {
            if let Ok(plot) = build_quiver_gpu_plot(
                x_in.as_ref().expect("x available"),
                y_in.as_ref().expect("y available"),
                u_gpu,
                v_gpu,
                &parsed,
                parsed.label.as_deref().unwrap_or("Data"),
            ) {
                let plot_index = figure.add_quiver_plot_on_axes(plot, axes);
                *plot_index_slot.borrow_mut() = Some((axes, plot_index));
                return Ok(());
            }
        }
        let x_tensor = x_in
            .take()
            .expect("x consumed")
            .into_tensor(BUILTIN_NAME)
            .map_err(map_quiver_invalid_argument)?;
        let y_tensor = y_in
            .take()
            .expect("y consumed")
            .into_tensor(BUILTIN_NAME)
            .map_err(map_quiver_invalid_argument)?;
        let u_tensor = u_in
            .take()
            .expect("u consumed")
            .into_tensor(BUILTIN_NAME)
            .map_err(map_quiver_invalid_argument)?;
        let v_tensor = v_in
            .take()
            .expect("v consumed")
            .into_tensor(BUILTIN_NAME)
            .map_err(map_quiver_invalid_argument)?;
        let (x_vals, y_vals, u_vals, v_vals) =
            materialize_quiver_components(x_tensor, y_tensor, u_tensor, v_tensor, BUILTIN_NAME)
                .map_err(map_quiver_invalid_argument)?;
        let label = parsed.label.clone().unwrap_or_else(|| "Data".into());
        let plot = QuiverPlot::new(x_vals, y_vals, u_vals, v_vals)
            .map_err(|e| plotting_error(BUILTIN_NAME, format!("quiver: {e}")))?
            .with_style(
                parsed.color,
                parsed.line_width,
                parsed.scale,
                parsed.head_size,
            )
            .with_label(label);
        let plot_index = figure.add_quiver_plot_on_axes(plot, axes);
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_quiver_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_quiver_internal(err));
    }
    Ok(handle)
}

fn build_quiver_gpu_plot(
    x: &QuiverCoordinateInput,
    y: &QuiverCoordinateInput,
    u: &runmat_accelerate_api::GpuTensorHandle,
    v: &runmat_accelerate_api::GpuTensorHandle,
    parsed: &ParsedQuiverStyle,
    label: &str,
) -> crate::BuiltinResult<QuiverPlot> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(BUILTIN_NAME)?;
    let u_ref = runmat_accelerate_api::export_wgpu_buffer(u)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "quiver: unable to export GPU U data"))?;
    let v_ref = runmat_accelerate_api::export_wgpu_buffer(v)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "quiver: unable to export GPU V data"))?;
    if u_ref.len != v_ref.len || u_ref.precision != v_ref.precision {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver: U and V GPU inputs must match",
        ));
    }
    let scalar = runmat_plot::gpu::ScalarType::from_is_f64(
        u_ref.precision == runmat_accelerate_api::ProviderPrecision::F64,
    );
    let x_axis = quiver_axis_source(x, u_ref.precision, "X")?;
    let y_axis = quiver_axis_source(y, u_ref.precision, "Y")?;
    let rows = u_ref.shape.first().copied().unwrap_or(u_ref.len).max(1);
    let cols = u_ref.shape.get(1).copied().unwrap_or(1).max(1);
    let count = u_ref.len;
    if count == 0 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver: GPU U/V inputs must be non-empty",
        ));
    }
    let x_len = x_axis.len();
    let y_len = y_axis.len();
    let xy_mode = if x_len == count && y_len == count {
        0u32
    } else if x_len == cols && y_len == rows {
        1u32
    } else {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver: GPU X/Y inputs must match U/V as full coordinates or meshgrid vectors",
        ));
    };
    if xy_mode == 1 && rows.checked_mul(cols) != Some(count) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver: meshgrid GPU metadata does not match U/V element count",
        ));
    }
    let count_u32 = u32::try_from(count).map_err(|_| {
        plotting_error(BUILTIN_NAME, "quiver: vector count exceeds supported range")
    })?;
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| plotting_error(BUILTIN_NAME, "quiver: row count exceeds supported range"))?;
    let cols_u32 = u32::try_from(cols).map_err(|_| {
        plotting_error(BUILTIN_NAME, "quiver: column count exceeds supported range")
    })?;
    let (min_x, max_x) = x_axis.bounds(BUILTIN_NAME).unwrap_or((0.0, 0.0));
    let (min_y, max_y) = y_axis.bounds(BUILTIN_NAME).unwrap_or((0.0, 0.0));
    let (min_u, max_u) = super::gpu_helpers::axis_bounds(u, BUILTIN_NAME).unwrap_or((0.0, 0.0));
    let (min_v, max_v) = super::gpu_helpers::axis_bounds(v, BUILTIN_NAME).unwrap_or((0.0, 0.0));
    let bounds = runmat_plot::core::BoundingBox::new(
        glam::Vec3::new(
            min_x + min_u.min(0.0) * parsed.scale,
            min_y + min_v.min(0.0) * parsed.scale,
            0.0,
        ),
        glam::Vec3::new(
            max_x + max_u.max(0.0) * parsed.scale,
            max_y + max_v.max(0.0) * parsed.scale,
            0.0,
        ),
    );
    let inputs = runmat_plot::gpu::quiver::QuiverGpuInputs {
        x_data: x_axis.axis_data(),
        y_data: y_axis.axis_data(),
        u_buffer: u_ref.buffer.clone(),
        v_buffer: v_ref.buffer.clone(),
        count: count_u32,
        rows: rows_u32,
        cols: cols_u32,
        xy_mode,
        scalar,
    };
    let gpu_source = runmat_plot::plots::QuiverGpuSource {
        x_data: OwnedAxisData::from_axis(&inputs.x_data),
        y_data: OwnedAxisData::from_axis(&inputs.y_data),
        u_buffer: inputs.u_buffer.clone(),
        v_buffer: inputs.v_buffer.clone(),
        count,
        rows,
        cols,
        xy_mode,
        scalar,
    };
    let gpu_vertices = runmat_plot::gpu::quiver::pack_vertices(
        &context.device,
        &context.queue,
        &inputs,
        &runmat_plot::gpu::quiver::QuiverGpuParams {
            color: parsed.color,
            scale: parsed.scale,
            head_size: parsed.head_size,
        },
    )
    .map_err(|e| {
        plotting_error(
            BUILTIN_NAME,
            format!("quiver: failed to build GPU vertices: {e}"),
        )
    })?;
    let mut plot = QuiverPlot::from_gpu_buffer(
        parsed.color,
        parsed.line_width,
        parsed.scale,
        parsed.head_size,
        gpu_vertices,
        count * 6,
        bounds,
    )
    .with_gpu_source(gpu_source)
    .with_label(label);
    plot.x = Vec::new();
    plot.y = Vec::new();
    plot.u = Vec::new();
    plot.v = Vec::new();
    Ok(plot)
}

enum QuiverAxisSource {
    HostF32(Vec<f32>),
    HostF64(Vec<f64>),
    Gpu {
        handle: runmat_accelerate_api::GpuTensorHandle,
        data: OwnedAxisData,
        len: usize,
    },
}

enum QuiverCoordinateInput {
    Explicit(NumericInput),
    ImplicitX { rows: usize, cols: usize },
    ImplicitY { rows: usize, cols: usize },
}

impl QuiverCoordinateInput {
    fn into_tensor(self, builtin: &'static str) -> crate::BuiltinResult<Tensor> {
        match self {
            Self::Explicit(input) => input.into_tensor(builtin),
            Self::ImplicitX { rows, cols } => implicit_quiver_grid_tensor(rows, cols, true),
            Self::ImplicitY { rows, cols } => implicit_quiver_grid_tensor(rows, cols, false),
        }
    }
}

impl QuiverAxisSource {
    fn len(&self) -> usize {
        match self {
            Self::HostF32(values) => values.len(),
            Self::HostF64(values) => values.len(),
            Self::Gpu { len, .. } => *len,
        }
    }

    fn axis_data(&self) -> AxisData<'_> {
        match self {
            Self::HostF32(values) => AxisData::F32(values),
            Self::HostF64(values) => AxisData::F64(values),
            Self::Gpu {
                data: OwnedAxisData::Buffer(buffer),
                ..
            } => AxisData::Buffer(buffer.clone()),
            Self::Gpu { .. } => unreachable!("GPU quiver axes are stored as GPU buffers"),
        }
    }

    fn bounds(&self, name: &'static str) -> crate::BuiltinResult<(f32, f32)> {
        match self {
            Self::HostF32(values) => Ok(host_bounds(values.iter().map(|value| f64::from(*value)))),
            Self::HostF64(values) => Ok(host_bounds(values.iter().copied())),
            Self::Gpu { handle, .. } => super::gpu_helpers::axis_bounds(handle, name),
        }
    }
}

fn quiver_axis_source(
    input: &QuiverCoordinateInput,
    precision: runmat_accelerate_api::ProviderPrecision,
    label: &'static str,
) -> crate::BuiltinResult<QuiverAxisSource> {
    match input {
        QuiverCoordinateInput::Explicit(NumericInput::Host(tensor)) => match precision {
            runmat_accelerate_api::ProviderPrecision::F32 => Ok(QuiverAxisSource::HostF32(
                tensor_utils::tensor_values_f64(tensor)
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
            )),
            runmat_accelerate_api::ProviderPrecision::F64 => Ok(QuiverAxisSource::HostF64(
                tensor_utils::tensor_values_f64(tensor),
            )),
        },
        QuiverCoordinateInput::Explicit(NumericInput::Gpu(handle)) => {
            let exported = runmat_accelerate_api::export_wgpu_buffer(handle).ok_or_else(|| {
                plotting_error(
                    BUILTIN_NAME,
                    format!("quiver: unable to export GPU {label} data"),
                )
            })?;
            if exported.precision != precision {
                return Err(plotting_error(
                    BUILTIN_NAME,
                    "quiver: GPU X, Y, U, and V inputs must have matching precision",
                ));
            }
            Ok(QuiverAxisSource::Gpu {
                handle: handle.clone(),
                data: OwnedAxisData::Buffer(exported.buffer.clone()),
                len: exported.len,
            })
        }
        QuiverCoordinateInput::ImplicitX { cols, .. } => Ok(implicit_quiver_axis(*cols, precision)),
        QuiverCoordinateInput::ImplicitY { rows, .. } => Ok(implicit_quiver_axis(*rows, precision)),
    }
}

fn implicit_quiver_axis(
    len: usize,
    precision: runmat_accelerate_api::ProviderPrecision,
) -> QuiverAxisSource {
    match precision {
        runmat_accelerate_api::ProviderPrecision::F32 => {
            QuiverAxisSource::HostF32((1..=len).map(|value| value as f32).collect())
        }
        runmat_accelerate_api::ProviderPrecision::F64 => {
            QuiverAxisSource::HostF64((1..=len).map(|value| value as f64).collect())
        }
    }
}

fn implicit_quiver_grid_tensor(
    rows: usize,
    cols: usize,
    x_axis: bool,
) -> crate::BuiltinResult<Tensor> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| plotting_error(BUILTIN_NAME, "quiver: implicit grid is too large"))?;
    let mut data = Vec::with_capacity(len);
    for col in 0..cols {
        for row in 0..rows {
            data.push(if x_axis {
                (col + 1) as f64
            } else {
                (row + 1) as f64
            });
        }
    }
    Ok(Tensor {
        data,
        shape: vec![len],
        rows: len,
        cols: 1,
        integer_data: None,
        dtype: runmat_builtins::NumericDType::F64,
    })
}

fn host_bounds(values: impl Iterator<Item = f64>) -> (f32, f32) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        if value.is_finite() {
            min = min.min(value);
            max = max.max(value);
        }
    }
    if min.is_finite() && max.is_finite() {
        (min as f32, max as f32)
    } else {
        (0.0, 0.0)
    }
}

struct ParsedQuiverStyle {
    color: glam::Vec4,
    line_width: f32,
    label: Option<String>,
    scale: f32,
    head_size: f32,
}

fn parse_quiver_style_args(args: &[Value]) -> crate::BuiltinResult<ParsedQuiverStyle> {
    let mut filtered = Vec::new();
    let mut scale = 1.0f32;
    let mut head_size = 0.1f32;
    let mut idx = 0usize;
    while idx < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]) {
            let key = key.trim().to_ascii_lowercase();
            if idx + 1 < args.len() {
                match key.as_str() {
                    "autoscalefactor" | "scale" => {
                        scale = value_as_f64(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "quiver: AutoScaleFactor must be numeric")
                        })? as f32;
                        idx += 2;
                        continue;
                    }
                    "maxheadsize" | "headsize" => {
                        head_size = value_as_f64(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "quiver: MaxHeadSize must be numeric")
                        })? as f32;
                        idx += 2;
                        continue;
                    }
                    _ => {}
                }
            }
        }
        filtered.push(args[idx].clone());
        idx += 1;
    }
    let parsed = parse_line_style_args(&filtered, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    Ok(ParsedQuiverStyle {
        color: parsed.appearance.color,
        line_width: parsed.appearance.line_width,
        label: parsed.label,
        scale,
        head_size,
    })
}

fn parse_quiver_args(args: Vec<Value>) -> crate::BuiltinResult<QuiverArgs> {
    if args.len() < 2 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver: expected U,V or X,Y,U,V inputs",
        ));
    }
    let mut it = args.into_iter();
    let mut target_axes = None;
    let first = it.next().unwrap();
    let first = if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(_, axes)) =
        crate::builtins::plotting::properties::resolve_plot_handle(&first, BUILTIN_NAME)
    {
        target_axes = Some(axes);
        it.next().ok_or_else(|| {
            plotting_error(BUILTIN_NAME, "quiver: expected data after axes handle")
        })?
    } else {
        first
    };
    let second = it.next().unwrap();
    let third = it.next();
    let fourth = it.next();
    match (third, fourth) {
        (None, _) => {
            let (rows, cols) = default_quiver_grid_shape(&first, &second, BUILTIN_NAME)?;
            Ok((
                target_axes,
                QuiverCoordinateInput::ImplicitX { rows, cols },
                QuiverCoordinateInput::ImplicitY { rows, cols },
                first,
                second,
                Vec::new(),
            ))
        }
        (Some(third), Some(fourth)) => Ok((
            target_axes,
            QuiverCoordinateInput::Explicit(
                NumericInput::from_value(first, BUILTIN_NAME)
                    .map_err(map_quiver_invalid_argument)?,
            ),
            QuiverCoordinateInput::Explicit(
                NumericInput::from_value(second, BUILTIN_NAME)
                    .map_err(map_quiver_invalid_argument)?,
            ),
            third,
            fourth,
            it.collect(),
        )),
        _ => Err(plotting_error(
            BUILTIN_NAME,
            "quiver: expected U,V or X,Y,U,V inputs",
        )),
    }
}

fn default_quiver_grid_shape(
    u: &Value,
    v: &Value,
    builtin: &'static str,
) -> crate::BuiltinResult<(usize, usize)> {
    let (u_rows, u_cols, u_len) = tensor_shape_from_value(u, builtin)?;
    let (v_rows, v_cols, v_len) = tensor_shape_from_value(v, builtin)?;
    if u_rows != v_rows || u_cols != v_cols || u_len != v_len {
        return Err(plotting_error(
            builtin,
            "quiver: U and V inputs must have identical size",
        ));
    }
    let rows = u_rows.max(1);
    let cols = u_cols.max(1);
    Ok((rows, cols))
}

fn tensor_shape_from_value(
    value: &Value,
    builtin: &'static str,
) -> crate::BuiltinResult<(usize, usize, usize)> {
    match value {
        Value::GpuTensor(handle) => {
            let rows = handle.shape.first().copied().unwrap_or(1).max(1);
            let cols = handle.shape.get(1).copied().unwrap_or(1).max(1);
            let len = handle.shape.iter().product::<usize>().max(1);
            Ok((rows, cols, len))
        }
        _ => {
            let tensor = Tensor::try_from(value)
                .map_err(|e| plotting_error(builtin, format!("quiver: {e}")))?;
            Ok((
                tensor.rows.max(1),
                tensor.cols.max(1),
                tensor_utils::tensor_element_len(&tensor),
            ))
        }
    }
}

fn materialize_quiver_components(
    x: Tensor,
    y: Tensor,
    u: Tensor,
    v: Tensor,
    builtin: &'static str,
) -> crate::BuiltinResult<QuiverComponents> {
    let u_rows = u.rows;
    let u_cols = u.cols;
    let v_rows = v.rows;
    let v_cols = v.cols;
    let x_len = tensor_utils::tensor_element_len(&x);
    let y_len = tensor_utils::tensor_element_len(&y);
    let u_len = tensor_utils::tensor_element_len(&u);
    let v_len = tensor_utils::tensor_element_len(&v);

    if u_rows != v_rows || u_cols != v_cols || u_len != v_len {
        return Err(plotting_error(
            builtin,
            "quiver: U and V inputs must have identical size",
        ));
    }

    let u_is_matrix = u_rows > 1 && u_cols > 1;
    let v_is_matrix = v_rows > 1 && v_cols > 1;
    if u_is_matrix != v_is_matrix {
        return Err(plotting_error(
            builtin,
            "quiver: U and V inputs must both be vectors or both be matrices",
        ));
    }

    if !u_is_matrix {
        if x_len != u_len || y_len != u_len {
            return Err(plotting_error(
                builtin,
                "quiver: X, Y, U, and V vectors must have the same length",
            ));
        }
        return Ok((
            tensor_utils::tensor_into_values_f64(x),
            tensor_utils::tensor_into_values_f64(y),
            tensor_utils::tensor_into_values_f64(u),
            tensor_utils::tensor_into_values_f64(v),
        ));
    }

    let rows = u_rows;
    let cols = u_cols;
    if x_len == rows * cols && y_len == rows * cols {
        return Ok((
            tensor_utils::tensor_into_values_f64(x),
            tensor_utils::tensor_into_values_f64(y),
            tensor_utils::tensor_into_values_f64(u),
            tensor_utils::tensor_into_values_f64(v),
        ));
    }
    if x_len == cols && y_len == rows {
        let x_values = tensor_utils::tensor_into_values_f64(x);
        let y_values = tensor_utils::tensor_into_values_f64(y);
        let mut out_x = Vec::with_capacity(rows * cols);
        let mut out_y = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                out_x.push(x_values[col]);
                out_y.push(y_values[row]);
            }
        }
        return Ok((
            out_x,
            out_y,
            tensor_utils::tensor_into_values_f64(u),
            tensor_utils::tensor_into_values_f64(v),
        ));
    }
    Err(plotting_error(
        builtin,
        "quiver: X and Y must match U/V as vectors or meshgrid-style matrices",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
        subplot::subplot_builtin,
    };
    use runmat_plot::plots::PlotElement;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            integer_data: None,
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    fn int_tensor(data: &[i16], shape: Vec<usize>) -> Tensor {
        let mut tensor =
            Tensor::new_integer(runmat_builtins::IntegerStorage::I16(data.to_vec()), shape)
                .expect("integer tensor");
        tensor.data.clear();
        tensor
    }

    #[test]
    fn quiver_axis_source_keeps_host_axes_on_gpu_pack_path() {
        let host =
            QuiverCoordinateInput::Explicit(NumericInput::Host(vec_tensor(&[1.0, 2.5, f64::NAN])));
        let f32_axis =
            quiver_axis_source(&host, runmat_accelerate_api::ProviderPrecision::F32, "X").unwrap();
        assert_eq!(f32_axis.len(), 3);
        match f32_axis.axis_data() {
            AxisData::F32(values) => {
                assert_eq!(&values[..2], &[1.0, 2.5]);
                assert!(values[2].is_nan());
            }
            _ => panic!("expected f32 axis data for f32 shader"),
        }
        assert_eq!(f32_axis.bounds(BUILTIN_NAME).unwrap(), (1.0, 2.5));

        let f64_axis =
            quiver_axis_source(&host, runmat_accelerate_api::ProviderPrecision::F64, "Y").unwrap();
        assert_eq!(f64_axis.len(), 3);
        match f64_axis.axis_data() {
            AxisData::F64(values) => {
                assert_eq!(&values[..2], &[1.0, 2.5]);
                assert!(values[2].is_nan());
            }
            _ => panic!("expected f64 axis data for f64 shader"),
        }
        assert_eq!(f64_axis.bounds(BUILTIN_NAME).unwrap(), (1.0, 2.5));
    }

    #[test]
    fn quiver_axis_source_reads_typed_integer_storage_exactly() {
        let mut tensor = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![-2, 0, 2]),
            vec![3],
        )
        .expect("typed quiver axis");
        tensor.data.clear();
        let host = QuiverCoordinateInput::Explicit(NumericInput::Host(tensor));

        let f64_axis =
            quiver_axis_source(&host, runmat_accelerate_api::ProviderPrecision::F64, "X").unwrap();
        match f64_axis.axis_data() {
            AxisData::F64(values) => assert_eq!(values, &[-2.0, 0.0, 2.0]),
            _ => panic!("expected f64 axis data"),
        }

        let f32_axis =
            quiver_axis_source(&host, runmat_accelerate_api::ProviderPrecision::F32, "X").unwrap();
        match f32_axis.axis_data() {
            AxisData::F32(values) => assert_eq!(values, &[-2.0, 0.0, 2.0]),
            _ => panic!("expected f32 axis data"),
        }
    }

    #[test]
    fn quiver_components_read_typed_integer_vectors_exactly() {
        let components = materialize_quiver_components(
            int_tensor(&[0, 1], vec![2]),
            int_tensor(&[2, 3], vec![2]),
            int_tensor(&[4, 5], vec![2]),
            int_tensor(&[6, 7], vec![2]),
            BUILTIN_NAME,
        )
        .expect("quiver components");

        assert_eq!(
            components,
            (
                vec![0.0, 1.0],
                vec![2.0, 3.0],
                vec![4.0, 5.0],
                vec![6.0, 7.0]
            )
        );
    }

    #[test]
    fn quiver_components_expand_typed_integer_meshgrid_axes() {
        let components = materialize_quiver_components(
            int_tensor(&[10, 20], vec![1, 2]),
            int_tensor(&[1, 2], vec![2, 1]),
            int_tensor(&[3, 4, 5, 6], vec![2, 2]),
            int_tensor(&[7, 8, 9, 10], vec![2, 2]),
            BUILTIN_NAME,
        )
        .expect("meshgrid-style components");

        assert_eq!(
            components,
            (
                vec![10.0, 10.0, 20.0, 20.0],
                vec![1.0, 2.0, 1.0, 2.0],
                vec![3.0, 4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0, 10.0]
            )
        );
    }

    #[test]
    fn quiver_axis_source_keeps_implicit_grid_compact() {
        let x_axis = QuiverCoordinateInput::ImplicitX { rows: 3, cols: 2 };
        let y_axis = QuiverCoordinateInput::ImplicitY { rows: 3, cols: 2 };
        let x_source =
            quiver_axis_source(&x_axis, runmat_accelerate_api::ProviderPrecision::F64, "X")
                .unwrap();
        let y_source =
            quiver_axis_source(&y_axis, runmat_accelerate_api::ProviderPrecision::F64, "Y")
                .unwrap();
        assert_eq!(x_source.len(), 2);
        assert_eq!(y_source.len(), 3);
        match x_source.axis_data() {
            AxisData::F64(values) => assert_eq!(values, &[1.0, 2.0]),
            _ => panic!("expected compact f64 x axis"),
        }
        match y_source.axis_data() {
            AxisData::F64(values) => assert_eq!(values, &[1.0, 2.0, 3.0]),
            _ => panic!("expected compact f64 y axis"),
        }

        let x_full = x_axis.into_tensor(BUILTIN_NAME).unwrap();
        let y_full = y_axis.into_tensor(BUILTIN_NAME).unwrap();
        assert_eq!(x_full.data, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert_eq!(y_full.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn quiver_builds_plot_and_defaults_grid() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = futures::executor::block_on(quiver_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, -1.0])),
            Value::Tensor(vec_tensor(&[0.5, 0.5])),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver");
        };
        assert_eq!(quiver.x, vec![1.0, 1.0]);
        assert_eq!(quiver.y, vec![1.0, 2.0]);
    }

    #[test]
    fn quiver_supports_axes_target_and_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let handle = futures::executor::block_on(quiver_builtin(vec![
            Value::Num(ax),
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[1.0, 0.0])),
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::String("AutoScaleFactor".into()),
            Value::Num(2.5),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("AutoScaleFactor".into())
            ])
            .unwrap(),
            Value::Num(2.5)
        );
        set_builtin(vec![
            Value::Num(handle),
            Value::String("MaxHeadSize".into()),
            Value::Num(0.3),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver");
        };
        assert_eq!(quiver.head_size, 0.3);
    }

    #[test]
    fn quiver_descriptor_signatures_cover_supported_forms() {
        let labels: Vec<&str> = QUIVER_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = quiver(U, V)"));
        assert!(labels.contains(&"h = quiver(X, Y, U, V)"));
        assert!(labels.contains(&"h = quiver(X, Y, U, V, Name, Value, ...)"));
        assert!(labels.contains(&"h = quiver(ax, U, V)"));
        assert!(labels.contains(&"h = quiver(ax, X, Y, U, V, Name, Value, ...)"));
    }

    #[test]
    fn quiver_missing_post_axes_input_uses_stable_identifier() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let err = futures::executor::block_on(quiver_builtin(vec![Value::Num(ax)]))
            .expect_err("missing post-axes data should fail");
        assert_eq!(err.identifier(), QUIVER_ERROR_INVALID_ARGUMENT.identifier);
    }
}
