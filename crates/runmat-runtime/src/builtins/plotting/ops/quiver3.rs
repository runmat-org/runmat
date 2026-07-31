//! MATLAB-compatible `quiver3` builtin.

use glam::Vec4;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::QuiverPlot;
use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_helpers;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_line_style_args, value_as_f64, LineStyleParseOptions};

const BUILTIN_NAME: &str = "quiver3";

type Quiver3Args = (
    Option<usize>,
    Value,
    Value,
    Value,
    Value,
    Value,
    Value,
    Vec<Value>,
);
type Quiver3Components = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered 3-D quiver plot.",
}];

const INPUTS_Z_U_V_W: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
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
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
];

const INPUTS_X_Y_Z_U_V_W: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base x-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base y-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
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
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
];

const INPUTS_AX_Z_U_V_W: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
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
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
];

const INPUTS_AX_X_Y_Z_U_V_W: [BuiltinParamDescriptor; 7] = [
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
        description: "Arrow base x-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base y-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
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
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
];

const INPUTS_Z_U_V_W_ARGS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
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
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional scale, line-style shorthand, or name/value style arguments.",
    },
];

const INPUTS_X_Y_Z_U_V_W_ARGS: [BuiltinParamDescriptor; 7] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base x-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base y-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
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
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional scale, line-style shorthand, or name/value style arguments.",
    },
];

const INPUTS_AX_Z_U_V_W_ARGS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
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
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional scale, line-style shorthand, or name/value style arguments.",
    },
];

const INPUTS_AX_X_Y_Z_U_V_W_ARGS: [BuiltinParamDescriptor; 8] = [
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
        description: "Arrow base x-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base y-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
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
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional scale, line-style shorthand, or name/value style arguments.",
    },
];

const QUIVER3_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = quiver3(Z, U, V, W)",
        inputs: &INPUTS_Z_U_V_W,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(X, Y, Z, U, V, W)",
        inputs: &INPUTS_X_Y_Z_U_V_W,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(Z, U, V, W, scaleOrStyleOrNameValue...)",
        inputs: &INPUTS_Z_U_V_W_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(X, Y, Z, U, V, W, scaleOrStyleOrNameValue...)",
        inputs: &INPUTS_X_Y_Z_U_V_W_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(ax, Z, U, V, W)",
        inputs: &INPUTS_AX_Z_U_V_W,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(ax, X, Y, Z, U, V, W)",
        inputs: &INPUTS_AX_X_Y_Z_U_V_W,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(ax, Z, U, V, W, scaleOrStyleOrNameValue...)",
        inputs: &INPUTS_AX_Z_U_V_W_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(ax, X, Y, Z, U, V, W, scaleOrStyleOrNameValue...)",
        inputs: &INPUTS_AX_X_Y_Z_U_V_W_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
];

pub const QUIVER3_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUIVER3.INVALID_ARGUMENT",
    identifier: Some("RunMat:quiver3:InvalidArgument"),
    when: "Input data, axes targeting, scale, or style arguments are invalid.",
    message: "quiver3: invalid argument",
};

pub const QUIVER3_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUIVER3.INTERNAL",
    identifier: Some("RunMat:quiver3:Internal"),
    when: "Internal quiver3 construction or rendering fails unexpectedly.",
    message: "quiver3: internal operation failed",
};

const QUIVER3_ERRORS: [BuiltinErrorDescriptor; 2] =
    [QUIVER3_ERROR_INVALID_ARGUMENT, QUIVER3_ERROR_INTERNAL];

pub const QUIVER3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &QUIVER3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &QUIVER3_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::quiver3")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "quiver3",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "quiver3 is a 3-D plotting sink. GPU inputs are gathered to build renderer geometry; a direct WGPU packer is tracked by the GPU fast-path audit.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::quiver3")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "quiver3",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "quiver3 performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "quiver3",
    category = "plotting",
    summary = "Render 3-D quiver vector-field plots.",
    keywords = "quiver3,plotting,3d,vector field,arrows",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::quiver3::QUIVER3_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::quiver3"
)]
pub async fn quiver3_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target_axes, x, y, z, u, v, w, rest) =
        parse_quiver3_args(args).map_err(map_quiver3_invalid_argument)?;
    let parsed = parse_quiver3_style_args(&rest).map_err(map_quiver3_invalid_argument)?;
    let x_tensor = NumericInput::from_value(x, BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?
        .into_tensor(BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?;
    let y_tensor = NumericInput::from_value(y, BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?
        .into_tensor(BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?;
    let z_tensor = NumericInput::from_value(z, BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?
        .into_tensor(BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?;
    let u_tensor = NumericInput::from_value(u, BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?
        .into_tensor(BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?;
    let v_tensor = NumericInput::from_value(v, BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?
        .into_tensor(BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?;
    let w_tensor = NumericInput::from_value(w, BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?
        .into_tensor(BUILTIN_NAME)
        .map_err(map_quiver3_invalid_argument)?;
    let (x_vals, y_vals, z_vals, u_vals, v_vals, w_vals) =
        materialize_quiver3_components(x_tensor, y_tensor, z_tensor, u_tensor, v_tensor, w_tensor)
            .map_err(map_quiver3_invalid_argument)?;
    let opts = PlotRenderOptions {
        title: "Quiver3",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let axes = target_axes.unwrap_or(axes);
        let label = parsed.label.clone().unwrap_or_else(|| "Data".into());
        let plot = QuiverPlot::new3d(
            x_vals.clone(),
            y_vals.clone(),
            z_vals.clone(),
            u_vals.clone(),
            v_vals.clone(),
            w_vals.clone(),
        )
        .map_err(|err| plotting_error(BUILTIN_NAME, err))?
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
        crate::builtins::plotting::state::register_quiver3_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_quiver3_internal(err));
    }
    Ok(handle)
}

#[derive(Clone)]
struct ParsedQuiver3Style {
    color: Vec4,
    line_width: f32,
    label: Option<String>,
    scale: f32,
    head_size: f32,
}

fn parse_quiver3_style_args(args: &[Value]) -> crate::BuiltinResult<ParsedQuiver3Style> {
    let mut filtered = Vec::new();
    let mut scale = 1.0f32;
    let mut head_size = 0.1f32;
    let mut idx = 0usize;
    if let Some(first) = args.first().and_then(value_as_f64) {
        scale = first as f32;
        idx = 1;
    }
    while idx < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]) {
            let key = key.trim().to_ascii_lowercase();
            if idx + 1 < args.len() {
                match key.as_str() {
                    "autoscalefactor" | "scale" => {
                        scale = value_as_f64(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "quiver3: AutoScaleFactor must be numeric")
                        })? as f32;
                        idx += 2;
                        continue;
                    }
                    "maxheadsize" | "headsize" => {
                        head_size = value_as_f64(&args[idx + 1]).ok_or_else(|| {
                            plotting_error(BUILTIN_NAME, "quiver3: MaxHeadSize must be numeric")
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
    Ok(ParsedQuiver3Style {
        color: parsed.appearance.color,
        line_width: parsed.appearance.line_width,
        label: parsed.label,
        scale,
        head_size,
    })
}

fn parse_quiver3_args(args: Vec<Value>) -> crate::BuiltinResult<Quiver3Args> {
    let mut it = args.into_iter();
    let Some(first) = it.next() else {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver3: expected Z,U,V,W or X,Y,Z,U,V,W inputs",
        ));
    };
    let mut target_axes = None;
    let first = if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(_, axes)) =
        crate::builtins::plotting::properties::resolve_plot_handle(&first, BUILTIN_NAME)
    {
        target_axes = Some(axes);
        it.next().ok_or_else(|| {
            plotting_error(BUILTIN_NAME, "quiver3: expected data after axes handle")
        })?
    } else {
        first
    };
    let mut values = Vec::with_capacity(1 + it.size_hint().0);
    values.push(first);
    values.extend(it);
    if values.len() < 4 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver3: expected Z,U,V,W or X,Y,Z,U,V,W inputs",
        ));
    }
    let explicit = values.len() >= 6 && values.iter().take(6).all(is_numeric_like);
    if explicit {
        let rest = values.split_off(6);
        let w = values.pop().unwrap();
        let v = values.pop().unwrap();
        let u = values.pop().unwrap();
        let z = values.pop().unwrap();
        let y = values.pop().unwrap();
        let x = values.pop().unwrap();
        Ok((target_axes, x, y, z, u, v, w, rest))
    } else {
        let rest = values.split_off(4);
        let w = values.pop().unwrap();
        let v = values.pop().unwrap();
        let u = values.pop().unwrap();
        let z = values.pop().unwrap();
        let (x, y) = default_quiver3_grid_from_value(&z)?;
        Ok((
            target_axes,
            Value::Tensor(x),
            Value::Tensor(y),
            z,
            u,
            v,
            w,
            rest,
        ))
    }
}

fn is_numeric_like(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Bool(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

fn default_quiver3_grid_from_value(value: &Value) -> crate::BuiltinResult<(Tensor, Tensor)> {
    let (rows, cols, len) = tensor_shape_from_value(value)?;
    let mut x = Vec::with_capacity(len);
    let mut y = Vec::with_capacity(len);
    for idx in 0..len {
        let row = idx % rows;
        let col = (idx / rows) % cols;
        x.push((col + 1) as f64);
        y.push((row + 1) as f64);
    }
    Ok((
        Tensor::new(x, vec![len]).expect("implicit quiver3 x grid"),
        Tensor::new(y, vec![len]).expect("implicit quiver3 y grid"),
    ))
}

fn tensor_shape_from_value(value: &Value) -> crate::BuiltinResult<(usize, usize, usize)> {
    match value {
        Value::GpuTensor(handle) => {
            let rows = handle.shape.first().copied().unwrap_or(1).max(1);
            let cols = handle.shape.get(1).copied().unwrap_or(1).max(1);
            let len = handle.shape.iter().product::<usize>().max(1);
            Ok((rows, cols, len))
        }
        _ => {
            let tensor = Tensor::try_from(value)
                .map_err(|err| plotting_error(BUILTIN_NAME, format!("quiver3: {err}")))?;
            Ok((
                tensor.rows.max(1),
                tensor.cols.max(1),
                tensor_helpers::tensor_element_len(&tensor),
            ))
        }
    }
}

fn materialize_quiver3_components(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    u: Tensor,
    v: Tensor,
    w: Tensor,
) -> crate::BuiltinResult<Quiver3Components> {
    let x_len = tensor_helpers::tensor_element_len(&x);
    let y_len = tensor_helpers::tensor_element_len(&y);
    let z_len = tensor_helpers::tensor_element_len(&z);
    let u_len = tensor_helpers::tensor_element_len(&u);
    let v_len = tensor_helpers::tensor_element_len(&v);
    let w_len = tensor_helpers::tensor_element_len(&w);
    let z_rows = z.rows;
    let z_cols = z.cols;
    let z_shape = z.shape.clone();
    let u_rows = u.rows;
    let u_cols = u.cols;
    let u_shape = u.shape.clone();
    let v_rows = v.rows;
    let v_cols = v.cols;
    let v_shape = v.shape.clone();
    let w_rows = w.rows;
    let w_cols = w.cols;
    let w_shape = w.shape.clone();
    let x_values = tensor_helpers::tensor_into_values_f64(x);
    let y_values = tensor_helpers::tensor_into_values_f64(y);
    let z_values = tensor_helpers::tensor_into_values_f64(z);
    let u_values = tensor_helpers::tensor_into_values_f64(u);
    let v_values = tensor_helpers::tensor_into_values_f64(v);
    let w_values = tensor_helpers::tensor_into_values_f64(w);

    if u_rows != v_rows
        || u_cols != v_cols
        || u_len != v_len
        || u_shape != v_shape
        || u_rows != w_rows
        || u_cols != w_cols
        || u_len != w_len
        || u_shape != w_shape
    {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver3: U, V, and W inputs must have identical size",
        ));
    }
    if z_rows != u_rows || z_cols != u_cols || z_len != u_len || z_shape != u_shape {
        return Err(plotting_error(
            BUILTIN_NAME,
            "quiver3: Z, U, V, and W inputs must have identical size",
        ));
    }
    if x_len == u_len && y_len == u_len {
        return Ok((x_values, y_values, z_values, u_values, v_values, w_values));
    }
    let u_is_matrix = u_rows > 1 && u_cols > 1;
    if !u_is_matrix {
        if x_len != u_len || y_len != u_len {
            return Err(plotting_error(
                BUILTIN_NAME,
                "quiver3: X, Y, Z, U, V, and W vectors must have the same length",
            ));
        }
        return Ok((x_values, y_values, z_values, u_values, v_values, w_values));
    }
    let rows = u_rows;
    let cols = u_cols;
    if x_len == rows * cols && y_len == rows * cols {
        return Ok((x_values, y_values, z_values, u_values, v_values, w_values));
    }
    if x_len == cols && y_len == rows {
        let mut out_x = Vec::with_capacity(rows * cols);
        let mut out_y = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                out_x.push(x_values[col]);
                out_y.push(y_values[row]);
            }
        }
        return Ok((out_x, out_y, z_values, u_values, v_values, w_values));
    }
    Err(plotting_error(
        BUILTIN_NAME,
        "quiver3: X and Y must match U/V/W as vectors or meshgrid-style coordinates",
    ))
}

fn quiver3_error_with_detail(descriptor: &BuiltinErrorDescriptor, detail: String) -> RuntimeError {
    let message = if detail.trim().is_empty() {
        descriptor.message.to_string()
    } else {
        detail
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_quiver3_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    quiver3_error_with_detail(&QUIVER3_DESCRIPTOR.errors[0], err.message)
}

fn map_quiver3_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    quiver3_error_with_detail(&QUIVER3_DESCRIPTOR.errors[1], err.message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use runmat_builtins::IntegerStorage;
    use runmat_plot::plots::PlotElement;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("quiver3 test vector")
    }

    fn int_vec_tensor(data: Vec<i16>) -> Tensor {
        let mut tensor = Tensor::new_integer(IntegerStorage::I16(data.clone()), vec![data.len()])
            .expect("integer tensor");
        tensor.data.clear();
        tensor
    }

    fn mat_tensor(rows: usize, cols: usize, data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![rows, cols]).expect("quiver3 test matrix")
    }

    fn nd_tensor(shape: Vec<usize>, data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), shape).expect("quiver3 test tensor")
    }

    #[test]
    fn quiver3_builds_default_grid_and_3d_vectors() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(mat_tensor(2, 2, &[10.0, 20.0, 30.0, 40.0])),
            Value::Tensor(mat_tensor(2, 2, &[1.0, 0.0, -1.0, 0.5])),
            Value::Tensor(mat_tensor(2, 2, &[0.0, 1.0, 0.5, -0.5])),
            Value::Tensor(mat_tensor(2, 2, &[2.0, 2.0, 1.0, 1.0])),
        ]))
        .unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver plot");
        };
        assert_eq!(quiver.x, vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(quiver.y, vec![1.0, 2.0, 1.0, 2.0]);
        assert_eq!(quiver.z.as_ref().unwrap(), &vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(quiver.w.as_ref().unwrap(), &vec![2.0, 2.0, 1.0, 1.0]);
    }

    #[test]
    fn quiver3_default_grid_covers_all_nd_elements() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let data: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        let handle = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(nd_tensor(vec![2, 2, 2], &data)),
            Value::Tensor(nd_tensor(vec![2, 2, 2], &[1.0; 8])),
            Value::Tensor(nd_tensor(vec![2, 2, 2], &[0.0; 8])),
            Value::Tensor(nd_tensor(vec![2, 2, 2], &[0.5; 8])),
        ]))
        .unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver plot");
        };
        assert_eq!(quiver.x.len(), 8);
        assert_eq!(quiver.y.len(), 8);
        assert_eq!(quiver.z.as_ref().unwrap().len(), 8);
        assert_eq!(quiver.x, vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
        assert_eq!(quiver.y, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn quiver3_rejects_equal_length_but_different_shapes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let err = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0, 3.0, 4.0])),
            Value::Tensor(mat_tensor(2, 2, &[1.0, 0.0, -1.0, 0.5])),
            Value::Tensor(mat_tensor(2, 2, &[0.0, 1.0, 0.5, -0.5])),
            Value::Tensor(mat_tensor(2, 2, &[2.0, 2.0, 1.0, 1.0])),
        ]))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:quiver3:InvalidArgument"));
        assert!(err.message.contains("identical size"));
    }

    #[test]
    fn quiver3_supports_explicit_coordinates_scale_style_and_get() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(vec_tensor(&[0.0, 1.0])),
            Value::Tensor(vec_tensor(&[2.0, 3.0])),
            Value::Tensor(vec_tensor(&[4.0, 5.0])),
            Value::Tensor(vec_tensor(&[0.25, 0.5])),
            Value::Tensor(vec_tensor(&[0.75, 1.0])),
            Value::Tensor(vec_tensor(&[1.25, 1.5])),
            Value::Num(2.0),
            Value::String("r".into()),
            Value::String("DisplayName".into()),
            Value::String("field".into()),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver plot");
        };
        assert_eq!(quiver.scale, 2.0);
        assert_eq!(quiver.label.as_deref(), Some("field"));
        let z_data = get_builtin(vec![Value::Num(handle), Value::String("ZData".into())]).unwrap();
        let Value::Tensor(z_data) = z_data else {
            panic!("expected tensor zdata");
        };
        assert_eq!(z_data.data, vec![4.0, 5.0]);
    }

    #[test]
    fn quiver3_explicit_vectors_read_typed_integer_storage_exactly() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(quiver3_builtin(vec![
            Value::Tensor(int_vec_tensor(vec![0, 1])),
            Value::Tensor(int_vec_tensor(vec![2, 3])),
            Value::Tensor(int_vec_tensor(vec![4, 5])),
            Value::Tensor(int_vec_tensor(vec![6, 7])),
            Value::Tensor(int_vec_tensor(vec![8, 9])),
            Value::Tensor(int_vec_tensor(vec![10, 11])),
        ]))
        .unwrap();
        assert!(handle.is_finite());
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Quiver(quiver) = fig.plots().next().unwrap() else {
            panic!("expected quiver plot");
        };
        assert_eq!(quiver.x, vec![0.0, 1.0]);
        assert_eq!(quiver.y, vec![2.0, 3.0]);
        assert_eq!(quiver.z.as_ref().unwrap(), &vec![4.0, 5.0]);
        assert_eq!(quiver.u, vec![6.0, 7.0]);
        assert_eq!(quiver.v, vec![8.0, 9.0]);
        assert_eq!(quiver.w.as_ref().unwrap(), &vec![10.0, 11.0]);
    }

    #[test]
    fn quiver3_rejects_mismatched_typed_integer_vector_lengths() {
        let err = materialize_quiver3_components(
            int_vec_tensor(vec![0, 1]),
            int_vec_tensor(vec![2]),
            int_vec_tensor(vec![4, 5]),
            int_vec_tensor(vec![6, 7]),
            int_vec_tensor(vec![8, 9]),
            int_vec_tensor(vec![10, 11]),
        )
        .expect_err("mismatched typed vectors should reject");

        assert!(format!("{err:?}").contains("same length"));
    }

    #[test]
    fn quiver3_descriptor_signatures_cover_supported_forms() {
        let labels: Vec<&str> = QUIVER3_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = quiver3(Z, U, V, W)"));
        assert!(labels.contains(&"h = quiver3(X, Y, Z, U, V, W)"));
        assert!(labels.contains(&"h = quiver3(Z, U, V, W, scaleOrStyleOrNameValue...)"));
        assert!(labels.contains(&"h = quiver3(X, Y, Z, U, V, W, scaleOrStyleOrNameValue...)"));
        assert!(labels.contains(&"h = quiver3(ax, Z, U, V, W)"));
        assert!(labels.contains(&"h = quiver3(ax, X, Y, Z, U, V, W)"));
        assert!(labels.contains(&"h = quiver3(ax, Z, U, V, W, scaleOrStyleOrNameValue...)"));
        assert!(labels.contains(&"h = quiver3(ax, X, Y, Z, U, V, W, scaleOrStyleOrNameValue...)"));
    }
}
