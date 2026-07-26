//! MATLAB-compatible `fsurf` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::math::optim::common::call_function;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::op_common::{apply_axes_target, split_leading_axes_handle, AxesTarget};
use super::plotting_error;
use super::state::PlotRenderOptions;
use super::state::{
    current_figure_handle, register_function_surface_handle, render_active_plot,
    FunctionSurfaceFunctionRef, FunctionSurfaceFunctionState,
};
use super::style::{parse_surface_style_args, value_as_string, SurfaceStyleDefaults};

const BUILTIN_NAME: &str = "fsurf";
const DEFAULT_DOMAIN: Domain = Domain {
    x_min: -5.0,
    x_max: 5.0,
    y_min: -5.0,
    y_max: 5.0,
};
const DEFAULT_MESH_DENSITY: usize = 35;
const MAX_MESH_DENSITY: usize = 400;

const FSURF_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered function surface.",
}];

const FSURF_INPUTS_F: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function of two variables, z = f(x,y).",
}];

const FSURF_INPUTS_F_DOMAIN: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function of two variables, z = f(x,y).",
    },
    BuiltinParamDescriptor {
        name: "xyinterval",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[-5 5 -5 5]"),
        description: "Two- or four-element domain vector.",
    },
];

const FSURF_INPUTS_PARAMETRIC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fx",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parametric x-coordinate function.",
    },
    BuiltinParamDescriptor {
        name: "fy",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parametric y-coordinate function.",
    },
    BuiltinParamDescriptor {
        name: "fz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parametric z-coordinate function.",
    },
];

const FSURF_INPUTS_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function-handle and optional domain arguments.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "MeshDensity and surface style name/value arguments.",
    },
];

const FSURF_INPUTS_AX_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function-handle and optional domain arguments.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "MeshDensity and surface style name/value arguments.",
    },
];

const FSURF_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = fsurf(f)",
        inputs: &FSURF_INPUTS_F,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(f, xyinterval)",
        inputs: &FSURF_INPUTS_F_DOMAIN,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(fx, fy, fz)",
        inputs: &FSURF_INPUTS_PARAMETRIC,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(fx, fy, fz, uvinterval)",
        inputs: &FSURF_INPUTS_PROPS,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(___, Name, Value, ...)",
        inputs: &FSURF_INPUTS_PROPS,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(ax, ___)",
        inputs: &FSURF_INPUTS_AX_PROPS,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
];

pub const FSURF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSURF.INVALID_ARGUMENT",
    identifier: Some("RunMat:fsurf:InvalidArgument"),
    when:
        "Function handles, domains, mesh density, axes target, or surface properties are invalid.",
    message: "fsurf: invalid argument",
};

pub const FSURF_ERROR_EVALUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSURF.EVALUATION",
    identifier: Some("RunMat:fsurf:EvaluationFailed"),
    when: "A sampled function handle fails or does not return a scalar numeric value.",
    message: "fsurf: function evaluation failed",
};

pub const FSURF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSURF.INTERNAL",
    identifier: Some("RunMat:fsurf:Internal"),
    when: "Surface construction or rendering fails unexpectedly.",
    message: "fsurf: internal operation failed",
};

const FSURF_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FSURF_ERROR_INVALID_ARGUMENT,
    FSURF_ERROR_EVALUATION,
    FSURF_ERROR_INTERNAL,
];

pub const FSURF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FSURF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FSURF_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::fsurf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fsurf",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "fsurf samples arbitrary MATLAB function handles on the host, then renders through the existing surface plot pipeline.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::fsurf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fsurf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fsurf performs callback sampling and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "fsurf",
    category = "plotting",
    summary = "Plot a function surface over a 2-D domain.",
    keywords = "fsurf,function surface,parametric surface,plotting,3d",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::fsurf::FSURF_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::fsurf"
)]
pub async fn fsurf_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    let parsed = parse_fsurf_args(args).map_err(map_fsurf_invalid)?;
    let surface = sample_surface(&parsed).await?;
    render_fsurf(surface, &parsed).map_err(map_fsurf_internal)
}

#[derive(Clone)]
enum SurfaceFunction {
    Explicit(Value),
    Parametric { x: Value, y: Value, z: Value },
}

#[derive(Clone, Copy)]
struct Domain {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
}

struct ParsedFsurf {
    target_axes: AxesTarget,
    function: SurfaceFunction,
    domain: Domain,
    mesh_density: usize,
    style_args: Vec<Value>,
}

fn fsurf_error_with_detail(
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

fn fsurf_invalid(detail: impl AsRef<str>) -> RuntimeError {
    fsurf_error_with_detail(&FSURF_ERROR_INVALID_ARGUMENT, detail)
}

fn map_fsurf_invalid(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fsurf_error_with_detail(&FSURF_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_fsurf_eval(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fsurf_error_with_detail(&FSURF_ERROR_EVALUATION, err.message)
}

fn map_fsurf_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fsurf_error_with_detail(&FSURF_ERROR_INTERNAL, err.message)
}

fn parse_fsurf_args(args: Vec<Value>) -> BuiltinResult<ParsedFsurf> {
    if args.is_empty() {
        return Err(fsurf_invalid("expected a function handle"));
    }

    let (target_axes, mut values) = split_leading_axes_handle(args, BUILTIN_NAME)?;
    if values.is_empty() {
        return Err(fsurf_invalid(
            "expected a function handle after axes handle",
        ));
    }

    let mut function_values = Vec::new();
    while function_values.len() < 3 && values.first().map(is_function_handle).unwrap_or(false) {
        function_values.push(values.remove(0));
    }

    let function = match function_values.len() {
        1 => SurfaceFunction::Explicit(function_values.remove(0)),
        3 => SurfaceFunction::Parametric {
            x: function_values.remove(0),
            y: function_values.remove(0),
            z: function_values.remove(0),
        },
        0 => return Err(fsurf_invalid("expected a function handle")),
        _ => {
            return Err(fsurf_invalid(
                "expected either one function handle or three parametric function handles",
            ))
        }
    };

    let mut domain = DEFAULT_DOMAIN;
    if values.first().is_some_and(is_numeric_domain_value) {
        domain = parse_domain(&values.remove(0))?;
    }
    let (mesh_density, style_args) = split_mesh_density(values)?;

    Ok(ParsedFsurf {
        target_axes,
        function,
        domain,
        mesh_density,
        style_args,
    })
}

fn is_function_handle(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}

fn is_numeric_domain_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

fn parse_domain(value: &Value) -> BuiltinResult<Domain> {
    let values = numeric_vector(value)?;
    match values.as_slice() {
        [lo, hi] => domain_from_values(*lo, *hi, *lo, *hi),
        [x_min, x_max, y_min, y_max] => domain_from_values(*x_min, *x_max, *y_min, *y_max),
        _ => Err(fsurf_invalid(
            "domain must be a two-element or four-element numeric vector",
        )),
    }
}

fn domain_from_values(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> BuiltinResult<Domain> {
    if ![x_min, x_max, y_min, y_max]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(fsurf_invalid("domain limits must be finite"));
    }
    if x_min >= x_max || y_min >= y_max {
        return Err(fsurf_invalid(
            "domain lower bounds must be less than upper bounds",
        ));
    }
    Ok(Domain {
        x_min,
        x_max,
        y_min,
        y_max,
    })
}

fn numeric_vector(value: &Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => Ok(tensor.data.clone()),
        other => Err(fsurf_invalid(format!(
            "expected numeric domain vector, got {other:?}"
        ))),
    }
}

fn split_mesh_density(args: Vec<Value>) -> BuiltinResult<(usize, Vec<Value>)> {
    let mut mesh_density = DEFAULT_MESH_DENSITY;
    let mut style_args = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        let Some(key) = value_as_string(&args[idx]) else {
            return Err(fsurf_invalid("name-value option names must be strings"));
        };
        if idx + 1 >= args.len() {
            return Err(fsurf_invalid("name-value arguments must come in pairs"));
        }
        if key.trim().eq_ignore_ascii_case("MeshDensity") {
            mesh_density = parse_mesh_density(&args[idx + 1])?;
        } else {
            style_args.push(args[idx].clone());
            style_args.push(args[idx + 1].clone());
        }
        idx += 2;
    }
    Ok((mesh_density, style_args))
}

fn parse_mesh_density(value: &Value) -> BuiltinResult<usize> {
    if let Some(count) = exact_integer_scalar(value) {
        return parse_mesh_density_integer(&count);
    }
    let values = numeric_vector(value)?;
    if values.len() != 1 {
        return Err(fsurf_invalid("MeshDensity must be a scalar"));
    }
    let raw = values[0];
    if !raw.is_finite() {
        return Err(fsurf_invalid("MeshDensity must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1.0e-9 || rounded < 2.0 {
        return Err(fsurf_invalid(
            "MeshDensity must be an integer greater than or equal to 2",
        ));
    }
    let density = rounded as usize;
    if density > MAX_MESH_DENSITY {
        return Err(fsurf_invalid(format!(
            "MeshDensity must be at most {MAX_MESH_DENSITY}"
        )));
    }
    Ok(density)
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn parse_mesh_density_integer(value: &IntValue) -> BuiltinResult<usize> {
    let Some(density) = value.try_to_usize() else {
        return Err(fsurf_invalid(
            "MeshDensity must be an integer greater than or equal to 2",
        ));
    };
    if density < 2 {
        return Err(fsurf_invalid(
            "MeshDensity must be an integer greater than or equal to 2",
        ));
    }
    if density > MAX_MESH_DENSITY {
        return Err(fsurf_invalid(format!(
            "MeshDensity must be at most {MAX_MESH_DENSITY}"
        )));
    }
    Ok(density)
}

async fn sample_surface(parsed: &ParsedFsurf) -> BuiltinResult<SurfacePlot> {
    let x_axis = linspace(
        parsed.domain.x_min,
        parsed.domain.x_max,
        parsed.mesh_density,
    );
    let y_axis = linspace(
        parsed.domain.y_min,
        parsed.domain.y_max,
        parsed.mesh_density,
    );
    let mut z_grid = vec![vec![0.0; y_axis.len()]; x_axis.len()];
    let mut x_grid = Vec::new();
    let mut y_grid = Vec::new();

    match &parsed.function {
        SurfaceFunction::Explicit(function) => {
            for (i, &x) in x_axis.iter().enumerate() {
                for (j, &y) in y_axis.iter().enumerate() {
                    z_grid[i][j] = call_surface_function(function, x, y).await?;
                }
            }
        }
        SurfaceFunction::Parametric { x, y, z } => {
            x_grid = vec![vec![0.0; y_axis.len()]; x_axis.len()];
            y_grid = vec![vec![0.0; y_axis.len()]; x_axis.len()];
            for (i, &u) in x_axis.iter().enumerate() {
                for (j, &v) in y_axis.iter().enumerate() {
                    x_grid[i][j] = call_surface_function(x, u, v).await?;
                    y_grid[i][j] = call_surface_function(y, u, v).await?;
                    z_grid[i][j] = call_surface_function(z, u, v).await?;
                }
            }
        }
    }

    let defaults = SurfaceStyleDefaults::new(
        ColorMap::Parula,
        ShadingMode::Smooth,
        false,
        1.0,
        false,
        true,
    );
    let style = parse_surface_style_args(BUILTIN_NAME, &parsed.style_args, defaults)
        .map_err(map_fsurf_invalid)?;
    let mut surface = match &parsed.function {
        SurfaceFunction::Explicit(_) => SurfacePlot::new(x_axis, y_axis, z_grid)
            .map_err(|err| plotting_error(BUILTIN_NAME, format!("fsurf: {err}")))?,
        SurfaceFunction::Parametric { .. } => {
            SurfacePlot::from_coordinate_grids(x_grid, y_grid, z_grid)
                .map_err(|err| plotting_error(BUILTIN_NAME, format!("fsurf: {err}")))?
        }
    };
    style.apply_to_plot(&mut surface);
    Ok(surface)
}

async fn call_surface_function(function: &Value, a: f64, b: f64) -> BuiltinResult<f64> {
    let value = call_function(function, vec![Value::Num(a), Value::Num(b)])
        .await
        .map_err(map_fsurf_eval)?;
    let value = crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(map_fsurf_eval)?;
    surface_value_to_scalar(value).map_err(map_fsurf_eval)
}

fn surface_value_to_scalar(value: Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Bool(value) => Ok(if value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            Ok(tensor_utils::tensor_value_f64(&tensor, 0))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(if array.data[0] != 0 { 1.0 } else { 0.0 })
        }
        other => Err(fsurf_error_with_detail(
            &FSURF_ERROR_EVALUATION,
            format!("function output must be a scalar real numeric value, got {other:?}"),
        )),
    }
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / (count as f64 - 1.0);
    (0..count).map(|idx| start + step * idx as f64).collect()
}

fn render_fsurf(surface: SurfacePlot, parsed: &ParsedFsurf) -> BuiltinResult<f64> {
    apply_axes_target(parsed.target_axes, BUILTIN_NAME)?;
    let mut surface = Some(surface);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = current_figure_handle();
    let target_axes_index = parsed.target_axes.map(|(_, axes)| axes);
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Function Surface",
            x_label: "X",
            y_label: "Y",
            axis_equal: false,
            ..Default::default()
        },
        move |figure, axes| {
            let axes = target_axes_index.unwrap_or(axes);
            let plot_index =
                figure.add_surface_plot_on_axes(surface.take().expect("fsurf consumed once"), axes);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = register_function_surface_handle(
        figure_handle,
        axes,
        plot_index,
        parsed.mesh_density,
        (parsed.domain.x_min, parsed.domain.x_max),
        (parsed.domain.y_min, parsed.domain.y_max),
        function_surface_state(&parsed.function),
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

fn function_surface_state(function: &SurfaceFunction) -> FunctionSurfaceFunctionState {
    match function {
        SurfaceFunction::Explicit(function) => {
            FunctionSurfaceFunctionState::Explicit(function_surface_ref(function))
        }
        SurfaceFunction::Parametric { x, y, z } => FunctionSurfaceFunctionState::Parametric {
            x: function_surface_ref(x),
            y: function_surface_ref(y),
            z: function_surface_ref(z),
        },
    }
}

pub(crate) fn function_surface_ref(function: &Value) -> FunctionSurfaceFunctionRef {
    match function {
        Value::FunctionHandle(name) => FunctionSurfaceFunctionRef::FunctionHandle(name.clone()),
        Value::ExternalFunctionHandle(name) => {
            FunctionSurfaceFunctionRef::ExternalFunctionHandle(name.clone())
        }
        Value::MethodFunctionHandle(name) => {
            FunctionSurfaceFunctionRef::MethodFunctionHandle(name.clone())
        }
        Value::BoundFunctionHandle { name, function } => {
            FunctionSurfaceFunctionRef::BoundFunctionHandle {
                name: name.clone(),
                function: *function,
            }
        }
        Value::Closure(closure) => FunctionSurfaceFunctionRef::ClosureSummary {
            function_name: closure.function_name.clone(),
            bound_function: closure.bound_function,
        },
        _ => FunctionSurfaceFunctionRef::FunctionHandle("<invalid>".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use futures::executor::block_on;
    use runmat_plot::plots::PlotElement;
    use std::sync::Arc;

    fn with_test_function(
        f: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
    ) -> crate::user_functions::FunctionInvokerGuard {
        let f = Arc::new(f);
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let f = Arc::clone(&f);
                let a = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected first scalar argument, got {other:?}"),
                };
                let b = match &args[1] {
                    Value::Num(value) => *value,
                    other => panic!("expected second scalar argument, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(f(a, b))) })
            },
        )))
    }

    #[test]
    fn fsurf_samples_explicit_function_surface() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let _invoker = with_test_function(|x, y| x * x + y);

        let handle = block_on(fsurf_builtin(vec![
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::Tensor(
                runmat_builtins::Tensor::new(vec![0.0, 1.0, 0.0, 2.0], vec![1, 4]).unwrap(),
            ),
            Value::String("MeshDensity".into()),
            Value::Num(3.0),
            Value::String("DisplayName".into()),
            Value::String("parabola".into()),
        ]))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functionsurface".into()));
        let mesh_density = get_builtin(vec![
            Value::Num(handle),
            Value::String("MeshDensity".into()),
        ])
        .unwrap();
        assert_eq!(mesh_density, Value::Num(3.0));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        assert_eq!(surface.x_data.len(), 3);
        assert_eq!(surface.y_data.len(), 3);
        assert_eq!(surface.label.as_deref(), Some("parabola"));
        let z = surface.z_data.as_ref().unwrap();
        assert!((z[2][2] - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn fsurf_supports_axes_target_and_parametric_surface() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let u = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected first scalar argument, got {other:?}"),
                };
                let v = match &args[1] {
                    Value::Num(value) => *value,
                    other => panic!("expected second scalar argument, got {other:?}"),
                };
                Box::pin(async move {
                    Ok(match _function {
                        11 => Value::Num(u),
                        12 => Value::Num(v),
                        13 => Value::Num(u + v),
                        other => panic!("unexpected function id {other}"),
                    })
                })
            },
        )));

        let handle = block_on(fsurf_builtin(vec![
            Value::Num(ax),
            Value::BoundFunctionHandle {
                name: "x".into(),
                function: 11,
            },
            Value::BoundFunctionHandle {
                name: "y".into(),
                function: 12,
            },
            Value::BoundFunctionHandle {
                name: "z".into(),
                function: 13,
            },
            Value::Tensor(
                runmat_builtins::Tensor::new(vec![-1.0, 1.0, 2.0, 4.0], vec![1, 4]).unwrap(),
            ),
            Value::String("MeshDensity".into()),
            Value::Num(2.0),
        ]))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functionsurface".into()));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        assert!(surface.x_grid.is_some());
        assert!(surface.y_grid.is_some());
        let z = surface.z_data.as_ref().unwrap();
        assert!((z[0][0] - 1.0).abs() < 1.0e-12);
        assert!((z[1][1] - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn fsurf_allows_nonfinite_explicit_samples() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let _invoker = with_test_function(|x, y| {
            if x == 0.0 && y == 0.0 {
                f64::NAN
            } else {
                x + y
            }
        });

        let handle = block_on(fsurf_builtin(vec![
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::Tensor(
                runmat_builtins::Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![1, 4]).unwrap(),
            ),
            Value::String("MeshDensity".into()),
            Value::Num(2.0),
        ]))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functionsurface".into()));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface plot");
        };
        let z = surface.z_data.as_ref().unwrap();
        assert!(z[0][0].is_nan());
        assert_eq!(z[1][1], 2.0);
    }

    #[test]
    fn fsurf_rejects_invalid_mesh_density() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _invoker = with_test_function(|x, y| x + y);
        let err = block_on(fsurf_builtin(vec![
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::String("MeshDensity".into()),
            Value::Num(1.0),
        ]))
        .expect_err("expected mesh-density validation error");
        assert_eq!(err.identifier(), FSURF_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn fsurf_mesh_density_reads_typed_integer_tensor_exactly() {
        let exact = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![400]),
            vec![1, 1],
        )
        .expect("typed density");
        assert_eq!(parse_mesh_density(&Value::Tensor(exact)).unwrap(), 400);

        let too_large = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![401]),
            vec![1, 1],
        )
        .expect("large density");
        assert!(parse_mesh_density(&Value::Tensor(too_large)).is_err());

        let negative = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::I64(vec![-1]),
            vec![1, 1],
        )
        .expect("negative density");
        assert!(parse_mesh_density(&Value::Tensor(negative)).is_err());
    }

    #[test]
    fn fsurf_function_scalar_reads_typed_integer_storage_exactly() {
        let mut tensor = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![12]),
            vec![1, 1],
        )
        .unwrap();
        tensor.data[0] = -3.0;

        assert_eq!(
            surface_value_to_scalar(Value::Tensor(tensor)).unwrap(),
            12.0
        );
    }

    #[test]
    fn fsurf_descriptor_covers_parametric_and_axes_forms() {
        let labels: Vec<&str> = FSURF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = fsurf(f)"));
        assert!(labels.contains(&"h = fsurf(fx, fy, fz)"));
        assert!(labels.contains(&"h = fsurf(ax, ___)"));
    }
}
