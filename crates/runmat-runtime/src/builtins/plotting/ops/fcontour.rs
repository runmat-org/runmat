//! MATLAB-compatible `fcontour` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ContourPlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::call_function;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::common::SurfaceDataInput;
use super::contour::{
    apply_contour_options, build_contour_plot, parse_level_spec, ContourArgs, ContourLevelSpec,
    ContourLineColor,
};
use super::fsurf::function_surface_ref;
use super::op_common::{apply_axes_target, split_leading_axes_handle, AxesTarget};
use super::plotting_error;
use super::state::{
    current_figure_handle, register_function_contour_handle, render_active_plot, PlotRenderOptions,
};
use super::style::value_as_string;

const BUILTIN_NAME: &str = "fcontour";
const DEFAULT_DOMAIN: Domain = Domain {
    x_min: -5.0,
    x_max: 5.0,
    y_min: -5.0,
    y_max: 5.0,
};
const DEFAULT_MESH_DENSITY: usize = 71;
const MAX_MESH_DENSITY: usize = 400;

const FCONTOUR_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered function contour.",
}];

const FCONTOUR_INPUTS_F: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function of two variables, z = f(x,y).",
}];

const FCONTOUR_INPUTS_F_DOMAIN: [BuiltinParamDescriptor; 2] = [
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

const FCONTOUR_INPUTS_PROPS: [BuiltinParamDescriptor; 2] = [
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
        description: "MeshDensity and contour name/value arguments.",
    },
];

const FCONTOUR_INPUTS_AX_PROPS: [BuiltinParamDescriptor; 3] = [
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
        description: "MeshDensity and contour name/value arguments.",
    },
];

const FCONTOUR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = fcontour(f)",
        inputs: &FCONTOUR_INPUTS_F,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fcontour(f, xyinterval)",
        inputs: &FCONTOUR_INPUTS_F_DOMAIN,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fcontour(___, Name, Value, ...)",
        inputs: &FCONTOUR_INPUTS_PROPS,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fcontour(ax, ___)",
        inputs: &FCONTOUR_INPUTS_AX_PROPS,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
];

pub const FCONTOUR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCONTOUR.INVALID_ARGUMENT",
    identifier: Some("RunMat:fcontour:InvalidArgument"),
    when: "Function handle, domain, mesh density, axes target, levels, or contour properties are invalid.",
    message: "fcontour: invalid argument",
};

pub const FCONTOUR_ERROR_EVALUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCONTOUR.EVALUATION",
    identifier: Some("RunMat:fcontour:EvaluationFailed"),
    when: "A sampled function handle fails or does not return a scalar numeric value.",
    message: "fcontour: function evaluation failed",
};

pub const FCONTOUR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCONTOUR.INTERNAL",
    identifier: Some("RunMat:fcontour:Internal"),
    when: "Contour construction or rendering fails unexpectedly.",
    message: "fcontour: internal operation failed",
};

const FCONTOUR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FCONTOUR_ERROR_INVALID_ARGUMENT,
    FCONTOUR_ERROR_EVALUATION,
    FCONTOUR_ERROR_INTERNAL,
];

pub const FCONTOUR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FCONTOUR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FCONTOUR_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::fcontour")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fcontour",
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
    notes: "fcontour samples arbitrary MATLAB function handles on the host, then renders through the existing contour plot pipeline.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::fcontour")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fcontour",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fcontour performs callback sampling and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "fcontour",
    category = "plotting",
    summary = "Plot contour lines of a function over a 2-D domain.",
    keywords = "fcontour,function contour,contour,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::fcontour::FCONTOUR_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::fcontour"
)]
pub async fn fcontour_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    let parsed = parse_fcontour_args(args).map_err(map_fcontour_invalid)?;
    let plot = sample_contour(&parsed).await?;
    render_fcontour(plot, &parsed).map_err(map_fcontour_internal)
}

#[derive(Clone, Copy)]
struct Domain {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
}

struct ParsedFcontour {
    target_axes: AxesTarget,
    function: Value,
    domain: Domain,
    mesh_density: usize,
    level_spec: ContourLevelSpec,
    contour_options: Vec<Value>,
    display_name: Option<String>,
}

fn fcontour_error_with_detail(
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

fn fcontour_invalid(detail: impl AsRef<str>) -> RuntimeError {
    fcontour_error_with_detail(&FCONTOUR_ERROR_INVALID_ARGUMENT, detail)
}

fn map_fcontour_invalid(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fcontour_error_with_detail(&FCONTOUR_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_fcontour_eval(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fcontour_error_with_detail(&FCONTOUR_ERROR_EVALUATION, err.message)
}

fn map_fcontour_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fcontour_error_with_detail(&FCONTOUR_ERROR_INTERNAL, err.message)
}

fn parse_fcontour_args(args: Vec<Value>) -> BuiltinResult<ParsedFcontour> {
    if args.is_empty() {
        return Err(fcontour_invalid("expected a function handle"));
    }

    let (target_axes, mut values) = split_leading_axes_handle(args, BUILTIN_NAME)?;
    if values.is_empty() {
        return Err(fcontour_invalid(
            "expected a function handle after axes handle",
        ));
    }
    if !is_function_handle(&values[0]) {
        return Err(fcontour_invalid("expected a function handle"));
    }
    let function = values.remove(0);

    let mut domain = DEFAULT_DOMAIN;
    if values.first().is_some_and(is_domain_value) {
        domain = parse_domain(&values.remove(0))?;
    }

    let mut level_spec = ContourLevelSpec::Auto;
    if values.first().is_some_and(is_level_value) {
        level_spec =
            parse_level_spec(values.remove(0), BUILTIN_NAME).map_err(map_fcontour_invalid)?;
    }

    let (mesh_density, display_name, contour_options) = split_fcontour_options(values)?;
    Ok(ParsedFcontour {
        target_axes,
        function,
        domain,
        mesh_density,
        level_spec,
        contour_options,
        display_name,
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

fn is_domain_value(value: &Value) -> bool {
    let Ok(values) = numeric_vector(value) else {
        return false;
    };
    matches!(values.len(), 2 | 4)
}

fn is_level_value(value: &Value) -> bool {
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
        _ => Err(fcontour_invalid(
            "domain must be a two-element or four-element numeric vector",
        )),
    }
}

fn domain_from_values(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> BuiltinResult<Domain> {
    if ![x_min, x_max, y_min, y_max]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(fcontour_invalid("domain limits must be finite"));
    }
    if x_min >= x_max || y_min >= y_max {
        return Err(fcontour_invalid(
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
        other => Err(fcontour_invalid(format!(
            "expected numeric vector, got {other:?}"
        ))),
    }
}

fn split_fcontour_options(args: Vec<Value>) -> BuiltinResult<(usize, Option<String>, Vec<Value>)> {
    let mut mesh_density = DEFAULT_MESH_DENSITY;
    let mut display_name = None;
    let mut contour_options = Vec::new();
    let mut idx = 0usize;
    if let Some(token) = args.first().and_then(value_as_string) {
        let trimmed = token.trim();
        if !is_fcontour_special_option(trimmed) && !is_contour_pair_option(trimmed) {
            contour_options.push(args[0].clone());
            idx = 1;
        }
    }
    while idx < args.len() {
        let Some(key) = value_as_string(&args[idx]) else {
            return Err(fcontour_invalid("name-value option names must be strings"));
        };
        if idx + 1 >= args.len() {
            return Err(fcontour_invalid("name-value arguments must come in pairs"));
        }
        let normalized = key.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "meshdensity" => {
                mesh_density = parse_mesh_density(&args[idx + 1])?;
            }
            "displayname" => {
                display_name = value_as_string(&args[idx + 1]).map(|s| s.to_string());
            }
            _ => {
                contour_options.push(args[idx].clone());
                contour_options.push(args[idx + 1].clone());
            }
        }
        idx += 2;
    }
    Ok((mesh_density, display_name, contour_options))
}

fn is_fcontour_special_option(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "meshdensity" | "displayname"
    )
}

fn is_contour_pair_option(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "levellist"
            | "levels"
            | "levelstep"
            | "linecolor"
            | "color"
            | "linewidth"
            | "levellistmode"
    )
}

fn parse_mesh_density(value: &Value) -> BuiltinResult<usize> {
    if let Some(count) = exact_integer_scalar(value) {
        return parse_mesh_density_integer(&count);
    }
    let values = numeric_vector(value)?;
    if values.len() != 1 {
        return Err(fcontour_invalid("MeshDensity must be a scalar"));
    }
    let raw = values[0];
    if !raw.is_finite() {
        return Err(fcontour_invalid("MeshDensity must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1.0e-9 || rounded < 2.0 {
        return Err(fcontour_invalid(
            "MeshDensity must be an integer greater than or equal to 2",
        ));
    }
    let density = rounded as usize;
    if density > MAX_MESH_DENSITY {
        return Err(fcontour_invalid(format!(
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
        return Err(fcontour_invalid(
            "MeshDensity must be an integer greater than or equal to 2",
        ));
    };
    if density < 2 {
        return Err(fcontour_invalid(
            "MeshDensity must be an integer greater than or equal to 2",
        ));
    }
    if density > MAX_MESH_DENSITY {
        return Err(fcontour_invalid(format!(
            "MeshDensity must be at most {MAX_MESH_DENSITY}"
        )));
    }
    Ok(density)
}

async fn sample_contour(parsed: &ParsedFcontour) -> BuiltinResult<ContourPlot> {
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
    let mut grid = vec![vec![0.0; x_axis.len()]; y_axis.len()];
    for (row, &y) in y_axis.iter().enumerate() {
        for (col, &x) in x_axis.iter().enumerate() {
            grid[row][col] = call_contour_function(&parsed.function, x, y).await?;
        }
    }

    let dummy = Tensor::new(vec![0.0], vec![1, 1])
        .map_err(|err| plotting_error(BUILTIN_NAME, format!("fcontour: {err}")))?;
    let mut contour_args = ContourArgs {
        name: BUILTIN_NAME,
        x_axis: x_axis.clone(),
        y_axis: y_axis.clone(),
        z_input: SurfaceDataInput::Host(dummy),
        level_spec: parsed.level_spec.clone(),
        line_color: ContourLineColor::Auto,
        line_width: 1.0,
    };
    apply_contour_options(&mut contour_args, &parsed.contour_options)
        .map_err(map_fcontour_invalid)?;

    if matches!(contour_args.line_color, ContourLineColor::None) {
        return build_contour_plot(
            BUILTIN_NAME,
            &x_axis,
            &y_axis,
            &grid,
            ColorMap::Parula,
            0.0,
            &ContourLevelSpec::Values(vec![0.0]),
            &ContourLineColor::None,
        )
        .map_err(map_fcontour_internal);
    }

    let mut plot = build_contour_plot(
        BUILTIN_NAME,
        &x_axis,
        &y_axis,
        &grid,
        ColorMap::Parula,
        0.0,
        &contour_args.level_spec,
        &contour_args.line_color,
    )
    .map_err(map_fcontour_internal)?
    .with_line_width(contour_args.line_width)
    .with_label("Function Contours");
    if let Some(display_name) = &parsed.display_name {
        plot.label = Some(display_name.clone());
    }
    Ok(plot)
}

async fn call_contour_function(function: &Value, x: f64, y: f64) -> BuiltinResult<f64> {
    let value = call_function(function, vec![Value::Num(x), Value::Num(y)])
        .await
        .map_err(map_fcontour_eval)?;
    let value = crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(map_fcontour_eval)?;
    contour_value_to_scalar(value).map_err(map_fcontour_eval)
}

fn contour_value_to_scalar(value: Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Bool(value) => Ok(if value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(if array.data[0] != 0 { 1.0 } else { 0.0 })
        }
        other => Err(fcontour_error_with_detail(
            &FCONTOUR_ERROR_EVALUATION,
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

fn render_fcontour(plot: ContourPlot, parsed: &ParsedFcontour) -> BuiltinResult<f64> {
    apply_axes_target(parsed.target_axes, BUILTIN_NAME)?;
    let mut plot = Some(plot);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = current_figure_handle();
    let target_axes_index = parsed.target_axes.map(|(_, axes)| axes);
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Function Contour",
            x_label: "X",
            y_label: "Y",
            axis_equal: true,
            ..Default::default()
        },
        move |figure, axes| {
            let axes = target_axes_index.unwrap_or(axes);
            let plot_index =
                figure.add_contour_plot_on_axes(plot.take().expect("fcontour consumed once"), axes);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = register_function_contour_handle(
        figure_handle,
        axes,
        plot_index,
        parsed.mesh_density,
        (parsed.domain.x_min, parsed.domain.x_max),
        (parsed.domain.y_min, parsed.domain.y_max),
        function_surface_ref(&parsed.function),
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
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected first scalar argument, got {other:?}"),
                };
                let y = match &args[1] {
                    Value::Num(value) => *value,
                    other => panic!("expected second scalar argument, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(f(x, y))) })
            },
        )))
    }

    #[test]
    fn fcontour_samples_function_and_returns_function_contour_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let _invoker = with_test_function(|x, y| x * x + y);

        let handle = block_on(fcontour_builtin(vec![
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::Tensor(
                runmat_builtins::Tensor::new(vec![0.0, 1.0, 0.0, 2.0], vec![1, 4]).unwrap(),
            ),
            Value::String("MeshDensity".into()),
            Value::Num(5.0),
            Value::String("LevelList".into()),
            Value::Tensor(runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::String("LineWidth".into()),
            Value::Num(2.0),
            Value::String("DisplayName".into()),
            Value::String("parabola".into()),
        ]))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functioncontour".into()));
        let mesh_density = get_builtin(vec![
            Value::Num(handle),
            Value::String("MeshDensity".into()),
        ])
        .unwrap();
        assert_eq!(mesh_density, Value::Num(5.0));
        let display_name = get_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayName".into()),
        ])
        .unwrap();
        assert_eq!(display_name, Value::String("parabola".into()));

        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Contour(contour) = fig.plots().next().unwrap() else {
            panic!("expected contour plot");
        };
        assert_eq!(contour.label.as_deref(), Some("parabola"));
        assert_eq!(contour.line_width, 2.0);
    }

    #[test]
    fn fcontour_supports_axes_target_and_linespec() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let _invoker = with_test_function(|x, y| x - y);

        let handle = block_on(fcontour_builtin(vec![
            Value::Num(ax),
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::Tensor(
                runmat_builtins::Tensor::new(vec![-1.0, 1.0, -1.0, 1.0], vec![1, 4]).unwrap(),
            ),
            Value::String("r--".into()),
            Value::String("MeshDensity".into()),
            Value::Num(3.0),
        ]))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functioncontour".into()));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
    }

    #[test]
    fn fcontour_rejects_invalid_mesh_density() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _invoker = with_test_function(|x, y| x + y);
        let err = block_on(fcontour_builtin(vec![
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::String("MeshDensity".into()),
            Value::Num(1.0),
        ]))
        .expect_err("expected mesh-density validation error");
        assert_eq!(err.identifier(), FCONTOUR_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn fcontour_mesh_density_reads_typed_integer_tensor_exactly() {
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
}
