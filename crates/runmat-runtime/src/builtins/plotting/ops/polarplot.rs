//! MATLAB-compatible `polarplot` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{AxesKind, LinePlot, LineStyle};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::common::{gather_tensor_from_gpu_async, numeric_pair};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::plot::build_line_plot;
use super::plotting_error;
use super::state::{
    current_axes_state, current_hold_enabled, line_color_for_series_index,
    next_line_color_for_axes, next_line_style_for_axes, render_active_plot,
    set_line_style_order_for_axes, PlotRenderOptions,
};
use super::style::{
    looks_like_option_name, parse_line_style_args, value_as_string, LineAppearance,
    LineStyleParseOptions,
};

const BUILTIN_NAME: &str = "polarplot";

const POLARPLOT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the first polar line object.",
}];

const POLARPLOT_INPUTS_THETA_RHO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "theta",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Angular coordinates in radians.",
    },
    BuiltinParamDescriptor {
        name: "rho",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Radial coordinates.",
    },
];

const POLARPLOT_INPUTS_RHO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "rhoOrZ",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Radial data with implicit theta, or complex data interpreted as angle/abs.",
}];

const POLARPLOT_INPUTS_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Polar plot inputs with optional axes handle, LineSpec, and Name/Value pairs.",
}];

const POLARPLOT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = polarplot(theta, rho)",
        inputs: &POLARPLOT_INPUTS_THETA_RHO,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarplot(theta, rho, LineSpec)",
        inputs: &POLARPLOT_INPUTS_ARGS,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarplot(theta1, rho1, ..., thetaN, rhoN)",
        inputs: &POLARPLOT_INPUTS_ARGS,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarplot(rho)",
        inputs: &POLARPLOT_INPUTS_RHO,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarplot(Z)",
        inputs: &POLARPLOT_INPUTS_RHO,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
];

pub const POLARPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:polarplot:InvalidArgument"),
    when: "Input data, series grammar, axes targeting, or style arguments are invalid.",
    message: "polarplot: invalid argument",
};

pub const POLARPLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARPLOT.INTERNAL",
    identifier: Some("RunMat:polarplot:Internal"),
    when: "Internal polar line construction or rendering fails unexpectedly.",
    message: "polarplot: internal operation failed",
};

const POLARPLOT_ERRORS: [BuiltinErrorDescriptor; 2] =
    [POLARPLOT_ERROR_INVALID_ARGUMENT, POLARPLOT_ERROR_INTERNAL];

pub const POLARPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POLARPLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POLARPLOT_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::polarplot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "polarplot",
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
    notes: "polarplot converts polar inputs to cartesian line vertices before rendering; GPU-resident real inputs are gathered for the conversion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::polarplot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "polarplot",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "polarplot performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "polarplot",
    category = "plotting",
    summary = "Create 2-D line plots in polar coordinates.",
    keywords = "polarplot,polar,plot,line,visualization",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::polarplot::POLARPLOT_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::polarplot"
)]
pub async fn polarplot_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    let (axes_target, args) =
        split_leading_axes_handle(args, BUILTIN_NAME).map_err(map_polarplot_invalid_argument)?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_polarplot_invalid_argument)?;

    let (mut series_plans, line_style_order) =
        parse_polar_series_specs(args).map_err(map_polarplot_invalid_argument)?;
    let axes = current_axes_state().active_index;
    let hold_enabled = current_hold_enabled();
    if let Some(order) = line_style_order.as_ref() {
        apply_polar_line_style_order(&mut series_plans, order);
        set_line_style_order_for_axes(axes, order);
    }

    let mut expanded_series = Vec::new();
    for plan in series_plans {
        let evaluated = evaluate_polar_data(plan.data).await?;
        for data in evaluated {
            expanded_series.push(ExpandedPolarSeries {
                data,
                appearance: plan.appearance.clone(),
                line_style_explicit: plan.line_style_explicit,
                color_explicit: plan.color_explicit,
                label: plan.label.clone(),
            });
        }
    }

    let total = expanded_series.len();
    let mut plots: Vec<LinePlot> = Vec::with_capacity(total);
    let mut max_radius = 1.0_f64;
    for (series_idx, mut series) in expanded_series.drain(..).enumerate() {
        if !series.line_style_explicit {
            series.appearance.line_style = if hold_enabled {
                next_line_style_for_axes(axes)
            } else {
                LineStyle::Solid
            };
            series.line_style_explicit = true;
        }
        if !series.color_explicit {
            series.appearance.color = if hold_enabled {
                next_line_color_for_axes(axes)
            } else {
                line_color_for_series_index(series_idx)
            };
        }

        let label = series.label.take().unwrap_or_else(|| {
            if total == 1 {
                "Data".to_string()
            } else {
                format!("Series {}", series_idx + 1)
            }
        });
        max_radius = max_radius.max(
            series
                .data
                .rho
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.abs())
                .fold(0.0, f64::max),
        );
        let (x, y) = polar_to_cartesian(&series.data.theta, &series.data.rho)?;
        plots.push(build_line_plot(x, y, &label, &series.appearance)?);
    }

    let limit = max_radius.max(1.0);
    let mut plots_opt = Some(plots);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let opts = PlotRenderOptions {
        title: "Polar Plot",
        x_label: "",
        y_label: "",
        axis_equal: true,
        grid: true,
    };
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes_index| {
        figure.set_axes_kind(axes_index, AxesKind::Polar);
        figure.set_axes_axis_equal(axes_index, true);
        figure.set_axes_limits(axes_index, Some((-limit, limit)), Some((-limit, limit)));
        figure.set_axes_labels(axes_index, "", "");

        let plots = plots_opt
            .take()
            .expect("polarplot series consumed exactly once");
        for (idx, plot) in plots.into_iter().enumerate() {
            let plot_index = figure.add_line_plot_on_axes(plot, axes_index);
            if idx == 0 {
                *plot_index_slot.borrow_mut() = Some((axes_index, plot_index));
            }
        }
        Ok(())
    });
    let Some((axes_index, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_line_handle(
        figure_handle,
        axes_index,
        plot_index,
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_polarplot_internal(err));
    }
    Ok(handle)
}

#[derive(Clone, Debug)]
struct PolarSeriesPlan {
    data: PolarSeriesInput,
    appearance: LineAppearance,
    line_style_explicit: bool,
    color_explicit: bool,
    label: Option<String>,
}

#[derive(Clone, Debug)]
struct ExpandedPolarSeries {
    data: EvaluatedPolarData,
    appearance: LineAppearance,
    line_style_explicit: bool,
    color_explicit: bool,
    label: Option<String>,
}

#[derive(Clone, Debug)]
enum PolarSeriesInput {
    ThetaRho(Value, Value),
    Rho(Value),
    Complex(ComplexInput),
}

#[derive(Clone, Debug)]
enum ComplexInput {
    Scalar(f64, f64),
    Tensor(ComplexTensor),
}

#[derive(Clone, Debug)]
struct EvaluatedPolarData {
    theta: Vec<f64>,
    rho: Vec<f64>,
}

fn parse_polar_series_specs(
    args: Vec<Value>,
) -> BuiltinResult<(Vec<PolarSeriesPlan>, Option<Vec<LineStyle>>)> {
    if args.is_empty() {
        return Err(polarplot_err("expected at least one data series"));
    }

    let mut plans = Vec::new();
    let mut line_style_order: Option<Vec<LineStyle>> = None;
    let mut inline_opts = LineStyleParseOptions::plot();
    inline_opts.forbid_leading_numeric = false;

    let mut idx = 0usize;
    while idx < args.len() {
        let first_val = args[idx].clone();
        idx += 1;

        let data = match first_val {
            Value::Complex(re, im) => PolarSeriesInput::Complex(ComplexInput::Scalar(re, im)),
            Value::ComplexTensor(tensor) => PolarSeriesInput::Complex(ComplexInput::Tensor(tensor)),
            value if is_real_numeric_value(&value) => {
                let shorthand_rho = match args.get(idx) {
                    None => true,
                    Some(Value::String(_) | Value::CharArray(_)) => true,
                    Some(next) => !is_real_numeric_value(next),
                };
                if shorthand_rho {
                    PolarSeriesInput::Rho(value)
                } else {
                    let rho = args[idx].clone();
                    idx += 1;
                    PolarSeriesInput::ThetaRho(value, rho)
                }
            }
            _ => {
                return Err(polarplot_err(
                    "expected numeric theta/rho data or complex Z data before style arguments",
                ));
            }
        };

        let mut style_tokens = Vec::new();
        loop {
            let should_consume =
                matches!(args.get(idx), Some(Value::String(_) | Value::CharArray(_)));
            if !should_consume {
                break;
            }
            let token = args[idx].clone();
            idx += 1;
            let token_text = value_as_string(&token)
                .ok_or_else(|| polarplot_err("style tokens must be char arrays or strings"))?;
            let lower = token_text.trim().to_ascii_lowercase();
            style_tokens.push(token);

            if looks_like_option_name(&lower) {
                if idx >= args.len() {
                    return Err(polarplot_err("name-value arguments must come in pairs"));
                }
                style_tokens.push(args[idx].clone());
                idx += 1;
            }
        }

        let applies_to_all_existing = !plans.is_empty()
            && idx == args.len()
            && style_tokens_are_name_value_only(&style_tokens);
        let parsed_style = parse_line_style_args(&style_tokens, &inline_opts)?;
        if let Some(order) = parsed_style.line_style_order.clone() {
            line_style_order = Some(order);
        }
        let plan = PolarSeriesPlan {
            data,
            appearance: parsed_style.appearance,
            line_style_explicit: parsed_style.line_style_explicit,
            color_explicit: parsed_style.color_explicit,
            label: parsed_style.label.clone(),
        };
        if applies_to_all_existing {
            for existing in plans.iter_mut() {
                merge_group_style(existing, &plan);
            }
        }
        plans.push(plan);
    }

    if plans.is_empty() {
        return Err(polarplot_err("expected at least one polar data series"));
    }
    Ok((plans, line_style_order))
}

fn style_tokens_are_name_value_only(tokens: &[Value]) -> bool {
    !tokens.is_empty()
        && tokens.len().is_multiple_of(2)
        && tokens
            .chunks_exact(2)
            .all(|pair| value_as_string(&pair[0]).is_some_and(|name| looks_like_option_name(&name)))
}

fn merge_group_style(target: &mut PolarSeriesPlan, source: &PolarSeriesPlan) {
    if source.color_explicit {
        target.appearance.color = source.appearance.color;
        target.color_explicit = true;
    }
    if source.line_style_explicit {
        target.appearance.line_style = source.appearance.line_style;
        target.line_style_explicit = true;
    }
    if source.appearance.line_width != LineAppearance::default().line_width {
        target.appearance.line_width = source.appearance.line_width;
    }
    if source.appearance.marker.is_some() {
        target.appearance.marker = source.appearance.marker.clone();
    }
    if source.label.is_some() {
        target.label = source.label.clone();
    }
}

async fn evaluate_polar_data(input: PolarSeriesInput) -> BuiltinResult<Vec<EvaluatedPolarData>> {
    match input {
        PolarSeriesInput::ThetaRho(theta, rho) => {
            let theta = real_tensor_from_value(theta).await?;
            let rho = real_tensor_from_value(rho).await?;
            evaluate_theta_rho_tensors(theta, rho)
        }
        PolarSeriesInput::Rho(rho) => {
            let rho = real_tensor_from_value(rho).await?;
            Ok(tensor_columns(&rho)
                .into_iter()
                .map(|rho| EvaluatedPolarData {
                    theta: implicit_theta(rho.len()),
                    rho,
                })
                .collect())
        }
        PolarSeriesInput::Complex(ComplexInput::Scalar(re, im)) => Ok(vec![EvaluatedPolarData {
            theta: vec![im.atan2(re)],
            rho: vec![re.hypot(im)],
        }]),
        PolarSeriesInput::Complex(ComplexInput::Tensor(tensor)) => {
            Ok(complex_tensor_columns(&tensor)
                .into_iter()
                .map(|column| {
                    let (theta, rho): (Vec<_>, Vec<_>) = column
                        .iter()
                        .map(|(re, im)| (im.atan2(*re), re.hypot(*im)))
                        .unzip();
                    EvaluatedPolarData { theta, rho }
                })
                .collect())
        }
    }
}

fn evaluate_theta_rho_tensors(
    theta: Tensor,
    rho: Tensor,
) -> BuiltinResult<Vec<EvaluatedPolarData>> {
    if tensor_is_vector(&theta) && tensor_is_vector(&rho) {
        let (theta, rho) = numeric_pair(theta, rho, BUILTIN_NAME)?;
        return Ok(vec![EvaluatedPolarData { theta, rho }]);
    }

    if tensor_is_vector(&theta) {
        let theta_vec = theta.data;
        let rho_rows = tensor_rows(&rho);
        if theta_vec.len() != rho_rows {
            return Err(polarplot_err(
                "theta vector length must match each rho matrix column",
            ));
        }
        return Ok(tensor_columns(&rho)
            .into_iter()
            .map(|rho| EvaluatedPolarData {
                theta: theta_vec.clone(),
                rho,
            })
            .collect());
    }

    if tensor_is_vector(&rho) {
        let rho_vec = rho.data;
        let theta_rows = tensor_rows(&theta);
        if rho_vec.len() != theta_rows {
            return Err(polarplot_err(
                "rho vector length must match each theta matrix column",
            ));
        }
        return Ok(tensor_columns(&theta)
            .into_iter()
            .map(|theta| EvaluatedPolarData {
                theta,
                rho: rho_vec.clone(),
            })
            .collect());
    }

    if tensor_rows(&theta) != tensor_rows(&rho) || tensor_cols(&theta) != tensor_cols(&rho) {
        return Err(polarplot_err(
            "theta and rho matrices must have matching row and column counts",
        ));
    }

    Ok(tensor_columns(&theta)
        .into_iter()
        .zip(tensor_columns(&rho))
        .map(|(theta, rho)| EvaluatedPolarData { theta, rho })
        .collect())
}

fn tensor_is_vector(tensor: &Tensor) -> bool {
    tensor.rows <= 1 || tensor.cols <= 1 || tensor.shape.len() <= 1
}

fn tensor_rows(tensor: &Tensor) -> usize {
    if tensor_is_vector(tensor) {
        tensor.data.len()
    } else {
        tensor.rows
    }
}

fn tensor_cols(tensor: &Tensor) -> usize {
    if tensor_is_vector(tensor) {
        1
    } else {
        tensor.cols
    }
}

fn tensor_columns(tensor: &Tensor) -> Vec<Vec<f64>> {
    if tensor_is_vector(tensor) {
        return vec![tensor.data.clone()];
    }
    (0..tensor.cols)
        .map(|col| {
            let start = col * tensor.rows;
            tensor.data[start..start + tensor.rows].to_vec()
        })
        .collect()
}

fn complex_tensor_columns(tensor: &ComplexTensor) -> Vec<Vec<(f64, f64)>> {
    if tensor.rows <= 1 || tensor.cols <= 1 || tensor.shape.len() <= 1 {
        return vec![tensor.data.clone()];
    }
    (0..tensor.cols)
        .map(|col| {
            let start = col * tensor.rows;
            tensor.data[start..start + tensor.rows].to_vec()
        })
        .collect()
}

async fn real_tensor_from_value(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await,
        Value::Num(value) => Ok(scalar_tensor(value)),
        Value::Int(value) => Ok(scalar_tensor(value.to_f64())),
        Value::Bool(value) => Ok(scalar_tensor(if value { 1.0 } else { 0.0 })),
        other => Tensor::try_from(&other)
            .map_err(|err| polarplot_err(format!("expected real numeric data: {err}"))),
    }
}

fn scalar_tensor(value: f64) -> Tensor {
    Tensor {
        data: vec![value],
        shape: vec![1],
        rows: 1,
        cols: 1,
        dtype: runmat_builtins::NumericDType::F64,
    }
}

fn implicit_theta(len: usize) -> Vec<f64> {
    match len {
        0 => Vec::new(),
        1 => vec![0.0],
        n => (0..n)
            .map(|idx| idx as f64 * std::f64::consts::TAU / (n - 1) as f64)
            .collect(),
    }
}

fn polar_to_cartesian(theta: &[f64], rho: &[f64]) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    if theta.len() != rho.len() {
        return Err(polarplot_err(
            "theta and rho inputs must have the same number of elements",
        ));
    }
    let x = theta
        .iter()
        .zip(rho.iter())
        .map(|(theta, rho)| rho * theta.cos())
        .collect();
    let y = theta
        .iter()
        .zip(rho.iter())
        .map(|(theta, rho)| rho * theta.sin())
        .collect();
    Ok((x, y))
}

fn is_real_numeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

fn apply_polar_line_style_order(plans: &mut [PolarSeriesPlan], order: &[LineStyle]) {
    if order.is_empty() {
        return;
    }
    let mut index = 0usize;
    for plan in plans.iter_mut() {
        if !plan.line_style_explicit {
            plan.appearance.line_style = order[index % order.len()];
            index += 1;
        }
    }
}

fn polarplot_error_with_detail(
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

fn map_polarplot_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    polarplot_error_with_detail(&POLARPLOT_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_polarplot_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    polarplot_error_with_detail(&POLARPLOT_ERROR_INTERNAL, err.message)
}

fn polarplot_err(msg: impl Into<String>) -> RuntimeError {
    plotting_error(BUILTIN_NAME, format!("polarplot: {}", msg.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clone_figure, current_figure_handle, reset_hold_state_for_run, reset_plot_state,
    };
    use runmat_builtins::NumericDType;

    fn tensor(data: &[f64]) -> Value {
        Value::Tensor(Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: NumericDType::F64,
        })
    }

    #[test]
    fn polar_to_cartesian_converts_theta_rho_pairs() {
        let (x, y) = polar_to_cartesian(
            &[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
            &[1.0, 2.0, 3.0],
        )
        .unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!(x[1].abs() < 1e-12);
        assert!((y[1] - 2.0).abs() < 1e-12);
        assert!((x[2] + 3.0).abs() < 1e-12);
    }

    #[test]
    fn implicit_theta_spans_zero_to_two_pi() {
        let theta = implicit_theta(5);
        assert_eq!(theta[0], 0.0);
        assert!((theta[4] - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn parses_multiple_series_with_styles_and_name_values() {
        let (plans, _) = parse_polar_series_specs(vec![
            tensor(&[0.0, 1.0]),
            tensor(&[1.0, 2.0]),
            Value::String("r--".to_string()),
            tensor(&[2.0, 3.0]),
            Value::String("LineWidth".to_string()),
            Value::Num(2.0),
        ])
        .unwrap();
        assert_eq!(plans.len(), 2);
        assert!(plans[0].line_style_explicit);
        assert!(plans[1].appearance.line_width > 1.0);
    }

    #[test]
    fn complex_input_maps_to_angle_and_radius() {
        let data = futures::executor::block_on(evaluate_polar_data(PolarSeriesInput::Complex(
            ComplexInput::Scalar(0.0, 2.0),
        )))
        .unwrap();
        assert_eq!(data.len(), 1);
        assert!((data[0].theta[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(data[0].rho[0], 2.0);
    }

    #[test]
    fn matrix_rho_expands_to_column_series() {
        let rho = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
            dtype: NumericDType::F64,
        };
        let data = futures::executor::block_on(evaluate_polar_data(PolarSeriesInput::Rho(
            Value::Tensor(rho),
        )))
        .unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].rho, vec![1.0, 2.0]);
        assert_eq!(data[1].rho, vec![3.0, 4.0]);
        assert_eq!(data[0].theta.len(), 2);
        assert_eq!(data[1].theta.len(), 2);
    }

    #[test]
    fn polarplot_sets_axes_metadata_and_adds_line() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();

        let handle = futures::executor::block_on(polarplot_builtin(vec![
            tensor(&[0.0, std::f64::consts::FRAC_PI_2]),
            tensor(&[1.0, 2.0]),
        ]))
        .unwrap();

        let figure = clone_figure(current_figure_handle()).unwrap();
        assert!(handle.is_finite());
        assert_eq!(figure.len(), 1);
        let meta = figure.axes_metadata(0).unwrap();
        assert_eq!(meta.axes_kind, AxesKind::Polar);
        assert!(meta.axis_equal);
        assert_eq!(meta.x_limits, Some((-2.0, 2.0)));
        assert_eq!(meta.y_limits, Some((-2.0, 2.0)));
    }

    #[test]
    fn polarplot_empty_input_creates_empty_line() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();

        let empty = Value::Tensor(Tensor {
            data: Vec::new(),
            shape: vec![0, 0],
            rows: 0,
            cols: 0,
            dtype: NumericDType::F64,
        });
        futures::executor::block_on(polarplot_builtin(vec![empty])).unwrap();

        let mut figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.len(), 1);
        let plot = figure.plots().next().unwrap();
        let runmat_plot::plots::PlotElement::Line(line) = plot else {
            panic!("expected line plot")
        };
        assert!(line.is_empty());
        let bounds = figure.bounds();
        assert!(bounds.min.x.is_infinite());
        assert!(bounds.max.x.is_infinite());
    }

    #[test]
    fn polarplot_matrix_columns_and_trailing_properties_apply_to_all_lines() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();

        let theta = tensor(&[0.0, std::f64::consts::FRAC_PI_2]);
        let rho = Value::Tensor(Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
            dtype: NumericDType::F64,
        });
        futures::executor::block_on(polarplot_builtin(vec![
            theta,
            rho,
            Value::String("LineWidth".to_string()),
            Value::Num(2.0),
        ]))
        .unwrap();

        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.len(), 2);
        for plot in figure.plots() {
            let runmat_plot::plots::PlotElement::Line(line) = plot else {
                panic!("expected line plot")
            };
            assert_eq!(line.len(), 2);
            assert_eq!(line.line_width, 2.0);
        }
    }

    #[test]
    fn cartesian_plot_after_polarplot_resets_axes_kind_and_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();

        futures::executor::block_on(polarplot_builtin(vec![
            tensor(&[0.0, std::f64::consts::FRAC_PI_2]),
            tensor(&[1.0, 2.0]),
        ]))
        .unwrap();
        futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![tensor(
            &[1.0, 2.0, 3.0],
        )]))
        .unwrap();

        let figure = clone_figure(current_figure_handle()).unwrap();
        let meta = figure.axes_metadata(0).unwrap();
        assert_eq!(meta.axes_kind, AxesKind::Cartesian);
        assert_eq!(meta.x_limits, None);
        assert_eq!(meta.y_limits, None);
    }
}
