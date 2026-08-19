//! Quantile-quantile plot compatibility helper.

use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use std::cmp::Ordering;

use glam::Vec4;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Type,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::line::LineMarkerAppearance;
use runmat_plot::plots::scatter::MarkerStyle;
use runmat_plot::plots::{LinePlot, LineStyle};
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_handle};
use crate::builtins::plotting::state::{
    register_line_handle, render_active_plot, PlotRenderOptions,
};
use crate::builtins::stats::summary::distribution_math::standard_normal_inv;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "qqplot";

const OUTPUT_HANDLES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line handles for the quantile points, quartile line, and fitted reference line.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample data vector or matrix.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional comparison sample data vector.",
};

const PARAM_PVEC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pvec",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional quantile percentages in the closed interval [0, 100].",
};

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

const INPUT_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUT_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUT_X_Y_P: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_PVEC];
const INPUT_AX_X: [BuiltinParamDescriptor; 2] = [PARAM_AX, PARAM_X];
const INPUT_AX_X_Y_P: [BuiltinParamDescriptor; 4] = [PARAM_AX, PARAM_X, PARAM_Y, PARAM_PVEC];
const OUTPUT_H: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLES];

const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "qqplot(x)",
        inputs: &INPUT_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "qqplot(x, y)",
        inputs: &INPUT_X_Y,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "qqplot(x, y, pvec)",
        inputs: &INPUT_X_Y_P,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "qqplot(ax, ___)",
        inputs: &INPUT_AX_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = qqplot(___)",
        inputs: &INPUT_X,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = qqplot(x, y, pvec)",
        inputs: &INPUT_X_Y_P,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = qqplot(ax, ___)",
        inputs: &INPUT_AX_X_Y_P,
        outputs: &OUTPUT_H,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QQPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:qqplot:InvalidArgument"),
    when: "Inputs, probabilities, or axes handles are malformed.",
    message: "qqplot: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QQPLOT.INTERNAL",
    identifier: Some("RunMat:qqplot:Internal"),
    when: "RunMat cannot construct quantile pairs or plotting primitives.",
    message: "qqplot: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const QQPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const QQPLOT_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qqplot-integer-sample",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qqplot accepts typed-integer sample data as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QqplotIntegerSampleExtension"),
};
const QQPLOT_INTEGER_PVEC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qqplot-integer-pvec",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qqplot accepts a typed-integer quantile vector as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QqplotIntegerPvecExtension"),
};
const QQPLOT_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qqplot-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qqplot accepts explicitly GPU-resident inputs as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QqplotExplicitGpuInputExtension"),
};
const QQPLOT_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qqplot-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qqplot accepts logical sample or percentage input as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QqplotLogicalInputExtension"),
};
pub const QQPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    QQPLOT_INTEGER_SAMPLE_EXTENSION,
    QQPLOT_INTEGER_PVEC_EXTENSION,
    QQPLOT_EXPLICIT_GPU_EXTENSION,
    QQPLOT_LOGICAL_INPUT_EXTENSION,
];
const QQPLOT_INTEGER_SAMPLES: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double sample data. RunMat admits typed integers only after exact binary64 representability is proved.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The optional second sample has the same independently gated checked floating boundary as x.",
    },
];
const QQPLOT_INTEGER_PVEC: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "pvec",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents single and double quantile percentages; typed integer percentages cross a checked binary64 normalization boundary.",
}];
pub const QQPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = qqplot(integer_x [, integer_y])",
        inputs: &QQPLOT_INTEGER_SAMPLES,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Quantiles, interpolation, and reference-line geometry are explicit floating boundaries. Automatic residency gathers transparently; explicit undocumented GPU intent is independently gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = qqplot(x,y,integer_pvec)",
        inputs: &QQPLOT_INTEGER_PVEC,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Typed quantile percentages are a separately gated RunMat extension and are normalized only after exactness and range checks.",
    },
];

fn qqplot_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Unknown
}

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INVALID_ARGUMENT, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INTERNAL, message)
}

#[derive(Clone, Debug)]
struct Series {
    theoretical: Vec<f64>,
    observed: Vec<f64>,
    reference_x: Vec<f64>,
    reference_y: Vec<f64>,
    quartile_x: Vec<f64>,
    quartile_y: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlotMode {
    Normal,
    TwoSample,
}

struct QqplotEvaluation {
    mode: PlotMode,
    series: Vec<Series>,
}

#[runtime_builtin(
    name = "qqplot",
    category = "stats/summary",
    summary = "Create a quantile-quantile plot against a normal distribution or another sample.",
    keywords = "qqplot,quantile,normal,probability,statistics,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(qqplot_type),
    descriptor(crate::builtins::stats::summary::qqplot::QQPLOT_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::qqplot::QQPLOT_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::summary::qqplot::QQPLOT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::qqplot"
)]
pub(crate) async fn qqplot_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (target, args) =
        split_leading_axes_handle(args, NAME).map_err(|err| invalid(format!("qqplot: {err}")))?;
    ensure_qqplot_boundaries(&args).await?;
    apply_axes_target(target, NAME).map_err(|err| invalid(err.message))?;

    let eval = parse_args(args).await?;
    if eval.series.is_empty() {
        return Err(invalid("qqplot: input contains no finite samples"));
    }

    let handles = render_series(eval)?;
    if handles.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new(Vec::new(), vec![0, 0])
                .map_err(|err| internal(format!("qqplot: {err}")))?,
        ));
    }
    Tensor::new(handles.clone(), vec![handles.len(), 1])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("qqplot: {err}")))
}

async fn ensure_qqplot_boundaries(args: &[Value]) -> BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64_async, value_has_logical_class,
        value_has_native_integer_class,
    };
    for (index, value) in args.iter().enumerate() {
        if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &QQPLOT_EXPLICIT_GPU_EXTENSION,
                NAME,
            )?;
        }
        if value_has_logical_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &QQPLOT_LOGICAL_INPUT_EXTENSION,
                NAME,
            )?;
        }
        if !value_has_native_integer_class(value) {
            continue;
        }
        let extension = if index == 2 {
            &QQPLOT_INTEGER_PVEC_EXTENSION
        } else {
            &QQPLOT_INTEGER_SAMPLE_EXTENSION
        };
        crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)?;
        if !native_integer_value_is_exact_f64_async(value).await? {
            let role = if index == 2 { "pvec" } else { "sample" };
            return Err(invalid(format!(
                "qqplot: integer {role} values must be exactly representable as double"
            )));
        }
    }
    Ok(())
}

async fn parse_args(args: Vec<Value>) -> BuiltinResult<QqplotEvaluation> {
    match args.len() {
        0 => Err(invalid("qqplot: sample input is required")),
        1 => {
            let x = value_to_tensor(args.into_iter().next().unwrap()).await?;
            Ok(QqplotEvaluation {
                mode: PlotMode::Normal,
                series: normal_series_from_tensor(x, None)?,
            })
        }
        2 => {
            let mut iter = args.into_iter();
            let x = value_to_tensor(iter.next().unwrap()).await?;
            let second = value_to_tensor(iter.next().unwrap()).await?;
            Ok(QqplotEvaluation {
                mode: PlotMode::TwoSample,
                series: two_sample_series_from_tensors(x, second, None)?,
            })
        }
        3 => {
            let mut iter = args.into_iter();
            let x = value_to_tensor(iter.next().unwrap()).await?;
            let y = value_to_tensor(iter.next().unwrap()).await?;
            let p = probabilities_from_tensor(value_to_tensor(iter.next().unwrap()).await?)?;
            if tensor::tensor_element_len(&y) == 0 {
                Ok(QqplotEvaluation {
                    mode: PlotMode::Normal,
                    series: normal_series_from_tensor(x, Some(p))?,
                })
            } else {
                Ok(QqplotEvaluation {
                    mode: PlotMode::TwoSample,
                    series: two_sample_series_from_tensors(x, y, Some(p))?,
                })
            }
        }
        _ => Err(invalid("qqplot: accepts at most x, y, and pvec inputs")),
    }
}

async fn value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("qqplot: {err}")))?;
    let tensor = tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| invalid(format!("qqplot: {err}")))?;
    if tensor.integer_storage().is_some() {
        return Tensor::new(tensor_values_f64(&tensor), tensor.shape.clone())
            .map_err(|err| internal(format!("qqplot: {err}")));
    }
    Ok(tensor)
}

fn normal_series_from_tensor(x: Tensor, pvec: Option<Vec<f64>>) -> BuiltinResult<Vec<Series>> {
    let shape = tensor::default_shape_for(&x.shape, tensor::tensor_element_len(&x));
    let columns = columns_from_tensor(x, &shape)?;
    columns
        .into_iter()
        .filter_map(|column| {
            let sorted = finite_sorted(column);
            if sorted.is_empty() {
                return None;
            }
            Some(normal_series(sorted, pvec.clone()))
        })
        .collect()
}

fn two_sample_series_from_tensors(
    x: Tensor,
    y: Tensor,
    pvec: Option<Vec<f64>>,
) -> BuiltinResult<Vec<Series>> {
    let x_shape = tensor::default_shape_for(&x.shape, tensor::tensor_element_len(&x));
    let y_shape = tensor::default_shape_for(&y.shape, tensor::tensor_element_len(&y));
    let x_columns = columns_from_tensor(x, &x_shape)?;
    let y_columns = columns_from_tensor(y, &y_shape)?;
    if x_columns.is_empty() || y_columns.is_empty() {
        return Err(invalid("qqplot: both samples must contain values"));
    }
    let pairs = column_pairs(&x_columns, &y_columns)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (x_col, y_col) in pairs {
        let x = finite_sorted(x_col);
        let y = finite_sorted(y_col);
        if x.is_empty() || y.is_empty() {
            continue;
        }
        let probabilities = pvec
            .clone()
            .unwrap_or_else(|| default_probabilities(x.len().min(y.len())));
        let theoretical = probabilities
            .iter()
            .map(|p| quantile_from_sorted(&x, *p))
            .collect::<Vec<_>>();
        let observed = probabilities
            .iter()
            .map(|p| quantile_from_sorted(&y, *p))
            .collect::<Vec<_>>();
        out.push(series_from_pairs(theoretical, observed));
    }
    if out.is_empty() {
        return Err(invalid("qqplot: both samples must contain finite values"));
    }
    Ok(out)
}

fn normal_series(sorted: Vec<f64>, pvec: Option<Vec<f64>>) -> BuiltinResult<Series> {
    let explicit_probabilities = pvec.is_some();
    let probabilities = pvec.unwrap_or_else(|| default_probabilities(sorted.len()));
    let theoretical = probabilities
        .iter()
        .map(|p| standard_normal_inv(*p))
        .collect::<Vec<_>>();
    let observed = if explicit_probabilities {
        probabilities
            .iter()
            .map(|p| quantile_from_sorted(&sorted, *p))
            .collect::<Vec<_>>()
    } else {
        sorted
    };
    Ok(series_from_pairs(theoretical, observed))
}

fn series_from_pairs(theoretical: Vec<f64>, observed: Vec<f64>) -> Series {
    let (reference_x, reference_y, quartile_x, quartile_y) =
        reference_lines(&theoretical, &observed);
    Series {
        theoretical,
        observed,
        reference_x,
        reference_y,
        quartile_x,
        quartile_y,
    }
}

fn columns_from_tensor(tensor: Tensor, shape: &[usize]) -> BuiltinResult<Vec<Vec<f64>>> {
    let data = tensor_values_f64(&tensor);
    if shape.is_empty() {
        return Ok(vec![data]);
    }
    if shape.iter().filter(|dim| **dim > 1).count() <= 1 {
        return Ok(vec![data]);
    }
    if shape.len() > 2 {
        return Err(invalid("qqplot: input must be a vector or 2-D matrix"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let mut out = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut values = Vec::with_capacity(rows);
        for row in 0..rows {
            values.push(data[row + col * rows]);
        }
        out.push(values);
    }
    Ok(out)
}

fn column_pairs(x: &[Vec<f64>], y: &[Vec<f64>]) -> BuiltinResult<Vec<(Vec<f64>, Vec<f64>)>> {
    if x.len() == y.len() {
        return Ok(x.iter().cloned().zip(y.iter().cloned()).collect());
    }
    if x.len() == 1 {
        return Ok(y
            .iter()
            .cloned()
            .map(|y_col| (x[0].clone(), y_col))
            .collect());
    }
    if y.len() == 1 {
        return Ok(x
            .iter()
            .cloned()
            .map(|x_col| (x_col, y[0].clone()))
            .collect());
    }
    Err(invalid(
        "qqplot: x and y matrices must have the same number of columns",
    ))
}

fn finite_sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.retain(|value| value.is_finite());
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values
}

fn probabilities_from_tensor(tensor: Tensor) -> BuiltinResult<Vec<f64>> {
    let data = tensor_values_f64(&tensor);
    if data.is_empty() {
        return Err(invalid("qqplot: pvec must not be empty"));
    }
    let mut probabilities = Vec::with_capacity(data.len());
    for value in data {
        if !value.is_finite() || !(0.0..=100.0).contains(&value) {
            return Err(invalid(
                "qqplot: pvec values must be finite percentages in the interval [0, 100]",
            ));
        }
        probabilities.push(value / 100.0);
    }
    probabilities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(probabilities)
}

fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}

fn default_probabilities(n: usize) -> Vec<f64> {
    (0..n).map(|idx| (idx as f64 + 0.5) / n as f64).collect()
}

fn quantile_from_sorted(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }
    let position = p * (values.len() - 1) as f64;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let weight = position - lo as f64;
        values[lo] * (1.0 - weight) + values[hi] * weight
    }
}

fn reference_lines(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut finite = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    finite.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let xs = finite.iter().map(|(x, _)| *x).collect::<Vec<_>>();
    let ys = finite.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    let x25 = quantile_from_sorted(&xs, 0.25);
    let x75 = quantile_from_sorted(&xs, 0.75);
    let y25 = quantile_from_sorted(&ys, 0.25);
    let y75 = quantile_from_sorted(&ys, 0.75);
    let slope = if (x75 - x25).abs() > f64::EPSILON {
        (y75 - y25) / (x75 - x25)
    } else {
        0.0
    };
    let intercept = y25 - slope * x25;
    let min_x = *xs.first().unwrap();
    let max_x = *xs.last().unwrap();
    let reference_x = if min_x < max_x {
        vec![min_x, max_x]
    } else {
        let pad = min_x.abs().max(1.0) * 0.5;
        vec![min_x - pad, max_x + pad]
    };
    let reference_y = reference_x
        .iter()
        .map(|x| slope * *x + intercept)
        .collect::<Vec<_>>();
    let quartile_x = vec![x25, x75];
    let quartile_y = vec![y25, y75];
    (reference_x, reference_y, quartile_x, quartile_y)
}

fn render_series(eval: QqplotEvaluation) -> BuiltinResult<Vec<f64>> {
    let mut plots = Vec::new();
    for (idx, series) in eval.series.iter().enumerate() {
        plots.push(data_plot(series, idx)?);
        if !series.quartile_x.is_empty() {
            plots.push(reference_plot(series, idx, true)?);
        }
        if !series.reference_x.is_empty() {
            plots.push(reference_plot(series, idx, false)?);
        }
    }

    let mut plots_opt = Some(plots);
    let plot_indices_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        NAME,
        PlotRenderOptions {
            title: "Q-Q Plot",
            x_label: match eval.mode {
                PlotMode::Normal => "Standard Normal Quantiles",
                PlotMode::TwoSample => "Quantiles of X",
            },
            y_label: match eval.mode {
                PlotMode::Normal => "Quantiles of Input Sample",
                PlotMode::TwoSample => "Quantiles of Y",
            },
            ..Default::default()
        },
        move |figure, axes_index| {
            let plots = plots_opt
                .take()
                .expect("qqplot series consumed exactly once");
            for plot in plots {
                let plot_index = figure.add_line_plot_on_axes(plot, axes_index);
                plot_indices_slot
                    .borrow_mut()
                    .push((axes_index, plot_index));
            }
            Ok(())
        },
    );

    let handles = plot_indices_out
        .borrow()
        .iter()
        .map(|(axes_index, plot_index)| {
            register_line_handle(figure_handle, *axes_index, *plot_index)
        })
        .collect::<Vec<_>>();
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handles);
        }
        return Err(internal(err.message));
    }
    Ok(handles)
}

fn data_plot(series: &Series, idx: usize) -> BuiltinResult<LinePlot> {
    let color = palette(idx);
    let mut plot = LinePlot::new(series.theoretical.clone(), series.observed.clone())
        .map_err(|err| internal(format!("qqplot: {err}")))?
        .with_style(color, 1.0, LineStyle::None)
        .with_label(format!("Data {}", idx + 1));
    plot.set_marker(Some(LineMarkerAppearance {
        kind: MarkerStyle::Plus,
        size: 7.0,
        edge_color: color,
        face_color: color,
        filled: false,
    }));
    Ok(plot)
}

fn reference_plot(series: &Series, idx: usize, quartile: bool) -> BuiltinResult<LinePlot> {
    let (x, y, label, style) = if quartile {
        (
            series.quartile_x.clone(),
            series.quartile_y.clone(),
            format!("Quartile {}", idx + 1),
            LineStyle::Solid,
        )
    } else {
        (
            series.reference_x.clone(),
            series.reference_y.clone(),
            format!("Reference {}", idx + 1),
            LineStyle::Dashed,
        )
    };
    LinePlot::new(x, y)
        .map_err(|err| internal(format!("qqplot: {err}")))
        .map(|plot| plot.with_style(palette(idx), 1.0, style).with_label(label))
}

fn palette(idx: usize) -> Vec4 {
    const COLORS: [Vec4; 6] = [
        Vec4::new(0.000, 0.447, 0.741, 1.0),
        Vec4::new(0.850, 0.325, 0.098, 1.0),
        Vec4::new(0.929, 0.694, 0.125, 1.0),
        Vec4::new(0.494, 0.184, 0.556, 1.0),
        Vec4::new(0.466, 0.674, 0.188, 1.0),
        Vec4::new(0.301, 0.745, 0.933, 1.0),
    ];
    COLORS[idx % COLORS.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage};

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new_integer(storage, vec![rows, cols]).unwrap())
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn qqplot_numeric_helpers_read_typed_integer_storage_exactly() {
        let wide = u64::MAX - 1;
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![wide, wide - 1]), vec![2, 1]).unwrap();
        assert_eq!(
            tensor_values_f64(&tensor),
            vec![
                IntValue::U64(wide).to_f64(),
                IntValue::U64(wide - 1).to_f64()
            ]
        );
        assert_eq!(
            columns_from_tensor(tensor, &[2, 1]).unwrap(),
            vec![vec![
                IntValue::U64(wide).to_f64(),
                IntValue::U64(wide - 1).to_f64()
            ]]
        );
        let pvec = Tensor::new_integer(IntegerStorage::U8(vec![75, 25]), vec![1, 2]).unwrap();
        assert_eq!(probabilities_from_tensor(pvec).unwrap(), vec![0.25, 0.75]);
    }

    #[test]
    fn qqplot_scalar_typed_integer_sample_uses_storage_len_for_default_shape() {
        let input = Tensor::new_integer(IntegerStorage::I16(vec![42]), Vec::new()).unwrap();

        let series = normal_series_from_tensor(input, None).unwrap();

        assert_eq!(series.len(), 1);
        assert_eq!(series[0].observed, vec![42.0]);
    }

    #[test]
    fn qqplot_normal_vector_returns_data_reference_and_quartile_handles() {
        let _guard = setup();
        let out = block_on(qqplot_builtin(vec![tensor(vec![1.0, 2.0, 4.0, 8.0], 4, 1)])).unwrap();
        let handles = tensor_data(out);
        assert_eq!(handles.len(), 3);

        let x = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("XData".into())]).unwrap(),
        );
        let y = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
        );
        assert_eq!(x.len(), 4);
        assert_eq!(y, vec![1.0, 2.0, 4.0, 8.0]);
        assert!(x[0] < x[1] && x[2] < x[3]);
        assert_eq!(
            get_builtin(vec![Value::Num(handles[0]), Value::String("Marker".into())]).unwrap(),
            Value::String("+".into())
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(handles[1]),
                Value::String("LineStyle".into())
            ])
            .unwrap(),
            Value::String("-".into())
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(handles[2]),
                Value::String("LineStyle".into())
            ])
            .unwrap(),
            Value::String("--".into())
        );
    }

    #[test]
    fn qqplot_percentage_vector_interpolates_sample_quantiles() {
        let _guard = setup();
        let out = block_on(qqplot_builtin(vec![
            tensor(vec![0.0, 10.0, 20.0], 3, 1),
            tensor(Vec::new(), 0, 0),
            tensor(vec![25.0, 50.0, 75.0], 1, 3),
        ]))
        .unwrap();
        let handles = tensor_data(out);
        let y = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
        );
        assert_eq!(y, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn qqplot_accepts_typed_integer_samples_and_pvec() {
        let _guard = setup();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let out = block_on(qqplot_builtin(vec![
            int_tensor(IntegerStorage::I16(vec![0, 10, 20]), 3, 1),
            int_tensor(IntegerStorage::I16(Vec::new()), 0, 0),
            int_tensor(IntegerStorage::U8(vec![25, 50, 75]), 1, 3),
        ]))
        .unwrap();
        let handles = tensor_data(out);
        let y = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
        );
        assert_eq!(y, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn qqplot_integer_extensions_are_gated_and_inexact_values_reject() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(qqplot_builtin(vec![int_tensor(
            IntegerStorage::I16(vec![1, 2]),
            2,
            1,
        )]))
        .expect_err("strict mode rejects integer samples");
        assert_eq!(
            error.identifier(),
            QQPLOT_INTEGER_SAMPLE_EXTENSION.error_identifier
        );
        drop(strict);

        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(qqplot_builtin(vec![
            tensor(vec![1.0, 2.0], 2, 1),
            tensor(Vec::new(), 0, 0),
            int_tensor(IntegerStorage::U8(vec![25, 75]), 1, 2),
        ]))
        .expect_err("strict mode rejects integer pvec");
        assert_eq!(
            error.identifier(),
            QQPLOT_INTEGER_PVEC_EXTENSION.error_identifier
        );
        drop(strict);

        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = block_on(qqplot_builtin(vec![int_tensor(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, 2]),
            2,
            1,
        )]))
        .expect_err("inexact integer sample rejects");
        assert!(error.message().contains("exactly representable as double"));
        drop(extensions);
    }

    #[test]
    fn qqplot_strict_mode_gates_explicit_gpu_before_provider_access() {
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 444,
            descriptor: Default::default(),
        };
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(qqplot_builtin(vec![Value::GpuTensor(handle)]))
            .expect_err("strict mode rejects explicit GPU input before gather");
        assert_eq!(
            error.identifier(),
            QQPLOT_EXPLICIT_GPU_EXTENSION.error_identifier
        );
    }

    #[test]
    fn qqplot_logical_input_is_a_gated_extension() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(qqplot_builtin(vec![Value::Bool(true)]))
            .expect_err("strict mode rejects logical sample extension");
        assert_eq!(
            error.identifier(),
            QQPLOT_LOGICAL_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn qqplot_two_sample_uses_common_probabilities() {
        let _guard = setup();
        let out = block_on(qqplot_builtin(vec![
            tensor(vec![1.0, 3.0, 5.0], 3, 1),
            tensor(vec![2.0, 4.0, 6.0], 3, 1),
            tensor(vec![50.0], 1, 1),
        ]))
        .unwrap();
        let handles = tensor_data(out);
        let x = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("XData".into())]).unwrap(),
        );
        let y = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
        );
        assert_eq!(x, vec![3.0]);
        assert_eq!(y, vec![4.0]);
    }

    #[test]
    fn qqplot_two_sample_matrices_pair_columns_and_do_not_treat_y_as_pvec() {
        let _guard = setup();
        let out = block_on(qqplot_builtin(vec![
            tensor(vec![0.1, 0.2, 1.0, 2.0], 2, 2),
            tensor(vec![0.3, 0.4, 3.0, 4.0], 2, 2),
            tensor(vec![50.0], 1, 1),
        ]))
        .unwrap();
        let handles = tensor_data(out);
        assert_eq!(handles.len(), 6);
        let x1 = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("XData".into())]).unwrap(),
        );
        let y1 = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
        );
        let x2 = tensor_data(
            get_builtin(vec![Value::Num(handles[3]), Value::String("XData".into())]).unwrap(),
        );
        let y2 = tensor_data(
            get_builtin(vec![Value::Num(handles[3]), Value::String("YData".into())]).unwrap(),
        );
        assert_eq!(x1, vec![0.15000000000000002]);
        assert_eq!(y1, vec![0.35]);
        assert_eq!(x2, vec![1.5]);
        assert_eq!(y2, vec![3.5]);
    }

    #[test]
    fn qqplot_matrix_creates_one_series_per_column_and_omits_nan() {
        let _guard = setup();
        let out = block_on(qqplot_builtin(vec![tensor(
            vec![1.0, 2.0, f64::NAN, 4.0, 10.0, 20.0],
            3,
            2,
        )]))
        .unwrap();
        let handles = tensor_data(out);
        assert_eq!(handles.len(), 6);
        let y = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
        );
        assert_eq!(y, vec![1.0, 2.0]);
    }

    #[test]
    fn qqplot_rejects_bad_inputs() {
        let err = block_on(qqplot_builtin(vec![
            tensor(vec![1.0, 2.0], 2, 1),
            tensor(vec![3.0, 4.0], 2, 1),
            tensor(vec![-1.0], 1, 1),
        ]))
        .unwrap_err();
        assert!(err.message.contains("pvec"));
    }
}
