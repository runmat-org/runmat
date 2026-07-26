//! Reference-line compatibility helper.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{LinePlot, LineStyle, PlotElement};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::apply_axes_target;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    append_active_plot, axes_metadata_snapshot, clone_figure, current_axes_state,
    register_line_handle, FigureHandle, PlotRenderOptions,
};
use crate::builtins::plotting::style::{
    marker_metadata_from_appearance, parse_line_style_args, value_as_f64, LineStyleParseOptions,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "refline";

const OUTPUT_HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line graphics handle.",
};

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

const PARAM_COEFFS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "coeffs",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element vector [slope intercept].",
};

const PARAM_SLOPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line slope.",
};

const PARAM_INTERCEPT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line y-intercept.",
};

const PARAM_STYLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "style",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional LineSpec or line Name/Value pairs.",
};

const INPUTS_EMPTY: [BuiltinParamDescriptor; 0] = [];
const INPUTS_COEFFS: [BuiltinParamDescriptor; 1] = [PARAM_COEFFS];
const INPUTS_SLOPE_INTERCEPT: [BuiltinParamDescriptor; 2] = [PARAM_SLOPE, PARAM_INTERCEPT];
const INPUTS_AX_STYLE: [BuiltinParamDescriptor; 2] = [PARAM_AX, PARAM_STYLE];
const INPUTS_COEFFS_STYLE: [BuiltinParamDescriptor; 2] = [PARAM_COEFFS, PARAM_STYLE];
const INPUTS_SLOPE_INTERCEPT_STYLE: [BuiltinParamDescriptor; 3] =
    [PARAM_SLOPE, PARAM_INTERCEPT, PARAM_STYLE];
const OUTPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLE];

const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "refline()",
        inputs: &INPUTS_EMPTY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "refline(coeffs)",
        inputs: &INPUTS_COEFFS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "refline(m, b)",
        inputs: &INPUTS_SLOPE_INTERCEPT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "refline(___, LineSpec, Name, Value)",
        inputs: &INPUTS_COEFFS_STYLE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "refline(ax, ___)",
        inputs: &INPUTS_AX_STYLE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = refline(___)",
        inputs: &INPUTS_SLOPE_INTERCEPT_STYLE,
        outputs: &OUTPUTS_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = refline(ax, ___)",
        inputs: &INPUTS_AX_STYLE,
        outputs: &OUTPUTS_HANDLE,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REFLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:refline:InvalidArgument"),
    when: "Axes handle, coefficient inputs, or style options are malformed.",
    message: "refline: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REFLINE.INTERNAL",
    identifier: Some("RunMat:refline:Internal"),
    when: "RunMat cannot construct or register the reference line.",
    message: "refline: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const REFLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn refline_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Unknown
}

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INVALID_ARGUMENT, message)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "refline",
    category = "stats/summary",
    summary = "Add a reference line y = m*x + b to the current or specified axes.",
    keywords = "refline,reference line,statistics,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(refline_type),
    descriptor(crate::builtins::stats::summary::refline::REFLINE_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::refline"
)]
pub(crate) async fn refline_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes(args)?;
    apply_axes_target(target, NAME).map_err(|err| {
        if err.identifier().is_some() {
            err
        } else {
            invalid_argument(err.message)
        }
    })?;

    let (plan, style_args) = parse_refline_coefficients(&args)?;
    let style = parse_line_style_args(style_args, &LineStyleParseOptions::generic(NAME))
        .map_err(|err| invalid_argument(err.message))?;
    let axes = current_axes_state();
    let figure_handle = axes.handle;

    let specs = match plan {
        ReflinePlan::Explicit { slope, intercept } => vec![ReflineLineSpec {
            slope,
            intercept,
            x_span: refline_x_span(figure_handle, axes.active_index)?,
        }],
        ReflinePlan::LeastSquares => least_squares_specs(figure_handle, axes.active_index).await?,
    };
    if specs.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new(Vec::new(), vec![0, 0])
                .map_err(|err| internal_error(format!("refline: {err}")))?,
        ));
    }

    let lines = specs
        .iter()
        .map(|spec| line_from_spec(*spec, &style))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let mut lines = Some(lines);

    let plot_indices_slot = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_out = std::rc::Rc::clone(&plot_indices_slot);
    let render_result =
        append_active_plot(NAME, PlotRenderOptions::default(), move |figure, axes| {
            let lines = lines
                .take()
                .ok_or_else(|| internal_error("refline: lines already rendered"))?;
            let mut indices = plot_indices_out.borrow_mut();
            for line in lines {
                let index = figure.add_line_plot_on_axes(line, axes);
                indices.push((axes, index));
            }
            Ok(())
        });

    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if !(lower.contains("plotting is unavailable") || lower.contains("non-main thread")) {
            return Err(err);
        }
    }

    let handles = plot_indices_slot
        .borrow()
        .iter()
        .map(|(axes_index, plot_index)| {
            register_line_handle(figure_handle, *axes_index, *plot_index)
        })
        .collect::<Vec<_>>();
    if handles.is_empty() {
        return Err(internal_error("refline: line was not registered"));
    }
    if handles.len() == 1 {
        return Ok(Value::Num(handles[0]));
    }

    let len = handles.len();
    Ok(Value::Tensor(Tensor::new(handles, vec![len, 1]).map_err(
        |err| internal_error(format!("refline: {err}")),
    )?))
}

type AxesTarget = Option<(FigureHandle, usize)>;

#[derive(Clone, Copy)]
enum ReflinePlan {
    Explicit { slope: f64, intercept: f64 },
    LeastSquares,
}

#[derive(Clone, Copy)]
struct ReflineLineSpec {
    slope: f64,
    intercept: f64,
    x_span: (f64, f64),
}

fn split_optional_axes(args: Vec<Value>) -> BuiltinResult<(AxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, NAME) {
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

fn parse_refline_coefficients(args: &[Value]) -> BuiltinResult<(ReflinePlan, &[Value])> {
    match args {
        [] => Ok((ReflinePlan::LeastSquares, &[])),
        [coeffs, rest @ ..] => {
            if let Some(pair) = coefficient_pair(coeffs)? {
                return Ok((
                    ReflinePlan::Explicit {
                        slope: pair.0,
                        intercept: pair.1,
                    },
                    rest,
                ));
            }
            if rest.is_empty() {
                return Err(invalid_argument(
                    "refline: expected coefficients as [slope intercept] or slope, intercept",
                ));
            }
            let slope = finite_scalar(coeffs, "slope")?;
            let intercept = finite_scalar(&rest[0], "intercept")?;
            Ok((ReflinePlan::Explicit { slope, intercept }, &rest[1..]))
        }
    }
}

fn coefficient_pair(value: &Value) -> BuiltinResult<Option<(f64, f64)>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.data.len() != 2 {
                return Ok(None);
            }
            let slope = tensor_utils::tensor_value_f64(tensor, 0);
            let intercept = tensor_utils::tensor_value_f64(tensor, 1);
            if !slope.is_finite() || !intercept.is_finite() {
                return Err(invalid_argument(
                    "refline: coefficients must contain finite slope and intercept",
                ));
            }
            Ok(Some((slope, intercept)))
        }
        _ => Ok(None),
    }
}

fn finite_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    let value = value_as_f64(value)
        .ok_or_else(|| invalid_argument(format!("refline: {name} must be numeric")))?;
    if !value.is_finite() {
        return Err(invalid_argument(format!("refline: {name} must be finite")));
    }
    Ok(value)
}

fn refline_x_span(handle: FigureHandle, axes_index: usize) -> BuiltinResult<(f64, f64)> {
    if let Some(limits) = axes_metadata_snapshot(handle, axes_index)
        .map_err(|err| internal_error(format!("refline: {err}")))?
        .x_limits
        .filter(|limits| valid_span(*limits))
    {
        return Ok(limits);
    }

    if let Some(mut figure) = clone_figure(handle) {
        let bounds = figure.data_bounds_for_axes(axes_index);
        let span = finite_or_padded_span(bounds.min.x as f64, bounds.max.x as f64);
        if let Some(span) = span {
            return Ok(span);
        }
    }

    Ok((0.0, 1.0))
}

async fn least_squares_specs(
    handle: FigureHandle,
    axes_index: usize,
) -> BuiltinResult<Vec<ReflineLineSpec>> {
    let x_span = refline_x_span(handle, axes_index)?;
    let Some(figure) = clone_figure(handle) else {
        return Ok(Vec::new());
    };

    let mut specs = Vec::new();
    for (plot_index, plot) in figure.plots().enumerate() {
        if figure
            .plot_axes_indices()
            .get(plot_index)
            .copied()
            .unwrap_or(0)
            != axes_index
        {
            continue;
        }
        if !plot.is_visible() {
            continue;
        }

        let data = match plot {
            PlotElement::Scatter(plot) => {
                Some(plot.export_scene_xy_data().await.map_err(|err| {
                    internal_error(format!("refline: unable to read scatter data: {err}"))
                })?)
            }
            PlotElement::Line(plot)
                if plot.marker.is_some() && matches!(plot.line_style, LineStyle::None) =>
            {
                Some(plot.export_scene_xy_data().await.map_err(|err| {
                    internal_error(format!("refline: unable to read line data: {err}"))
                })?)
            }
            _ => None,
        };
        let Some((x, y)) = data else {
            continue;
        };
        if let Some((slope, intercept)) = least_squares_coefficients(&x, &y)? {
            specs.push(ReflineLineSpec {
                slope,
                intercept,
                x_span,
            });
        }
    }

    Ok(specs)
}

fn least_squares_coefficients(x: &[f64], y: &[f64]) -> BuiltinResult<Option<(f64, f64)>> {
    let mut n = 0usize;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for (&x, &y) in x.iter().zip(y.iter()) {
        if x.is_finite() && y.is_finite() {
            n += 1;
            sum_x += x;
            sum_y += y;
        }
    }
    if n == 0 {
        return Ok(None);
    }

    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / n as f64;
    let mut ss_xx = 0.0;
    let mut ss_xy = 0.0;
    for (&x, &y) in x.iter().zip(y.iter()) {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let dx = x - mean_x;
        ss_xx += dx * dx;
        ss_xy += dx * (y - mean_y);
    }

    let slope = if ss_xx > f64::EPSILON {
        ss_xy / ss_xx
    } else {
        0.0
    };
    let intercept = mean_y - slope * mean_x;
    if !slope.is_finite() || !intercept.is_finite() {
        return Err(invalid_argument(
            "refline: least-squares coefficients must be finite",
        ));
    }
    Ok(Some((slope, intercept)))
}

fn line_from_spec(
    spec: ReflineLineSpec,
    style: &crate::builtins::plotting::style::ParsedLineStyle,
) -> BuiltinResult<LinePlot> {
    let y_span = (
        spec.slope * spec.x_span.0 + spec.intercept,
        spec.slope * spec.x_span.1 + spec.intercept,
    );
    if !y_span.0.is_finite() || !y_span.1.is_finite() {
        return Err(invalid_argument(
            "refline: computed line coordinates must be finite",
        ));
    }

    let mut line = LinePlot::new(vec![spec.x_span.0, spec.x_span.1], vec![y_span.0, y_span.1])
        .map_err(|err| internal_error(format!("refline: {err}")))?
        .with_style(
            style.appearance.color,
            style.appearance.line_width,
            style.appearance.line_style,
        );
    line.set_marker(marker_metadata_from_appearance(&style.appearance));
    if let Some(label) = style.label.clone() {
        line = line.with_label(label);
    }
    Ok(line)
}

fn finite_or_padded_span(min: f64, max: f64) -> Option<(f64, f64)> {
    if !min.is_finite() || !max.is_finite() {
        return None;
    }
    if min < max {
        return Some((min, max));
    }
    if min == max {
        let pad = min.abs().max(1.0) * 0.5;
        return Some((min - pad, max + pad));
    }
    None
}

fn valid_span((min, max): (f64, f64)) -> bool {
    min.is_finite() && max.is_finite() && min < max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::scatter::scatter_builtin;
    use crate::builtins::plotting::state::{encode_axes_handle, PlotTestLockGuard};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, Tensor};

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
        let mut tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
        tensor.data.fill(f64::NAN);
        Value::Tensor(tensor)
    }

    fn x_data(handle: f64) -> Vec<f64> {
        let value = get_builtin(vec![Value::Num(handle), Value::String("XData".into())]).unwrap();
        Tensor::try_from(&value).unwrap().data
    }

    fn y_data(handle: f64) -> Vec<f64> {
        let value = get_builtin(vec![Value::Num(handle), Value::String("YData".into())]).unwrap();
        Tensor::try_from(&value).unwrap().data
    }

    #[test]
    fn refline_without_coefficients_adds_least_squares_line() {
        let _guard = setup();
        block_on(scatter_builtin(
            tensor(vec![1.0, 2.0, 3.0], 1, 3),
            tensor(vec![2.0, 4.0, 6.0], 1, 3),
            Vec::new(),
        ))
        .unwrap();
        let handle = block_on(refline_builtin(Vec::new())).unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(x_data(handle), vec![1.0, 3.0]);
        assert_eq!(y_data(handle), vec![2.0, 6.0]);
    }

    #[test]
    fn refline_accepts_coeff_vector_and_slope_intercept_forms() {
        let _guard = setup();
        let handle = block_on(refline_builtin(vec![tensor(vec![2.0, -1.0], 1, 2)])).unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(y_data(handle), vec![-1.0, 1.0]);

        let handle = block_on(refline_builtin(vec![Value::Num(-0.5), Value::Num(3.0)])).unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(y_data(handle), vec![3.0, 2.5]);
    }

    #[test]
    fn refline_reads_typed_integer_coefficients_exactly() {
        let _guard = setup();
        let handle = block_on(refline_builtin(vec![int_tensor(
            IntegerStorage::I16(vec![2, -1]),
            1,
            2,
        )]))
        .unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(y_data(handle), vec![-1.0, 1.0]);
    }

    #[test]
    fn refline_spans_existing_plot_or_explicit_axes_limits() {
        let _guard = setup();
        let _ = block_on(plot_builtin(vec![
            tensor(vec![2.0, 4.0], 1, 2),
            tensor(vec![10.0, 20.0], 1, 2),
        ]))
        .unwrap();
        let handle = block_on(refline_builtin(vec![Value::Num(1.0), Value::Num(0.0)])).unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(x_data(handle), vec![2.0, 4.0]);
        assert_eq!(y_data(handle), vec![2.0, 4.0]);

        let _ = crate::builtins::plotting::xlim::xlim_builtin(vec![tensor(vec![-1.0, 3.0], 1, 2)])
            .unwrap();
        let handle = block_on(refline_builtin(vec![Value::Num(2.0), Value::Num(1.0)])).unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(x_data(handle), vec![-1.0, 3.0]);
        assert_eq!(y_data(handle), vec![-1.0, 7.0]);
    }

    #[test]
    fn refline_targets_axes_and_accepts_style_args() {
        let _guard = setup();
        configure_subplot(1, 2, 0).unwrap();
        let _ = block_on(plot_builtin(vec![
            tensor(vec![100.0, 200.0], 1, 2),
            tensor(vec![10.0, 20.0], 1, 2),
        ]))
        .unwrap();
        configure_subplot(1, 2, 1).unwrap();
        let _ = block_on(plot_builtin(vec![
            tensor(vec![2.0, 4.0], 1, 2),
            tensor(vec![10.0, 20.0], 1, 2),
        ]))
        .unwrap();
        let fig = current_figure_handle();
        let ax = encode_axes_handle(fig, 1);
        let handle = block_on(refline_builtin(vec![
            Value::Num(ax),
            Value::Num(0.0),
            Value::Num(2.0),
            Value::String("--r".into()),
            Value::String("DisplayName".into()),
            Value::String("threshold".into()),
        ]))
        .unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(x_data(handle), vec![2.0, 4.0]);
        let style =
            get_builtin(vec![Value::Num(handle), Value::String("LineStyle".into())]).unwrap();
        assert_eq!(style, Value::String("--".into()));
        let name = get_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayName".into()),
        ])
        .unwrap();
        assert_eq!(name, Value::String("threshold".into()));
        let figure = clone_figure(fig).unwrap();
        assert_eq!(figure.len(), 3);
    }

    #[test]
    fn refline_rejects_bad_coefficients() {
        let _guard = setup();
        let err = block_on(refline_builtin(vec![tensor(vec![1.0, 2.0, 3.0], 1, 3)])).unwrap_err();
        assert!(err.message.contains("coefficients"));

        let err =
            block_on(refline_builtin(vec![Value::Num(f64::NAN), Value::Num(0.0)])).unwrap_err();
        assert!(err.message.contains("finite"));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn refline_least_squares_reads_gpu_scatter_source_data() {
        use runmat_accelerate_api::AccelProvider;
        use runmat_plot::core::{BoundingBox, GpuVertexBuffer};
        use runmat_plot::gpu::scatter2::Scatter2GpuInputs;
        use runmat_plot::gpu::ScalarType;
        use runmat_plot::plots::scatter::ScatterGpuStyle;
        use runmat_plot::plots::ScatterPlot;

        let _guard = setup();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            tracing::warn!("Skipping refline GPU scatter regression: no WGPU provider");
            return;
        };
        let context = crate::builtins::plotting::context::ensure_context_from_provider()
            .expect("shared plotting context");

        let x = block_on(crate::call_builtin_async(
            "gpuArray",
            &[tensor(vec![1.0, 2.0, 3.0], 1, 3)],
        ))
        .expect("gpu x");
        let y = block_on(crate::call_builtin_async(
            "gpuArray",
            &[tensor(vec![2.0, 4.0, 6.0], 1, 3)],
        ))
        .expect("gpu y");
        let Value::GpuTensor(x_handle) = x.clone() else {
            panic!("expected gpu x");
        };
        let Value::GpuTensor(y_handle) = y.clone() else {
            panic!("expected gpu y");
        };
        let x_ref = runmat_accelerate_api::export_wgpu_buffer(&x_handle).expect("export x");
        let y_ref = runmat_accelerate_api::export_wgpu_buffer(&y_handle).expect("export y");
        let dummy_vertices = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("refline-lsline-gpu-scatter-test-dummy-vertices"),
            size: 16,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let scatter = ScatterPlot::from_gpu_buffer(
            GpuVertexBuffer::new(std::sync::Arc::new(dummy_vertices), 0),
            0,
            BoundingBox::new(
                glam::Vec3::new(1.0, 2.0, 0.0),
                glam::Vec3::new(3.0, 6.0, 0.0),
            ),
            ScatterGpuStyle {
                color: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
                edge_color: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
                edge_thickness: 1.0,
                marker_size: 12.0,
                marker_style: runmat_plot::plots::MarkerStyle::Circle,
                filled: false,
                has_per_point_sizes: false,
                has_per_point_colors: false,
                edge_from_vertex_colors: false,
            },
        )
        .with_gpu_source_inputs(Scatter2GpuInputs {
            x_buffer: x_ref.buffer.clone(),
            y_buffer: y_ref.buffer.clone(),
            len: x_ref.len as u32,
            scalar: ScalarType::from_is_f64(
                x_ref.precision == runmat_accelerate_api::ProviderPrecision::F64,
            ),
        });
        let mut scatter = Some(scatter);
        append_active_plot(NAME, PlotRenderOptions::default(), move |figure, axes| {
            let scatter = scatter
                .take()
                .ok_or_else(|| internal_error("refline: test scatter already inserted"))?;
            figure.add_scatter_plot_on_axes(scatter, axes);
            Ok(())
        })
        .expect("insert GPU scatter plot");

        let handle = block_on(refline_builtin(Vec::new())).expect("gpu-backed refline");
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(x_data(handle), vec![1.0, 3.0]);
        assert_eq!(y_data(handle), vec![2.0, 6.0]);

        if let Value::GpuTensor(handle) = x {
            provider.free(&handle).ok();
        }
        if let Value::GpuTensor(handle) = y {
            provider.free(&handle).ok();
        }
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }
}
