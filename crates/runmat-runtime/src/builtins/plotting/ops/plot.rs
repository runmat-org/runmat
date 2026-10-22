//! MATLAB-compatible `plot` builtin.

use log::trace;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_plot::plots::NumericPlotData;
use runmat_plot::plots::{LinePlot, LineStyle};
use runmat_value::{Tensor, Value};

#[path = "plot/matrix_series.rs"]
mod matrix_series;
use matrix_series::split_matrix_series;

use super::common::numeric_plot_data_pair;
use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{continue_line_color_cycle_after, PlotRenderOptions, PreparedPlotTarget};
use super::style::{
    looks_like_plot_option_name, marker_metadata_from_appearance, parse_line_style_args,
    value_as_string, LineAppearance, LineStyleParseOptions, MarkerAppearance, MarkerColor,
    MarkerKind, DEFAULT_LINE_MARKER_SIZE,
};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "plot";

pub use runmat_builtins::{PLOT_DESCRIPTOR, PLOT_ERROR_INTERNAL, PLOT_ERROR_INVALID_ARGUMENT};

fn plot_error_with_detail(
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

fn map_plot_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    plot_error_with_detail(&PLOT_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_plot_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    crate::builtins::plotting::state::with_render_cause(
        plot_error_with_detail(&PLOT_ERROR_INTERNAL, err.message()),
        err,
    )
}

#[runtime_builtin(
    name = "plot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::plot"
)]
pub async fn plot_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    plot_builtin_with_runtime(runtime.as_ref(), args).await
}

pub async fn plot_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    plot_builtin_with_runtime(None, args).await
}

pub(super) async fn plot_builtin_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &PLOT_DESCRIPTOR,
        &runmat_builtins::PLOT_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    let (prepared, _) =
        super::op_common::axes_target::prepare_axes_call(runtime, BUILTIN_NAME, args.view())?;
    let explicit = prepared.is_explicit();
    let args = if explicit {
        args.into_range(1..)?
    } else {
        args
    };
    if explicit && args.is_empty() {
        return Err(plot_error_with_detail(
            &PLOT_ERROR_INVALID_ARGUMENT,
            "expected data after axes handle",
        ));
    }
    let (mut series_plans, line_style_order) =
        parse_series_specs(args, runtime).map_err(map_plot_invalid_argument)?;
    let target = PreparedPlotTarget::prepare(prepared.resolve(BUILTIN_NAME)?, explicit)
        .map_err(|source| super::properties::map_figure_error(BUILTIN_NAME, source))?;
    let hold_enabled = target
        .hold_enabled()
        .map_err(|source| super::properties::map_figure_error(BUILTIN_NAME, source))?;
    if let Some(order) = line_style_order.as_ref() {
        apply_line_style_order(&mut series_plans, order);
        target
            .set_style_order(order)
            .map_err(|source| super::properties::map_figure_error(BUILTIN_NAME, source))?;
    }
    let opts = PlotRenderOptions::default();
    let total = series_plans.len();
    let mut plots: Vec<LinePlot> = Vec::with_capacity(series_plans.len());
    for (series_idx, mut plan) in series_plans.drain(..).enumerate() {
        let (line_style, color) = target
            .series_style(
                series_idx,
                hold_enabled,
                !plan.line_style_explicit,
                !plan.color_explicit,
            )
            .map_err(|source| super::properties::map_figure_error(BUILTIN_NAME, source))?;
        if let Some(line_style) = line_style {
            plan.appearance.line_style = line_style;
            plan.line_style_explicit = true;
        }
        if let Some(color) = color {
            plan.appearance.color = color;
        }
        let inferred_label = plan
            .label
            .take()
            .or_else(|| plan.source_y_arg_index.and_then(crate::callsite::arg_text));
        let label = inferred_label.unwrap_or_else(|| {
            if total == 1 {
                "Data".to_string()
            } else {
                format!("Series {}", series_idx + 1)
            }
        });
        let SeriesRenderPlan {
            data,
            appearance,
            requires_cpu,
            ..
        } = plan;

        let x_kind = match &data.x {
            NumericInput::Host(_) => "Host",
            NumericInput::Gpu(_) => "Gpu",
        };
        let y_kind = match &data.y {
            NumericInput::Host(_) => "Host",
            NumericInput::Gpu(_) => "Gpu",
        };
        let gpu_pair = data.gpu_handles().map(|(x, y)| {
            format!(
                "x(device_id={}, buffer_id={}, shape={:?}) y(device_id={}, buffer_id={}, shape={:?})",
                x.device_id(), x.buffer_id(), x.shape(), y.device_id(), y.buffer_id(), y.shape()
            )
        });
        trace!(
            "plot: series={} requires_cpu={} inputs=({}, {}) gpu_pair={}",
            series_idx + 1,
            requires_cpu,
            x_kind,
            y_kind,
            gpu_pair
                .as_deref()
                .unwrap_or("<none: at least one input not GPU-resident>")
        );

        if !requires_cpu {
            if let Some((x_gpu, y_gpu)) = data.gpu_handles() {
                match build_line_gpu_plot_async(x_gpu, y_gpu, &label, &appearance).await? {
                    Some(line_plot) => {
                        trace!("plot: series={} used GPU path", series_idx + 1);
                        plots.push(line_plot);
                        continue;
                    }
                    None => {
                        trace!(
                            "plot: series={} GPU path unavailable before preparation",
                            series_idx + 1
                        );
                    }
                }
            }
        }

        trace!(
            "plot: series={} falling back to CPU gather path",
            series_idx + 1
        );
        let (x_tensor, y_tensor) = data
            .into_tensors_async("plot")
            .await
            .map_err(map_plot_invalid_argument)?;
        let (x_data, y_data) = numeric_plot_data_pair(x_tensor, y_tensor, "plot")
            .map_err(map_plot_invalid_argument)?;
        plots.push(
            build_line_plot_from_numeric_data(x_data, y_data, &label, &appearance)
                .map_err(map_plot_invalid_argument)?,
        );
    }

    let mut plots_opt = Some(plots);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let render_result = target.render(runtime, BUILTIN_NAME, opts, move |figure, axes_index| {
        let plots = plots_opt.take().expect("plot series consumed exactly once");
        for (idx, plot) in plots.into_iter().enumerate() {
            let plot_index = figure.add_line_plot_on_axes(plot, axes_index);
            if idx == 0 {
                *plot_index_slot.borrow_mut() = Some((axes_index, plot_index));
            }
        }
        if !hold_enabled {
            // Rendering cleared the axes and reset the color order; resume it after these series.
            continue_line_color_cycle_after(axes_index, total);
        }
        Ok(())
    });
    let Some((axes_index, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = target
        .register_line(axes_index, plot_index)
        .map_err(|source| super::properties::map_figure_error(BUILTIN_NAME, source))?;
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_plot_internal(err));
    }
    Ok(handle)
}

pub(super) fn build_line_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<LinePlot> {
    build_line_plot_for_builtin(BUILTIN_NAME, x, y, label, appearance)
}

pub(super) fn build_line_plot_for_builtin(
    builtin: &'static str,
    x: Vec<f64>,
    y: Vec<f64>,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<LinePlot> {
    let point_count = x.len();
    let mut plot = LinePlot::new(x, y)
        .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?
        .with_label(label)
        .with_handle_visibility(appearance.handle_visibility.clone())
        .with_style(
            appearance.color,
            appearance.line_width,
            appearance.line_style,
        );
    apply_marker_metadata(&mut plot, appearance, point_count);
    if let Some(visible) = appearance.visible {
        plot.set_visible(visible);
    }
    Ok(plot)
}

pub(super) fn build_line_plot_from_numeric_data(
    x: NumericPlotData,
    y: NumericPlotData,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<LinePlot> {
    build_line_plot_from_numeric_data_for_builtin(BUILTIN_NAME, x, y, label, appearance)
}

pub(super) fn build_line_plot_from_numeric_data_for_builtin(
    builtin: &'static str,
    x: NumericPlotData,
    y: NumericPlotData,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<LinePlot> {
    let point_count = x.len();
    let mut plot = LinePlot::from_numeric_data(x, y)
        .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?
        .with_label(label)
        .with_handle_visibility(appearance.handle_visibility.clone())
        .with_style(
            appearance.color,
            appearance.line_width,
            appearance.line_style,
        );
    apply_marker_metadata(&mut plot, appearance, point_count);
    if let Some(visible) = appearance.visible {
        plot.set_visible(visible);
    }
    Ok(plot)
}

fn parse_series_specs(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<(Vec<SeriesRenderPlan>, Option<Vec<LineStyle>>)> {
    if args.is_empty() {
        return Err(plot_err("expected at least one data series"));
    }
    let mut plans = Vec::new();
    let mut line_style_order: Option<Vec<LineStyle>> = None;
    let mut inline_opts = LineStyleParseOptions::plot();
    inline_opts.forbid_leading_numeric = false;

    let mut idx = 0usize;
    while idx < args.len() {
        let first_val = args[idx].clone();
        idx += 1;
        if !is_numeric_value(&first_val) {
            return Err(plot_err(
                "expected numeric X data before style arguments or options",
            ));
        }

        let shorthand_plot_y = match args.get(idx) {
            None => true,
            Some(Value::String(_) | Value::CharArray(_)) => true,
            Some(next) => !is_numeric_value(next),
        };

        let (x_val, y_val, y_arg_index) = if shorthand_plot_y {
            let y_val = first_val;
            let x_val = infer_plot_x_from_y(&y_val, runtime)?;
            (x_val, y_val, idx.saturating_sub(1))
        } else {
            let y_arg_index = idx;
            let y_val = args[idx].clone();
            idx += 1;
            if !is_numeric_value(&y_val) {
                return Err(plot_err("expected numeric Y argument after X data"));
            }
            (first_val, y_val, y_arg_index)
        };

        let pairs = split_matrix_series(x_val, y_val, shorthand_plot_y, runtime)?;

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
                .map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "plot",
                        source.to_string(),
                        source,
                    )
                })?
                .ok_or_else(|| plot_err("style tokens must be char arrays or strings"))?;
            let lower = token_text.trim().to_ascii_lowercase();
            style_tokens.push(token);

            if looks_like_plot_option_name(&lower) {
                if idx >= args.len() {
                    return Err(plot_err("name-value arguments must come in pairs"));
                }
                style_tokens.push(args[idx].clone());
                idx += 1;
            }
        }

        let parsed_style = parse_line_style_args(&style_tokens, &inline_opts)?;
        if let Some(order) = parsed_style.line_style_order.clone() {
            line_style_order = Some(order);
        }
        // A matrix argument is one line per column (or per row), each with the same style.
        for (x_part, y_part) in pairs {
            plans.push(SeriesRenderPlan {
                data: PlotSeriesInput::new(x_part, y_part)?,
                appearance: parsed_style.appearance.clone(),
                requires_cpu: parsed_style.requires_cpu_fallback,
                line_style_explicit: parsed_style.line_style_explicit,
                color_explicit: parsed_style.color_explicit,
                label: parsed_style.label.clone(),
                source_y_arg_index: Some(y_arg_index),
            });
        }
    }

    if plans.is_empty() {
        return Err(plot_err("expected at least one X/Y data pair"));
    }

    Ok((plans, line_style_order))
}

fn is_numeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

fn infer_plot_x_from_y(
    y: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let work = super::coordinate_admission::CoordinateWork::admit(runtime, BUILTIN_NAME)?;
    work.finish(NumericInput::implicit_x(y, BUILTIN_NAME, 0))
}

fn plot_err(msg: impl Into<String>) -> RuntimeError {
    plotting_error(BUILTIN_NAME, format!("plot: {}", msg.into()))
}

fn apply_line_style_order(plans: &mut [SeriesRenderPlan], order: &[LineStyle]) {
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

fn apply_marker_metadata(plot: &mut LinePlot, appearance: &LineAppearance, point_count: usize) {
    let marker = marker_metadata_from_appearance(appearance).or_else(|| {
        if point_count == 1 {
            // MATLAB renders a lone point for plot(x, y) when x/y have length 1. If the user
            // didn't specify a marker, default to a small point marker so the plot isn't blank.
            let mut temp = appearance.clone();
            temp.marker = Some(MarkerAppearance {
                kind: MarkerKind::Point,
                size: Some(DEFAULT_LINE_MARKER_SIZE),
                edge_color: MarkerColor::Auto,
                face_color: MarkerColor::Auto,
            });
            marker_metadata_from_appearance(&temp)
        } else {
            None
        }
    });
    if let Some(marker) = marker {
        plot.set_marker(Some(marker));
    }
}

#[path = "plot/gpu_line.rs"]
mod gpu_line;

async fn build_line_gpu_plot_async(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<Option<LinePlot>> {
    build_line_gpu_plot_async_for_builtin(BUILTIN_NAME, x, y, label, appearance).await
}

pub(super) async fn build_line_gpu_plot_async_for_builtin(
    builtin: &'static str,
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<Option<LinePlot>> {
    gpu_line::prepare(builtin, x, y, label, appearance).await
}

#[derive(Debug)]
struct PlotSeriesInput {
    x: NumericInput,
    y: NumericInput,
}

#[derive(Debug)]
struct SeriesRenderPlan {
    data: PlotSeriesInput,
    appearance: LineAppearance,
    requires_cpu: bool,
    line_style_explicit: bool,
    color_explicit: bool,
    label: Option<String>,
    source_y_arg_index: Option<usize>,
}

impl PlotSeriesInput {
    fn new(x: Value, y: Value) -> BuiltinResult<Self> {
        Ok(Self {
            x: NumericInput::from_value(x, BUILTIN_NAME)?,
            y: NumericInput::from_value(y, BUILTIN_NAME)?,
        })
    }

    fn gpu_handles(&self) -> Option<(&GpuTensorHandle, &GpuTensorHandle)> {
        match (self.x.gpu_handle(), self.y.gpu_handle()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        }
    }

    async fn into_tensors_async(self, name: &'static str) -> BuiltinResult<(Tensor, Tensor)> {
        let x = self.x.into_tensor_async(name).await?;
        let y = self.y.into_tensor_async(name).await?;
        Ok((x, y))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    fn parse_series_specs(
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<(Vec<SeriesRenderPlan>, Option<Vec<LineStyle>>)> {
        super::parse_series_specs(args, None)
    }
    fn infer_plot_x_from_y(y: &Value) -> BuiltinResult<Value> {
        super::infer_plot_x_from_y(y, None)
    }
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::state::{
        clear_figure, current_axes_handle_for_figure, line_color_for_series_index,
        reset_hold_state_for_run, set_hold, HoldMode,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clone_figure, configure_subplot, current_figure_handle};
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_plot::plots::PlotElement;
    use runmat_value::IntegerStorage;

    fn setup_plot_tests() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("plot test vector")
    }

    fn tensor_with_shape(data: &[f64], shape: Vec<usize>) -> Tensor {
        Tensor::new(data.to_vec(), shape).unwrap()
    }

    fn cleared_int_value(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn infer_plot_x_reads_typed_integer_length_without_mirror() {
        for y in [
            Value::Num(4.0),
            Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
            Value::Tensor(Tensor::from_f32(vec![4.0], vec![1, 1]).unwrap()),
        ] {
            let Value::Tensor(x) = infer_plot_x_from_y(&y).expect("scalar implicit axis") else {
                panic!("tensor axis")
            };
            assert_eq!(x.materialize_f64(), vec![1.0]);
            let (plans, _) = parse_series_specs(
                crate::builtins::common::test_support::builtin_arguments(vec![y]),
            )
            .expect("scalar series");
            assert_eq!(plans.len(), 1);
            assert_eq!(plans[0].data.y.len(), 1);
        }
        let x = infer_plot_x_from_y(&cleared_int_value(
            IntegerStorage::U16(vec![7, 8, 9]),
            vec![3, 1],
        ))
        .expect("x");

        let Value::Tensor(x) = x else {
            panic!("expected tensor");
        };
        assert_eq!(x.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    fn assert_plotting_unavailable(err: &RuntimeError) {
        assert!(
            crate::builtins::plotting::state::presentation_unavailable(err),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn build_plot_requires_equal_lengths() {
        let _guard = setup_plot_tests();
        assert!(build_line_plot(
            vec![1.0, 2.0],
            vec![1.0],
            "Series 1",
            &LineAppearance::default()
        )
        .is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plot_builtin_produces_figure_even_without_backend() {
        let _guard = setup_plot_tests();
        let result = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
        ]));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn plot_keeps_handle_after_typed_unavailable_presentation() {
        let guard = setup_plot_tests();
        let _unmanaged = guard.disable_host_managed_plot_env();
        let handle = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[2.0, 3.0])),
        ]))
        .expect("only presentation was unavailable");
        assert!(handle > 0.0);
        assert!(crate::builtins::plotting::state::clone_figure(
            crate::builtins::plotting::state::current_figure_handle(),
        )
        .is_some());
    }

    #[test]
    fn plot_owner_internal_wrapper_preserves_render_cause_and_retry() {
        let cause = crate::build_runtime_error("render operational failure")
            .with_source(runmat_plot::PlotPresentationError::InteractiveBackend(
                "native setup failed".into(),
            ))
            .with_span((7usize, 2usize).into())
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        let mapped = map_plot_internal(cause);
        assert_eq!(mapped.identifier(), PLOT_ERROR_INTERNAL.identifier);
        assert_eq!(mapped.span, Some((7usize, 2usize).into()));
        assert_eq!(mapped.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert_eq!(
            crate::builtins::plotting::state::presentation_failure_kind(&mapped),
            Some(runmat_plot::PresentationFailureKind::InteractiveBackend)
        );
    }

    #[test]
    fn plot_builtin_accepts_matching_numel_vectors_with_different_shapes() {
        let _guard = setup_plot_tests();
        let cases = [
            (
                tensor_with_shape(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]),
                tensor_with_shape(&[10.0, 20.0, 30.0, 40.0], vec![4, 1]),
            ),
            (
                tensor_with_shape(&[1.0, 2.0, 3.0, 4.0], vec![4, 1]),
                tensor_with_shape(&[10.0, 20.0, 30.0, 40.0], vec![1, 4]),
            ),
        ];

        for (x, y) in cases {
            let _ = clear_figure(None);
            block_on(plot_builtin(vec![Value::Tensor(x), Value::Tensor(y)]))
                .expect("plot should accept row/column vector pairs with matching numel");

            let fig = clone_figure(current_figure_handle()).expect("figure exists");
            let PlotElement::Line(line) = fig.plots().next().expect("plot created") else {
                panic!("expected line plot");
            };
            let (x, y) = line.host_xy_f64().unwrap().unwrap();
            assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);
            assert_eq!(y, vec![10.0, 20.0, 30.0, 40.0]);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plot_builtin_infers_label_from_callsite() {
        let _guard = setup_plot_tests();
        let source = "plot(a, b);";
        let _source_guard = crate::source_context::replace_current_source(Some(source));
        let spans = vec![
            runmat_types::Span { start: 5, end: 6 }, // "a"
            runmat_types::Span { start: 8, end: 9 }, // "b"
        ];
        let _callsite_guard =
            crate::callsite::push_callsite(None, Some(spans)).expect("plot callsite");

        let _ = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
        ]));

        let handle = current_figure_handle();
        let fig = clone_figure(handle).expect("figure exists");
        let entries = fig.legend_entries();
        assert!(!entries.is_empty(), "expected legend entries");
        assert_eq!(entries[0].label, "b");
    }

    #[test]
    fn plot_returns_handle_and_supports_line_properties() {
        let _guard = setup_plot_tests();
        let handle = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
        ]))
        .expect("plot should return a handle");
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap(),
            Value::String("line".into())
        );
        set_builtin(vec![
            Value::Num(handle),
            Value::String("LineWidth".into()),
            Value::Num(2.5),
            Value::String("DisplayName".into()),
            Value::String("Series A".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("LineWidth".into())]).unwrap(),
            Value::Num(2.5)
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayName".into())
            ])
            .unwrap(),
            Value::String("Series A".into())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_series_specs_handles_interleaved_styles() {
        let _guard = setup_plot_tests();
        let args = vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
            Value::String("r".into()),
            Value::Tensor(tensor_from(&[2.0, 3.0])),
            Value::Tensor(tensor_from(&[3.0, 4.0])),
            Value::String("--".into()),
        ];
        let (plans, order) = parse_series_specs(
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .expect("series parsed");
        assert_eq!(plans.len(), 2);
        assert!(order.is_none());
        assert_eq!(plans[1].appearance.line_style, LineStyle::Dashed);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plot_builtin_accepts_multi_series_with_per_series_color_tokens() {
        let _guard = setup_plot_tests();
        block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])),
            Value::String("r".into()),
            Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])),
            Value::Tensor(tensor_from(&[2.0, 3.0, 4.0])),
            Value::String("b".into()),
            Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])),
            Value::Tensor(tensor_from(&[-2.0, -3.0, -4.0])),
            Value::String("b".into()),
        ]))
        .expect("multi-series plot should parse");

        let fig = clone_figure(current_figure_handle()).expect("figure exists");
        let lines = fig.plots().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        let PlotElement::Line(first) = lines[0] else {
            panic!("expected first line plot");
        };
        let PlotElement::Line(second) = lines[1] else {
            panic!("expected second line plot");
        };
        assert_eq!(first.color, glam::Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(second.color, glam::Vec4::new(0.0, 0.0, 1.0, 1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plot_builtin_accepts_linespec_after_name_value_pairs() {
        let _guard = setup_plot_tests();
        let handle = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0, 2.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0, 0.0])),
            Value::String("LineWidth".into()),
            Value::Num(1.2),
            Value::String("b".into()),
        ]))
        .expect("plot should accept trailing color linespec");

        let width = get_builtin(vec![Value::Num(handle), Value::String("LineWidth".into())])
            .expect("LineWidth property");
        assert!(matches!(width, Value::Num(value) if (value - 1.2).abs() < 1.0e-6));
        let fig = clone_figure(current_figure_handle()).expect("figure exists");
        let PlotElement::Line(line) = fig.plots().next().expect("line exists") else {
            panic!("expected line plot");
        };
        assert_eq!(line.color, glam::Vec4::new(0.0, 0.0, 1.0, 1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plot_handle_visibility_off_hides_line_from_automatic_legend() {
        let _guard = setup_plot_tests();
        let handle = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::String("DisplayName".into()),
            Value::String("hidden".into()),
            Value::String("HandleVisibility".into()),
            Value::String("off".into()),
        ]))
        .expect("plot should accept HandleVisibility");

        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("HandleVisibility".into())
            ])
            .unwrap(),
            Value::String("off".into())
        );
        let fig = clone_figure(current_figure_handle()).expect("figure exists");
        assert!(fig.legend_entries().is_empty());
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayName".into())
            ])
            .unwrap(),
            Value::String("hidden".into())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn set_rejects_invalid_handle_visibility_for_line_handles() {
        let _guard = setup_plot_tests();
        let handle = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
        ]))
        .expect("plot should return a handle");

        let err = set_builtin(vec![
            Value::Num(handle),
            Value::String("HandleVisibility".into()),
            Value::String("maybe".into()),
        ])
        .expect_err("invalid HandleVisibility should fail");
        assert!(err.to_string().contains("HandleVisibility"));
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("HandleVisibility".into())
            ])
            .unwrap(),
            Value::String("on".into())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_series_specs_rejects_duplicate_standalone_linespec_tokens() {
        let _guard = setup_plot_tests();
        let args = vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::String("r".into()),
            Value::String("b".into()),
        ];
        let err = parse_series_specs(crate::builtins::common::test_support::builtin_arguments(
            args,
        ))
        .expect_err("duplicate LineSpec tokens should fail");
        assert!(err.to_string().contains("only one LineSpec"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_series_specs_errors_on_incomplete_pair() {
        let _guard = setup_plot_tests();
        let args = vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::String("linewidth".into()),
        ];
        let err = parse_series_specs(crate::builtins::common::test_support::builtin_arguments(
            args,
        ))
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("name-value arguments must come in pairs"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_series_specs_rejects_style_before_data() {
        let _guard = setup_plot_tests();
        let args = vec![Value::String("linewidth".into()), Value::Num(2.0)];
        let err = parse_series_specs(crate::builtins::common::test_support::builtin_arguments(
            args,
        ))
        .unwrap_err();
        assert!(err.to_string().contains("expected numeric X data"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_series_specs_extracts_line_style_order() {
        let _guard = setup_plot_tests();
        let args = vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
            Value::String("LineStyleOrder".into()),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["--".into(), ":".into()], vec![1, 2]).unwrap(),
            ),
        ];
        let (mut plans, order) = parse_series_specs(
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .expect("parsed");
        assert!(plans.len() == 1);
        assert!(order.is_some());
        apply_line_style_order(&mut plans, order.as_ref().unwrap());
        assert_eq!(plans[0].appearance.line_style, LineStyle::Dashed);
    }

    #[test]
    fn parse_series_specs_splits_matrix_y_into_one_series_per_column_or_row() {
        let wide = 9_007_199_254_740_993_u64;
        for source in [
            Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap(),
            Tensor::new_integer(
                IntegerStorage::U64(vec![
                    wide,
                    wide + 2,
                    wide + 4,
                    wide + 6,
                    wide + 8,
                    wide + 10,
                ]),
                vec![3, 2],
            )
            .unwrap(),
        ] {
            let original = source.clone();
            let (plans, _) =
                parse_series_specs(crate::builtins::common::test_support::builtin_arguments(
                    vec![Value::Tensor(source.clone())],
                ))
                .expect("native matrix series");
            assert_eq!(plans.len(), 2);
            for (column, plan) in plans.into_iter().enumerate() {
                let NumericInput::Host(y) = plan.data.y else {
                    panic!("host series")
                };
                assert_eq!(y.numeric_dtype(), source.numeric_dtype());
                if source.as_f32_slice().is_some() {
                    assert_eq!(
                        y.as_f32_slice().unwrap(),
                        if column == 0 {
                            &[1.0, 2.0, 3.0]
                        } else {
                            &[4.0, 5.0, 6.0]
                        }
                    );
                } else {
                    assert_eq!(
                        y.integer_storage(),
                        Some(&IntegerStorage::U64(if column == 0 {
                            vec![wide, wide + 2, wide + 4]
                        } else {
                            vec![wide + 6, wide + 8, wide + 10]
                        }))
                    );
                }
            }
            assert_eq!(source.shape(), original.shape());
            assert_eq!(source.as_f32_slice(), original.as_f32_slice());
            assert_eq!(source.integer_storage(), original.integer_storage());
        }
        let x = Value::Tensor(tensor_with_shape(&[1.0, 2.0, 3.0], vec![1, 3]));
        // 3x2: columns match x
        let cols = Value::Tensor(tensor_with_shape(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        ));
        let (plans, _) = parse_series_specs(
            crate::builtins::common::test_support::builtin_arguments(vec![x.clone(), cols.clone()]),
        )
        .expect("columns");
        assert_eq!(plans.len(), 2);
        // 2x3: rows match x
        let rows = Value::Tensor(tensor_with_shape(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ));
        let (plans, _) = parse_series_specs(
            crate::builtins::common::test_support::builtin_arguments(vec![x.clone(), rows]),
        )
        .expect("rows");
        assert_eq!(plans.len(), 2);
        // plot(Y): one line per column, not one flattened line
        let (plans, _) = parse_series_specs(
            crate::builtins::common::test_support::builtin_arguments(vec![cols]),
        )
        .expect("implicit x");
        assert_eq!(plans.len(), 2);
        // a length that matches neither side is an error
        let bad = Value::Tensor(tensor_with_shape(&[0.0; 8], vec![2, 4]));
        assert!(
            parse_series_specs(crate::builtins::common::test_support::builtin_arguments(
                vec![x, bad]
            ))
            .is_err()
        );
    }

    #[test]
    fn parse_series_specs_supports_plot_y_shorthand() {
        let _guard = setup_plot_tests();
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        let (empty_plans, _) = parse_series_specs(
            crate::builtins::common::test_support::builtin_arguments(vec![empty]),
        )
        .expect("empty plot-y input");
        assert_eq!(empty_plans.len(), 1);
        let (empty_x, empty_y) = futures::executor::block_on(
            empty_plans
                .into_iter()
                .next()
                .unwrap()
                .data
                .into_tensors_async("plot"),
        )
        .unwrap();
        assert!(empty_x.is_empty());
        assert!(empty_y.is_empty());
        let (empty_x, empty_y) = numeric_plot_data_pair(empty_x, empty_y, "plot").unwrap();
        assert!(build_line_plot_from_numeric_data(
            empty_x,
            empty_y,
            "empty",
            &LineAppearance::default()
        )
        .unwrap()
        .is_empty());
        let args = vec![Value::Tensor(tensor_from(&[10.0, 20.0, 30.0]))];
        let (plans, order) = parse_series_specs(
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .expect("parsed");
        assert!(order.is_none());
        assert_eq!(plans.len(), 1);
        let (x_tensor, y_tensor) = futures::executor::block_on(
            plans
                .into_iter()
                .next()
                .unwrap()
                .data
                .into_tensors_async("plot"),
        )
        .unwrap();
        assert_eq!(x_tensor.materialize_f64(), vec![1.0, 2.0, 3.0]);
        assert_eq!(y_tensor.materialize_f64(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn parse_series_specs_supports_plot_y_with_style() {
        let _guard = setup_plot_tests();
        let args = vec![
            Value::Tensor(tensor_from(&[1.0, 2.0])),
            Value::String("r--".into()),
        ];
        let (plans, order) = parse_series_specs(
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .expect("parsed");
        assert!(order.is_none());
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].appearance.line_style, LineStyle::Dashed);
    }

    #[test]
    fn parse_series_specs_treats_marker_only_linespec_as_no_line() {
        let _guard = setup_plot_tests();
        let args = vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
            Value::String("x".into()),
        ];
        let (plans, order) = parse_series_specs(
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .expect("parsed");
        assert!(order.is_none());
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].appearance.line_style, LineStyle::None);
        let marker = plans[0].appearance.marker.as_ref().expect("marker");
        assert!(matches!(marker.kind, MarkerKind::Cross));
        assert_eq!(marker.size, None);
    }

    #[test]
    fn plot_builtin_multi_series_cycles_color_and_keeps_solid_default_style() {
        let _guard = setup_plot_tests();
        let result = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[2.0, 3.0])),
        ]));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }

        let fig = clone_figure(current_figure_handle()).expect("figure exists");
        let plots = fig.plots().collect::<Vec<_>>();
        assert_eq!(plots.len(), 3);
        for (idx, element) in plots.iter().enumerate() {
            let PlotElement::Line(line) = element else {
                panic!("expected line plot")
            };
            assert_eq!(
                line.line_style,
                LineStyle::Solid,
                "series {} style",
                idx + 1
            );
            assert_eq!(
                line.color,
                line_color_for_series_index(idx),
                "series {} color",
                idx + 1
            );
        }
    }

    #[test]
    fn plot_builtin_hold_on_advances_implicit_color_cycle() {
        let _guard = setup_plot_tests();
        set_hold(HoldMode::On);

        block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
        ]))
        .expect("first plot");
        block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
        ]))
        .expect("second plot");

        let fig = clone_figure(current_figure_handle()).expect("figure exists");
        let plots = fig.plots().collect::<Vec<_>>();
        assert_eq!(plots.len(), 2);
        let PlotElement::Line(first) = plots[0] else {
            panic!("expected first line")
        };
        let PlotElement::Line(second) = plots[1] else {
            panic!("expected second line")
        };
        assert_eq!(first.color, line_color_for_series_index(0));
        assert_eq!(second.color, line_color_for_series_index(1));
        assert_eq!(first.line_style, LineStyle::Solid);
        assert_eq!(second.line_style, LineStyle::Solid);
    }

    #[test]
    fn plot_builtin_hold_on_after_plain_plot_takes_next_color() {
        let _guard = setup_plot_tests();
        set_hold(HoldMode::Off);
        let _ = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
        ]));
        set_hold(HoldMode::On);
        let _ = block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
        ]));

        let fig = clone_figure(current_figure_handle()).expect("figure exists");
        let colors = fig
            .plots()
            .map(|plot| match plot {
                PlotElement::Line(line) => line.color,
                _ => panic!("expected line plots"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            colors,
            vec![
                line_color_for_series_index(0),
                line_color_for_series_index(1)
            ]
        );
    }

    #[test]
    fn plot_builtin_supports_plot_y_shorthand_end_to_end() {
        let _guard = setup_plot_tests();
        let result = block_on(plot_builtin(vec![Value::Tensor(tensor_from(&[
            5.0, 6.0, 7.0,
        ]))]));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
        let handle = current_figure_handle();
        let fig = clone_figure(handle).expect("figure exists");
        assert!(!fig.legend_enabled);
        assert_eq!(fig.title.as_deref().unwrap_or_default(), "");
        assert_eq!(fig.x_label.as_deref().unwrap_or_default(), "");
        assert_eq!(fig.y_label.as_deref().unwrap_or_default(), "");
        let first_plot = fig.plots().next().expect("plot created");
        let PlotElement::Line(line) = first_plot else {
            panic!("expected line plot");
        };
        let (x, y) = line.host_xy_f64().unwrap().unwrap();
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
        assert_eq!(y, vec![5.0, 6.0, 7.0]);

        crate::builtins::plotting::title::title_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "Measured".into(),
            )]),
        )
        .unwrap();
        crate::builtins::plotting::xlabel::xlabel_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "Time".into(),
            )]),
        )
        .unwrap();
        crate::builtins::plotting::ylabel::ylabel_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "Signal".into(),
            )]),
        )
        .unwrap();
        set_hold(HoldMode::On);
        block_on(plot_builtin(vec![Value::Tensor(tensor_from(&[
            8.0, 9.0, 10.0,
        ]))]))
        .unwrap();
        let retained = clone_figure(handle).expect("figure exists");
        assert_eq!(retained.title.as_deref(), Some("Measured"));
        assert_eq!(retained.x_label.as_deref(), Some("Time"));
        assert_eq!(retained.y_label.as_deref(), Some("Signal"));

        set_hold(HoldMode::Off);
        block_on(plot_builtin(vec![Value::Tensor(tensor_from(&[
            11.0, 12.0, 13.0,
        ]))]))
        .unwrap();
        let replaced = clone_figure(handle).expect("figure exists");
        assert_eq!(replaced.len(), 1);
        assert!(replaced.title.is_none());
        assert!(replaced.x_label.is_none());
        assert!(replaced.y_label.is_none());
    }

    #[test]
    fn plot_builtin_accepts_leading_axes_handle() {
        let _guard = setup_plot_tests();
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        block_on(plot_builtin(vec![
            Value::Num(ax),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
        ]))
        .unwrap();
        let fig = clone_figure(fig_handle).unwrap();
        assert_eq!(fig.plot_axes_indices(), &[1]);
    }

    #[test]
    fn plot_builtin_accepts_scalar_point_with_style_args() {
        let _guard = setup_plot_tests();
        block_on(plot_builtin(vec![
            Value::Num(0.0),
            Value::Num(1.5),
            Value::String("o".into()),
            Value::String("LineWidth".into()),
            Value::Num(1.5),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Line(line) = fig.plots().next().unwrap() else {
            panic!("expected line")
        };
        let (x, y) = line.host_xy_f64().unwrap().unwrap();
        assert_eq!(x, vec![0.0]);
        assert_eq!(y, vec![1.5]);
    }

    #[test]
    fn plot_descriptor_signatures_cover_supported_forms() {
        let labels: Vec<&str> = PLOT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = plot(Y)"));
        assert!(labels.contains(&"h = plot(X, Y)"));
        assert!(labels.contains(&"h = plot(X, Y, Name, Value, ...)"));
        assert!(labels.contains(&"h = plot(ax, X, Y)"));
        assert!(labels.contains(&"h = plot(X1, Y1, ..., Xn, Yn)"));
    }

    #[test]
    fn plot_missing_post_axes_input_uses_stable_identifier() {
        let _guard = setup_plot_tests();
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        let err = block_on(plot_builtin(vec![Value::Num(ax)]))
            .expect_err("missing post-axes data should fail");
        assert_eq!(err.identifier(), PLOT_ERROR_INVALID_ARGUMENT.identifier);
    }
}
