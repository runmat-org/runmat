//! MATLAB-compatible `plot` builtin.

use log::trace;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{LineGpuStyle, LinePlot, LineStyle};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::numeric_pair;
use super::plotting_error;
use super::state::{
    current_axes_state, current_hold_enabled, next_line_style_for_axes, render_active_plot,
    set_line_style_order_for_axes, PlotRenderOptions,
};
use super::style::{
    looks_like_option_name, marker_metadata_from_appearance, parse_line_style_args,
    value_as_string, LineAppearance, LineStyleParseOptions, MarkerAppearance, MarkerColor,
    MarkerKind, DEFAULT_LINE_MARKER_SIZE,
};
use crate::builtins::plotting::type_resolvers::string_type;
use std::convert::TryFrom;

use crate::{BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::plot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "plot",
    op_kind: GpuOpKind::Custom("plot-render"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    // Plotting is a sink: it does not participate in fusion, but it *can* consume GPU-resident
    // tensors directly (zero-copy) when the web renderer shares the provider WGPU context.
    // Do not force implicit gathers here; that defeats GPU-resident workloads.
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Plots are rendered by the host renderer; GPU inputs may be consumed zero-copy when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::plot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "plot",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "plot performs I/O and terminates fusion graphs.",
};

const BUILTIN_NAME: &str = "plot";

#[runtime_builtin(
    name = "plot",
    category = "plotting",
    summary = "Create MATLAB-compatible 2-D line plots.",
    keywords = "plot,line,2d,visualization",
    sink = true,
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::plot"
)]
pub async fn plot_builtin(x: Value, y: Value, rest: Vec<Value>) -> crate::BuiltinResult<String> {
    let mut args = Vec::with_capacity(2 + rest.len());
    args.push(x);
    args.push(y);
    args.extend(rest);

    let (mut series_plans, line_style_order) = parse_series_specs(args)?;
    let axes = current_axes_state().active_index;
    let hold_enabled = current_hold_enabled();
    if let Some(order) = line_style_order.as_ref() {
        apply_line_style_order(&mut series_plans, order);
        set_line_style_order_for_axes(axes, order);
    }
    let opts = PlotRenderOptions {
        title: "Plot",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    let total = series_plans.len();
    let mut plots: Vec<LinePlot> = Vec::with_capacity(series_plans.len());
    for (series_idx, mut plan) in series_plans.drain(..).enumerate() {
        if !plan.line_style_explicit {
            plan.appearance.line_style = if hold_enabled {
                next_line_style_for_axes(axes)
            } else {
                match series_idx % 4 {
                    0 => LineStyle::Solid,
                    1 => LineStyle::Dashed,
                    2 => LineStyle::Dotted,
                    _ => LineStyle::DashDot,
                }
            };
            plan.line_style_explicit = true;
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
            LineInput::Host(_) => "Host",
            LineInput::Gpu(_) => "Gpu",
        };
        let y_kind = match &data.y {
            LineInput::Host(_) => "Host",
            LineInput::Gpu(_) => "Gpu",
        };
        let gpu_pair = data.gpu_handles().map(|(x, y)| {
            format!(
                "x(device_id={}, buffer_id={}, shape={:?}) y(device_id={}, buffer_id={}, shape={:?})",
                x.device_id, x.buffer_id, x.shape, y.device_id, y.buffer_id, y.shape
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
                match build_line_gpu_plot_async(x_gpu, y_gpu, &label, &appearance).await {
                    Ok(line_plot) => {
                        trace!("plot: series={} used GPU path", series_idx + 1);
                        plots.push(line_plot);
                        continue;
                    }
                    Err(err) => {
                        trace!(
                            "plot: series={} GPU path unavailable: {err}",
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
        let (x_tensor, y_tensor) = data.into_tensors_async("plot").await?;
        let (x_vals, y_vals) = numeric_pair(x_tensor, y_tensor, "plot")?;
        plots.push(build_line_plot(x_vals, y_vals, &label, &appearance)?);
    }

    let mut plots_opt = Some(plots);
    let rendered = render_active_plot(BUILTIN_NAME, opts, move |figure, axes_index| {
        let plots = plots_opt.take().expect("plot series consumed exactly once");
        for plot in plots {
            figure.add_line_plot_on_axes(plot, axes_index);
        }
        Ok(())
    })?;
    Ok(rendered)
}

fn build_line_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<LinePlot> {
    let point_count = x.len();
    let mut plot = LinePlot::new(x, y)
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("plot: {e}")))?
        .with_label(label)
        .with_style(
            appearance.color,
            appearance.line_width,
            appearance.line_style,
        );
    apply_marker_metadata(&mut plot, appearance, point_count);
    Ok(plot)
}

#[derive(Debug)]
enum LineInput {
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl LineInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = Tensor::try_from(&other)
                    .map_err(|e| plotting_error(BUILTIN_NAME, format!("plot: {e}")))?;
                Ok(Self::Host(tensor))
            }
        }
    }

    fn gpu_handle(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Gpu(handle) => Some(handle),
            Self::Host(_) => None,
        }
    }
}

fn parse_series_specs(
    args: Vec<Value>,
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
        let x_val = args[idx].clone();
        idx += 1;
        if !is_numeric_value(&x_val) {
            return Err(plot_err(
                "expected numeric X data before style arguments or options",
            ));
        }
        if idx >= args.len() {
            return Err(plot_err("expected Y argument after X data"));
        }
        let y_arg_index = idx;
        let y_val = args[idx].clone();
        idx += 1;
        if !is_numeric_value(&y_val) {
            return Err(plot_err("expected numeric Y argument after X data"));
        }

        let series_input = PlotSeriesInput::new(x_val, y_val)?;

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
                .ok_or_else(|| plot_err("style tokens must be char arrays or strings"))?;
            let lower = token_text.trim().to_ascii_lowercase();
            style_tokens.push(token);

            if looks_like_option_name(&lower) {
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
        plans.push(SeriesRenderPlan {
            data: series_input,
            appearance: parsed_style.appearance,
            requires_cpu: parsed_style.requires_cpu_fallback,
            line_style_explicit: parsed_style.line_style_explicit,
            label: parsed_style.label.clone(),
            source_y_arg_index: Some(y_arg_index),
        });
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

async fn build_line_gpu_plot_async(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<LinePlot> {
    let api_provider_present = runmat_accelerate_api::provider().is_some();
    let api_provider_for_x_present = runmat_accelerate_api::provider_for_handle(x).is_some();
    let api_provider_for_y_present = runmat_accelerate_api::provider_for_handle(y).is_some();
    let shared_ctx_present = runmat_plot::shared_wgpu_context().is_some();
    trace!(
        "plot-gpu: attempt label={label:?} x(device_id={}, buffer_id={}, shape={:?}) y(device_id={}, buffer_id={}, shape={:?}) shared_ctx_present={} api_provider_present={} api_provider_for_x_present={} api_provider_for_y_present={}",
        x.device_id,
        x.buffer_id,
        x.shape,
        y.device_id,
        y.buffer_id,
        y.shape,
        shared_ctx_present,
        api_provider_present,
        api_provider_for_x_present,
        api_provider_for_y_present
    );
    let context = crate::builtins::plotting::gpu_helpers::ensure_shared_wgpu_context(BUILTIN_NAME)?;

    let x_ref = match runmat_accelerate_api::export_wgpu_buffer(x) {
        Some(buf) => {
            trace!(
                "plot-gpu: export_wgpu_buffer(X) ok len={} element_size={} precision={:?}",
                buf.len,
                buf.element_size,
                buf.precision
            );
            buf
        }
        None => {
            trace!(
                "plot-gpu: export_wgpu_buffer(X) FAILED (api_provider_present={} api_provider_for_x_present={} x_device_id={})",
                api_provider_present, api_provider_for_x_present, x.device_id
            );
            return Err(plotting_error(
                BUILTIN_NAME,
                "plot: unable to export GPU X data",
            ));
        }
    };
    let y_ref = match runmat_accelerate_api::export_wgpu_buffer(y) {
        Some(buf) => {
            trace!(
                "plot-gpu: export_wgpu_buffer(Y) ok len={} element_size={} precision={:?}",
                buf.len,
                buf.element_size,
                buf.precision
            );
            buf
        }
        None => {
            trace!(
                "plot-gpu: export_wgpu_buffer(Y) FAILED (api_provider_present={} api_provider_for_y_present={} y_device_id={})",
                api_provider_present, api_provider_for_y_present, y.device_id
            );
            return Err(plotting_error(
                BUILTIN_NAME,
                "plot: unable to export GPU Y data",
            ));
        }
    };

    if x_ref.len < 2 {
        return Err(plot_err("inputs must contain at least two elements"));
    }
    if x_ref.len != y_ref.len {
        return Err(plot_err("X and Y inputs must have identical lengths"));
    }
    if x_ref.precision != y_ref.precision {
        return Err(plot_err("X and Y gpuArrays must have matching precision"));
    }
    let len_u32 =
        u32::try_from(x_ref.len).map_err(|_| plot_err("point count exceeds supported range"))?;
    let scalar = ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64);

    let inputs = runmat_plot::gpu::line::LineGpuInputs {
        x_buffer: x_ref.buffer.clone(),
        y_buffer: y_ref.buffer.clone(),
        len: len_u32,
        scalar,
    };
    let marker_meta = marker_metadata_from_appearance(appearance);

    let marker_gpu_vertices = if let Some(marker) = marker_meta.as_ref() {
        let marker_params = runmat_plot::gpu::line::LineGpuParams {
            color: marker.face_color,
            half_width_data: 0.0,
            thick: false,
            line_style: LineStyle::Solid,
            marker_size: marker.size.max(1.0),
        };
        Some(
            runmat_plot::gpu::line::pack_marker_vertices_from_xy(
                &context.device,
                &context.queue,
                &inputs,
                &marker_params,
            )
            .map_err(|e| {
                plotting_error(
                    BUILTIN_NAME,
                    format!("plot: failed to build marker vertices: {e}"),
                )
            })?,
        )
    } else {
        None
    };

    let bounds = super::gpu_helpers::gpu_xy_bounds_async(x, y, "plot").await?;
    let gpu_style = LineGpuStyle {
        color: appearance.color,
        line_width: appearance.line_width,
        line_style: appearance.line_style,
        marker: marker_meta.clone(),
    };
    let mut plot = LinePlot::from_gpu_xy(inputs, gpu_style, bounds, marker_gpu_vertices);
    plot = plot.with_label(label);
    Ok(plot)
}

#[derive(Debug)]
struct PlotSeriesInput {
    x: LineInput,
    y: LineInput,
}

#[derive(Debug)]
struct SeriesRenderPlan {
    data: PlotSeriesInput,
    appearance: LineAppearance,
    requires_cpu: bool,
    line_style_explicit: bool,
    label: Option<String>,
    source_y_arg_index: Option<usize>,
}

impl PlotSeriesInput {
    fn new(x: Value, y: Value) -> BuiltinResult<Self> {
        Ok(Self {
            x: LineInput::from_value(x)?,
            y: LineInput::from_value(y)?,
        })
    }

    fn gpu_handles(&self) -> Option<(&GpuTensorHandle, &GpuTensorHandle)> {
        match (self.x.gpu_handle(), self.y.gpu_handle()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        }
    }

    async fn into_tensors_async(self, name: &'static str) -> BuiltinResult<(Tensor, Tensor)> {
        let x = match self.x {
            LineInput::Host(t) => t,
            LineInput::Gpu(h) => super::gpu_helpers::gather_tensor_from_gpu_async(h, name).await?,
        };
        let y = match self.y {
            LineInput::Host(t) => t,
            LineInput::Gpu(h) => super::gpu_helpers::gather_tensor_from_gpu_async(h, name).await?,
        };
        Ok((x, y))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use crate::builtins::plotting::state::{clear_figure, reset_hold_state_for_run};
    use crate::builtins::plotting::{clone_figure, current_figure_handle};
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
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
    fn build_plot_requires_equal_lengths() {
        setup_plot_tests();
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
        setup_plot_tests();
        let result = block_on(plot_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Vec::new(),
        ));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plot_builtin_infers_label_from_callsite() {
        setup_plot_tests();
        let source = "plot(a, b);";
        let _source_guard = crate::source_context::replace_current_source(Some(source));
        let spans = vec![
            runmat_hir::Span { start: 5, end: 6 }, // "a"
            runmat_hir::Span { start: 8, end: 9 }, // "b"
        ];
        let _callsite_guard = crate::callsite::push_callsite(None, Some(spans));

        let _ = block_on(plot_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Vec::new(),
        ));

        let handle = current_figure_handle();
        let fig = clone_figure(handle).expect("figure exists");
        let entries = fig.legend_entries();
        assert!(!entries.is_empty(), "expected legend entries");
        assert_eq!(entries[0].label, "b");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_series_specs_handles_interleaved_styles() {
        setup_plot_tests();
        let args = vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
            Value::String("r".into()),
            Value::Tensor(tensor_from(&[2.0, 3.0])),
            Value::Tensor(tensor_from(&[3.0, 4.0])),
            Value::String("--".into()),
        ];
        let (plans, order) = parse_series_specs(args).expect("series parsed");
        assert_eq!(plans.len(), 2);
        assert!(order.is_none());
        assert_eq!(plans[1].appearance.line_style, LineStyle::Dashed);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_series_specs_errors_on_incomplete_pair() {
        setup_plot_tests();
        let args = vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::String("linewidth".into()),
        ];
        let err = parse_series_specs(args).unwrap_err();
        assert!(err.to_string().contains("expected numeric Y argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_series_specs_rejects_style_before_data() {
        setup_plot_tests();
        let args = vec![Value::String("linewidth".into()), Value::Num(2.0)];
        let err = parse_series_specs(args).unwrap_err();
        assert!(err.to_string().contains("expected numeric X data"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_series_specs_extracts_line_style_order() {
        setup_plot_tests();
        let args = vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[1.0, 2.0])),
            Value::String("LineStyleOrder".into()),
            Value::StringArray(runmat_builtins::StringArray {
                data: vec!["--".into(), ":".into()],
                shape: vec![1, 2],
                rows: 1,
                cols: 2,
            }),
        ];
        let (mut plans, order) = parse_series_specs(args).expect("parsed");
        assert!(plans.len() == 1);
        assert!(order.is_some());
        apply_line_style_order(&mut plans, order.as_ref().unwrap());
        assert_eq!(plans[0].appearance.line_style, LineStyle::Dashed);
    }

    #[test]
    fn plot_type_is_string() {
        assert_eq!(
            string_type(&[Type::tensor(), Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}
