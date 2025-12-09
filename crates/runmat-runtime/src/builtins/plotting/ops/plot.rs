//! MATLAB-compatible `plot` builtin.

use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::core::PipelineType;
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{LineGpuStyle, LinePlot, LineStyle};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::numeric_pair;
use super::gpu_helpers::{gather_tensor_from_gpu, gpu_xy_bounds};
use super::state::{
    next_line_style_for_axes, render_active_plot, set_line_style_order_for_axes, PlotRenderOptions,
};
use super::style::{
    looks_like_option_name, marker_metadata_from_appearance, parse_line_style_args,
    value_as_string, LineAppearance, LineStyleParseOptions, DEFAULT_LINE_MARKER_SIZE,
};
use std::collections::VecDeque;
use std::convert::TryFrom;

#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "plot")]
pub const DOC_MD: &str = r#"---
title: "plot"
category: "plotting"
keywords: ["plot", "line plot", "2D", "visualization", "gpuArray"]
summary: "Draw 2-D line plots that mirror MATLAB's `plot(x, y)` semantics."
references:
  - https://www.mathworks.com/help/matlab/ref/plot.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Single-precision gpuArray inputs render zero-copy via the shared WebGPU context; other data is gathered to the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::plotting::plot::tests"
---

# What does `plot` do?
`plot(x, y)` creates a 2-D line plot with MATLAB-compatible styling. Inputs may be real row/column
vectors or single-precision gpuArray vectors when a shared WebGPU context is available.

## Behaviour highlights
- Both `x` and `y` must contain the same number of elements; mismatched lengths raise errors.
- Default styling mirrors MATLAB: blue solid line with circular markers disabled. Line width matches
  MATLAB's default (approx. 1 pt) but can be adjusted once the interactive window is open.
- Multiple calls to `plot` append to the current figure when users call `hold on` (future work).
- Single-precision gpuArray vectors stay on the device and feed a zero-copy line packer. Double-precision
  data or dashed/marker-heavy styles fall back to the CPU path automatically.

## Examples

```matlab
plot(0:0.1:2*pi, sin(0:0.1:2*pi));
plot(time, amplitude);
```

## GPU residency
`plot` terminates fusion graphs. If the inputs are `single` gpuArrays and the shared plotter
device is active, their buffers are consumed zero-copy by the renderer. Otherwise the tensors are
gathered before plotting, matching MATLAB semantics.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "plot",
    op_kind: GpuOpKind::Custom("plot-render"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Plots are rendered on the host; gpuArray inputs are gathered before rendering.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "plot",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "plot performs I/O and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "plot",
    category = "plotting",
    summary = "Create MATLAB-compatible 2-D line plots.",
    keywords = "plot,line,2d,visualization",
    sink = true
)]
pub fn plot_builtin(x: Value, y: Value, rest: Vec<Value>) -> Result<String, String> {
    let mut args = Vec::with_capacity(2 + rest.len());
    args.push(x);
    args.push(y);
    args.extend(rest);

    let (mut series_plans, line_style_order) = parse_series_specs(args)?;
    if let Some(order) = line_style_order.as_ref() {
        apply_line_style_order(&mut series_plans, order);
    }
    let opts = PlotRenderOptions {
        title: "Plot",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    render_active_plot(opts, move |figure, axes| {
        if let Some(order) = line_style_order.clone() {
            set_line_style_order_for_axes(axes, &order);
        }
        render_series(figure, axes, &mut series_plans)
    })
}

fn build_line_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    label: &str,
    appearance: &LineAppearance,
) -> Result<LinePlot, String> {
    let mut plot = LinePlot::new(x, y)
        .map_err(|e| format!("plot: {e}"))?
        .with_label(label)
        .with_style(
            appearance.color,
            appearance.line_width,
            appearance.line_style,
        );
    apply_marker_metadata(&mut plot, appearance);
    Ok(plot)
}

#[derive(Debug)]
enum LineInput {
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl LineInput {
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = Tensor::try_from(&other).map_err(|e| format!("plot: {e}"))?;
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

    fn into_tensor(self, name: &str) -> Result<Tensor, String> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Gpu(handle) => gather_tensor_from_gpu(handle, name),
        }
    }
}

fn parse_series_specs(
    args: Vec<Value>,
) -> Result<(Vec<SeriesRenderPlan>, Option<Vec<LineStyle>>), String> {
    let mut queue: VecDeque<Value> = VecDeque::from(args);
    if queue.is_empty() {
        return Err(plot_err("expected at least one data series"));
    }
    let mut plans = Vec::new();
    let mut line_style_order: Option<Vec<LineStyle>> = None;
    let mut inline_opts = LineStyleParseOptions::plot();
    inline_opts.forbid_leading_numeric = false;
    while let Some(x_val) = queue.pop_front() {
        if !is_numeric_value(&x_val) {
            return Err(plot_err(
                "expected numeric X data before style arguments or options",
            ));
        }
        let y_val = queue
            .pop_front()
            .ok_or_else(|| plot_err("expected Y argument after X data"))?;
        if !is_numeric_value(&y_val) {
            return Err(plot_err("expected numeric Y argument after X data"));
        }
        let series_input = PlotSeriesInput::new(x_val, y_val)?;
        let style_tokens = consume_inline_style_tokens(&mut queue)?;
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

fn plot_err(msg: impl Into<String>) -> String {
    format!("plot: {}", msg.into())
}

fn consume_inline_style_tokens(queue: &mut VecDeque<Value>) -> Result<Vec<Value>, String> {
    let mut tokens = Vec::new();
    loop {
        let should_consume = matches!(queue.front(), Some(Value::String(_) | Value::CharArray(_)));
        if !should_consume {
            break;
        }

        let token = queue.pop_front().expect("front value exists");
        let token_text = value_as_string(&token)
            .ok_or_else(|| plot_err("style tokens must be char arrays or strings"))?;
        let lower = token_text.trim().to_ascii_lowercase();
        tokens.push(token);

        if looks_like_option_name(&lower) {
            let value = queue
                .pop_front()
                .ok_or_else(|| plot_err("name-value arguments must come in pairs"))?;
            tokens.push(value);
        }
    }
    Ok(tokens)
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

fn apply_marker_metadata(plot: &mut LinePlot, appearance: &LineAppearance) {
    if let Some(marker) = marker_metadata_from_appearance(appearance) {
        plot.set_marker(Some(marker));
    }
}

fn build_line_gpu_plot(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    label: &str,
    appearance: &LineAppearance,
) -> Result<LinePlot, String> {
    let context = runmat_plot::shared_wgpu_context()
        .ok_or_else(|| "plot: plotting GPU context unavailable".to_string())?;

    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| "plot: unable to export GPU X data".to_string())?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| "plot: unable to export GPU Y data".to_string())?;

    if x_ref.len < 2 {
        return Err("plot: inputs must contain at least two elements".to_string());
    }
    if x_ref.len != y_ref.len {
        return Err("plot: X and Y inputs must have identical lengths".to_string());
    }
    if x_ref.precision != y_ref.precision {
        return Err("plot: X and Y gpuArrays must have matching precision".to_string());
    }
    let len_u32 = u32::try_from(x_ref.len)
        .map_err(|_| "plot: point count exceeds supported range".to_string())?;
    let scalar = ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64);

    let inputs = runmat_plot::gpu::line::LineGpuInputs {
        x_buffer: x_ref.buffer.clone(),
        y_buffer: y_ref.buffer.clone(),
        len: len_u32,
        scalar,
    };
    let params = runmat_plot::gpu::line::LineGpuParams {
        color: appearance.color,
        line_width: appearance.line_width,
        line_style: appearance.line_style,
        marker_size: DEFAULT_LINE_MARKER_SIZE,
    };
    let marker_meta = marker_metadata_from_appearance(appearance);

    let gpu_vertices = runmat_plot::gpu::line::pack_vertices_from_xy(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| format!("plot: failed to build GPU vertices: {e}"))?;

    let marker_gpu_vertices = if let Some(marker) = marker_meta.as_ref() {
        let marker_params = runmat_plot::gpu::line::LineGpuParams {
            color: marker.face_color,
            line_width: 1.0,
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
            .map_err(|e| format!("plot: failed to build marker vertices: {e}"))?,
        )
    } else {
        None
    };

    let bounds = gpu_xy_bounds(x, y, "plot")?;
    let gpu_style = LineGpuStyle {
        color: appearance.color,
        line_width: appearance.line_width,
        line_style: appearance.line_style,
        marker: marker_meta.clone(),
    };
    let pipeline = if appearance.line_width > 1.0 {
        PipelineType::Triangles
    } else {
        PipelineType::Lines
    };
    let mut plot = LinePlot::from_gpu_buffer(
        gpu_vertices,
        x_ref.len,
        gpu_style,
        bounds,
        pipeline,
        marker_gpu_vertices,
    );
    plot = plot.with_label(label);
    Ok(plot)
}

fn render_series(
    figure: &mut runmat_plot::plots::Figure,
    axes_index: usize,
    plans: &mut Vec<SeriesRenderPlan>,
) -> Result<(), String> {
    let total = plans.len();
    for (series_idx, plan) in plans.drain(..).enumerate() {
        let SeriesRenderPlan {
            data,
            mut appearance,
            requires_cpu,
            line_style_explicit,
            label,
        } = plan;

        if !line_style_explicit {
            appearance.line_style = next_line_style_for_axes(axes_index);
        }

        let label = label.unwrap_or_else(|| {
            if total == 1 {
                "Data".to_string()
            } else {
                format!("Series {}", series_idx + 1)
            }
        });

        if !requires_cpu {
            if let Some((x_gpu, y_gpu)) = data.gpu_handles() {
                match build_line_gpu_plot(x_gpu, y_gpu, &label, &appearance) {
                    Ok(line_plot) => {
                        figure.add_line_plot_on_axes(line_plot, axes_index);
                        continue;
                    }
                    Err(err) => {
                        warn!("plot GPU path unavailable: {err}");
                    }
                }
            }
        }

        let (x_tensor, y_tensor) = data.into_tensors("plot")?;
        let (x_vals, y_vals) = numeric_pair(x_tensor, y_tensor, "plot")?;
        let line_plot = build_line_plot(x_vals, y_vals, &label, &appearance)?;
        figure.add_line_plot_on_axes(line_plot, axes_index);
    }
    Ok(())
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
}

impl PlotSeriesInput {
    fn new(x: Value, y: Value) -> Result<Self, String> {
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

    fn into_tensors(self, name: &str) -> Result<(Tensor, Tensor), String> {
        let x = self.x.into_tensor(name)?;
        let y = self.y.into_tensor(name)?;
        Ok((x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    #[test]
    fn build_plot_requires_equal_lengths() {
        assert!(build_line_plot(
            vec![1.0, 2.0],
            vec![1.0],
            "Series 1",
            &LineAppearance::default()
        )
        .is_err());
    }

    #[test]
    fn plot_builtin_produces_figure_even_without_backend() {
        let result = plot_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Vec::new(),
        );
        assert!(result.is_ok() || result.unwrap_err().contains("Plotting is unavailable"));
    }

    #[test]
    fn parse_series_specs_handles_interleaved_styles() {
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

    #[test]
    fn parse_series_specs_errors_on_incomplete_pair() {
        let args = vec![
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::String("linewidth".into()),
        ];
        let err = parse_series_specs(args).unwrap_err();
        assert!(err.contains("expected numeric Y argument"));
    }

    #[test]
    fn parse_series_specs_rejects_style_before_data() {
        let args = vec![Value::String("linewidth".into()), Value::Num(2.0)];
        let err = parse_series_specs(args).unwrap_err();
        assert!(err.contains("expected numeric X data"));
    }

    #[test]
    fn parse_series_specs_extracts_line_style_order() {
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
}
