//! MATLAB-compatible `bar` builtin.

use glam::{Vec3, Vec4};
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::bar::{BarGpuInputs, BarGpuParams, BarOrientation};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::BarChart;
use std::convert::TryFrom;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec};

use super::common::numeric_vector;
use super::gpu_helpers::axis_bounds;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_bar_style_args, BarStyle, BarStyleDefaults};

#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "bar"
category: "plotting"
keywords: ["bar", "bar chart", "categories", "gpuArray"]
summary: "Render MATLAB-compatible bar charts."
references:
  - https://www.mathworks.com/help/matlab/ref/bar.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["single"]
  broadcasting: "none"
  notes: "Single-precision gpuArray vectors stay on the device and feed the shared WebGPU renderer; other inputs gather automatically."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::plotting::bar::tests"
---

# What does `bar` do?
`bar(y)` plots the values of `y` as vertical bars with MATLAB-compatible defaults. RunMat matches
MATLAB's behaviour for 1-D inputs: categories are labelled `1,2,...,n`, grid lines are enabled, and
the current figure receives a descriptive title and axis labels.

## Behaviour highlights
- Inputs must be numeric vectors. Scalars produce a single bar; empty inputs raise MATLAB-style
  errors.
- Single-precision gpuArray inputs stay on the device and stream directly into the shared renderer
  for zero-copy plotting when WebGPU is available. Other data gathers automatically.
- Future work will add grouped/stacked variants; this initial builtin focuses on the common `bar(y)`
  form used throughout the standard library tests.

## Examples
```matlab
values = [3 5 2 9];
bar(values);
```

## GPU residency
`bar` terminates fusion graphs. Single-precision gpuArray vectors reuse provider buffers via the
shared WebGPU context; other precisions gather to the host before plotting.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "bar",
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
    notes: "Single-precision gpuArray vectors render zero-copy when the shared renderer is active; other contexts gather first.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "bar",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "bar performs I/O and terminates fusion graphs.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("bar", DOC_MD);

#[runtime_builtin(
    name = "bar",
    category = "plotting",
    summary = "Render a MATLAB-compatible bar chart.",
    keywords = "bar,barchart,plotting",
    sink = true
)]
pub fn bar_builtin(values: Value, rest: Vec<Value>) -> Result<String, String> {
    let defaults = BarStyleDefaults::new(default_bar_color(), DEFAULT_BAR_WIDTH);
    let style = parse_bar_style_args("bar", &rest, defaults)?;
    let mut input = Some(BarInput::from_value(values)?);
    let opts = PlotRenderOptions {
        title: "Bar Chart",
        x_label: "Category",
        y_label: "Value",
        ..Default::default()
    };
    render_active_plot(opts, move |figure, axes| {
        let style = style.clone();
        let arg = input.take().expect("bar input consumed once");
        if let Some(handle) = arg.gpu_handle() {
            match build_bar_gpu_chart(handle, &style) {
                Ok(mut bar) => {
                    apply_bar_style(&mut bar, &style, BAR_DEFAULT_LABEL);
                    figure.add_bar_chart_on_axes(bar, axes);
                    return Ok(());
                }
                Err(err) => {
                    warn!("bar GPU path unavailable: {err}");
                }
            }
        }
        let tensor = arg.into_tensor("bar")?;
        let vector = numeric_vector(tensor);
        let mut bar = build_bar_chart(vector)?;
        apply_bar_style(&mut bar, &style, BAR_DEFAULT_LABEL);
        figure.add_bar_chart_on_axes(bar, axes);
        Ok(())
    })
}

const DEFAULT_BAR_WIDTH: f32 = 0.75;
const BAR_DEFAULT_LABEL: &str = "Series 1";

fn default_bar_color() -> Vec4 {
    Vec4::new(0.2, 0.6, 0.9, 0.95)
}

fn build_bar_chart(values: Vec<f64>) -> Result<BarChart, String> {
    if values.is_empty() {
        return Err("bar: input cannot be empty".to_string());
    }
    let labels: Vec<String> = (1..=values.len()).map(|idx| format!("{idx}")).collect();

    let bar = BarChart::new(labels, values).map_err(|err| format!("bar: {err}"))?;
    Ok(bar)
}

fn build_bar_gpu_chart(values: &GpuTensorHandle, style: &BarStyle) -> Result<BarChart, String> {
    let context = runmat_plot::shared_wgpu_context()
        .ok_or_else(|| "bar: plotting GPU context unavailable".to_string())?;
    let exported = runmat_accelerate_api::export_wgpu_buffer(values)
        .ok_or_else(|| "bar: unable to export GPU values".to_string())?;

    if exported.len == 0 {
        return Err("bar: input cannot be empty".to_string());
    }
    let len_u32 = u32::try_from(exported.len)
        .map_err(|_| "bar: category count exceeds supported range".to_string())?;
    let scalar = ScalarType::from_is_f64(exported.precision == ProviderPrecision::F64);
    let inputs = BarGpuInputs {
        values_buffer: exported.buffer.clone(),
        len: len_u32,
        scalar,
    };
    let params = BarGpuParams {
        color: style.face_rgba(),
        bar_width: style.bar_width,
        group_index: 0,
        group_count: 1,
        orientation: BarOrientation::Vertical,
    };
    let gpu_vertices = runmat_plot::gpu::bar::pack_vertices_from_values(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| format!("bar: failed to build GPU vertices: {e}"))?;

    let labels: Vec<String> = (1..=exported.len).map(|idx| format!("{idx}")).collect();
    let bounds = build_bar_gpu_bounds(values, exported.len, params.bar_width)?;
    let vertex_count = gpu_vertices.vertex_count;

    let mut chart = BarChart::from_gpu_buffer(
        labels,
        exported.len,
        gpu_vertices,
        vertex_count,
        bounds,
        params.color,
        params.bar_width,
    );
    apply_bar_style(&mut chart, style, BAR_DEFAULT_LABEL);
    Ok(chart)
}

fn build_bar_gpu_bounds(
    values: &GpuTensorHandle,
    len: usize,
    bar_width: f32,
) -> Result<BoundingBox, String> {
    let (min_y, max_y) = axis_bounds(values, "bar")?;
    let min_y = min_y.min(0.0);
    let max_y = max_y.max(0.0);
    let min_x = 1.0 - bar_width * 0.5;
    let max_x = len as f32 + bar_width * 0.5;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, 0.0),
        Vec3::new(max_x, max_y, 0.0),
    ))
}

enum BarInput {
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl BarInput {
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = Tensor::try_from(&other).map_err(|e| format!("bar: {e}"))?;
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

    fn into_tensor(self, context: &str) -> Result<Tensor, String> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Gpu(handle) => gather_tensor_from_gpu(handle, context),
        }
    }
}

fn gather_tensor_from_gpu(handle: GpuTensorHandle, context: &str) -> Result<Tensor, String> {
    let value = Value::GpuTensor(handle);
    let gathered = gather_if_needed(&value)?;
    Tensor::try_from(&gathered).map_err(|e| format!("{context}: {e}"))
}

pub(crate) fn apply_bar_style(bar: &mut BarChart, style: &BarStyle, default_label: &str) {
    let face = style.face_rgba();
    bar.apply_face_style(face, style.bar_width);

    let outline = style.edge_rgba();
    bar.apply_outline_style(outline, style.line_width);

    if let Some(label) = &style.label {
        bar.label = Some(label.clone());
    } else if bar.label.is_none() {
        bar.label = Some(default_label.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::Value;

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
    fn bar_requires_non_empty_input() {
        assert!(build_bar_chart(vec![]).is_err());
    }

    #[test]
    fn bar_builtin_matches_backend_contract() {
        let out = bar_builtin(Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])), Vec::new());
        if let Err(msg) = out {
            assert!(
                msg.contains("Plotting is unavailable"),
                "unexpected error: {msg}"
            );
        }
    }
}
