//! MATLAB-compatible `stairs` builtin.

use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::line::{
    self, LineGpuInputs as MarkerGpuInputs, LineGpuParams as MarkerGpuParams,
};
use runmat_plot::gpu::stairs::{StairsGpuInputs, StairsGpuParams};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{LineMarkerAppearance, LineStyle, StairsPlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::numeric_pair;
use super::gpu_helpers::{gather_tensor_from_gpu, gpu_xy_bounds};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{
    marker_metadata_from_appearance, parse_line_style_args, LineAppearance, LineStyleParseOptions,
    DEFAULT_LINE_MARKER_SIZE,
};
use std::convert::TryFrom;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "stairs",
        wasm_path = "crate::builtins::plotting::ops::stairs"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "stairs"
category: "plotting"
keywords: ["stairs", "step plot", "2-D plotting", "gpuArray"]
summary: "Render MATLAB-compatible stairs (step) plots."
references:
  - https://www.mathworks.com/help/matlab/ref/stairs.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["single"]
  broadcasting: "none"
  notes: "Single-precision gpuArray vectors stay on the device when a shared WebGPU renderer is available; other inputs gather first."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::plotting::stairs::tests"
---

# What does `stairs` do?
`stairs(x, y)` draws a stairstep graph of the data in `y` versus the points in `x`, matching MATLAB's
default styling. Each successive pair of points generates a horizontal segment followed by a vertical
jump.

## Behaviour highlights
- Inputs must be real vectors of matching, non-zero length.
- With `single` precision gpuArrays and the shared WebGPU renderer active, RunMat packs the vertex
  buffer directly on the GPU. Other data falls back to the CPU path automatically.
- Like MATLAB, repeated calls append to the current axes when `hold on` is enabled.

## Example
```matlab
t = 0:5;
stairs(t, cumsum(rand(size(t))));
```
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::plotting::ops::stairs")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "stairs",
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
    notes: "Stairs plots terminate fusion graphs and render out-of-band.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::plotting::ops::stairs")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "stairs",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "stairs performs I/O and therefore terminates fusion graphs.",
};

#[runtime_builtin(
    name = "stairs",
    category = "plotting",
    summary = "Render MATLAB-compatible stairs plots.",
    keywords = "stairs,plotting,step",
    sink = true,
    wasm_path = "crate::builtins::plotting::ops::stairs"
)]
pub fn stairs_builtin(x: Value, y: Value, rest: Vec<Value>) -> Result<String, String> {
    let parsed_style = parse_line_style_args(&rest, &LineStyleParseOptions::stairs())?;
    let mut x_input = Some(StairsInput::from_value(x)?);
    let mut y_input = Some(StairsInput::from_value(y)?);
    let opts = PlotRenderOptions {
        title: "Stairs",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    render_active_plot(opts, move |figure, axes| {
        let appearance = parsed_style.appearance.clone();
        let marker_meta = marker_metadata_from_appearance(&appearance);
        let label = parsed_style
            .label
            .clone()
            .unwrap_or_else(|| "Data".to_string());
        let x_arg = x_input.take().expect("stairs x consumed once");
        let y_arg = y_input.take().expect("stairs y consumed once");

        if let (Some(x_gpu), Some(y_gpu)) = (x_arg.gpu_handle(), y_arg.gpu_handle()) {
            match build_stairs_gpu_plot(x_gpu, y_gpu, &appearance, marker_meta.clone(), &label) {
                Ok(plot) => {
                    figure.add_stairs_plot_on_axes(plot, axes);
                    return Ok(());
                }
                Err(err) => warn!("stairs GPU path unavailable: {err}"),
            }
        }

        let (x_tensor, y_tensor) = (x_arg.into_tensor("stairs")?, y_arg.into_tensor("stairs")?);
        let (x_vals, y_vals) = numeric_pair(x_tensor, y_tensor, "stairs")?;
        let plot = build_stairs_plot(x_vals, y_vals, &appearance, marker_meta, &label)?;
        figure.add_stairs_plot_on_axes(plot, axes);
        Ok(())
    })
}

fn build_stairs_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    appearance: &LineAppearance,
    marker_meta: Option<LineMarkerAppearance>,
    label: &str,
) -> Result<StairsPlot, String> {
    if x.len() != y.len() {
        return Err("stairs: X and Y inputs must share the same length".to_string());
    }
    if x.len() < 2 {
        return Err("stairs: inputs must contain at least two elements".to_string());
    }
    let mut plot = StairsPlot::new(x, y)
        .map_err(|e| format!("stairs: {e}"))?
        .with_style(appearance.color, appearance.line_width)
        .with_label(label);
    apply_stairs_marker_metadata(&mut plot, marker_meta);
    Ok(plot)
}

fn build_stairs_gpu_plot(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    appearance: &LineAppearance,
    marker_meta: Option<LineMarkerAppearance>,
    label: &str,
) -> Result<StairsPlot, String> {
    let context = runmat_plot::shared_wgpu_context()
        .ok_or_else(|| "stairs: plotting GPU context unavailable".to_string())?;

    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| "stairs: unable to export GPU X data".to_string())?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| "stairs: unable to export GPU Y data".to_string())?;

    if x_ref.len < 2 {
        return Err("stairs: inputs must contain at least two elements".to_string());
    }
    if x_ref.len != y_ref.len {
        return Err("stairs: X and Y inputs must have identical lengths".to_string());
    }
    if x_ref.precision != y_ref.precision {
        return Err("stairs: X and Y gpuArrays must share the same precision".to_string());
    }

    let len_u32 = u32::try_from(x_ref.len)
        .map_err(|_| "stairs: point count exceeds supported range".to_string())?;
    let scalar = ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64);

    let inputs = StairsGpuInputs {
        x_buffer: x_ref.buffer.clone(),
        y_buffer: y_ref.buffer.clone(),
        len: len_u32,
        scalar,
    };
    let params = StairsGpuParams {
        color: appearance.color,
    };

    let gpu_vertices = runmat_plot::gpu::stairs::pack_vertices_from_xy(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| format!("stairs: failed to build GPU vertices: {e}"))?;

    let marker_gpu = if let Some(marker) = marker_meta.clone() {
        let marker_inputs = MarkerGpuInputs {
            x_buffer: x_ref.buffer.clone(),
            y_buffer: y_ref.buffer.clone(),
            len: len_u32,
            scalar,
        };
        let marker_params = MarkerGpuParams {
            color: marker.face_color,
            line_width: 1.0,
            line_style: LineStyle::Solid,
            marker_size: marker.size.max(DEFAULT_LINE_MARKER_SIZE),
        };
        Some(
            line::pack_marker_vertices_from_xy(
                &context.device,
                &context.queue,
                &marker_inputs,
                &marker_params,
            )
            .map_err(|e| format!("stairs: failed to build marker vertices: {e}"))?,
        )
    } else {
        None
    };

    let bounds = gpu_xy_bounds(x, y, "stairs")?;
    let vertex_count = (x_ref.len - 1) * 4;
    let mut plot =
        StairsPlot::from_gpu_buffer(appearance.color, gpu_vertices, vertex_count, bounds)
            .with_style(appearance.color, appearance.line_width)
            .with_label(label);
    apply_stairs_marker_metadata(&mut plot, marker_meta);
    plot.set_marker_gpu_vertices(marker_gpu);
    Ok(plot)
}

fn apply_stairs_marker_metadata(plot: &mut StairsPlot, marker_meta: Option<LineMarkerAppearance>) {
    if let Some(marker) = marker_meta {
        plot.set_marker(Some(marker));
    } else {
        plot.set_marker(None);
    }
}

enum StairsInput {
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl StairsInput {
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = Tensor::try_from(&other).map_err(|e| format!("stairs: {e}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    #[ctor::ctor]
    fn init_plot_test_env() {
        crate::builtins::plotting::state::disable_rendering_for_tests();
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

    #[test]
    fn stairs_requires_matching_lengths() {
        let res = stairs_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0])),
            Vec::new(),
        );
        assert!(res.is_err());
    }

    #[test]
    fn stairs_requires_minimum_length() {
        let res = stairs_builtin(
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(tensor_from(&[1.0])),
            Vec::new(),
        );
        assert!(res.is_err());
    }
}
