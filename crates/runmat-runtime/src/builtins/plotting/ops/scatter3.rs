//! MATLAB-compatible `scatter3` builtin.

use glam::{Vec3, Vec4};
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::scatter2::{ScatterAttributeBuffer, ScatterColorBuffer};
use runmat_plot::gpu::scatter3::{Scatter3GpuInputs, Scatter3GpuParams};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::surface::ColorMap;
use runmat_plot::plots::LineStyle;
use runmat_plot::plots::Scatter3Plot;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;
use std::convert::TryFrom;

use super::common::numeric_triplet;
use super::gpu_helpers::axis_bounds;
use super::perf::scatter3_lod_stride;
use super::point::{
    convert_rgb_color_matrix, convert_scalar_color_values, convert_size_vector,
    map_scalar_values_to_colors, validate_gpu_color_matrix, validate_gpu_vector_length, PointArgs,
    PointColorArg, PointGpuColor, PointSizeArg,
};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::LineStyleParseOptions;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "scatter3")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "scatter3"
category: "plotting"
keywords: ["scatter3", "3-D scatter", "point cloud", "gpuArray"]
summary: "Create MATLAB-compatible 3-D scatter plots."
references:
  - https://www.mathworks.com/help/matlab/ref/scatter3.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Rendering runs on the host or WebGPU canvas after gathering gpuArray data when necessary."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::plotting::scatter3::tests"
---

# What does `scatter3` do?
`scatter3(x, y, z)` plots points in 3-D space using MATLAB-compatible defaults. Each input must
contain the same number of elements; row and column vectors are both accepted.

## Behaviour highlights
- Inputs may be real doubles, single-precision tensors, or gathered gpuArray values. Complex data
  currently raises an error matching MATLAB.
- Points inherit MATLAB’s default styling: blue markers with mild transparency. Future work will
  add size/color arguments, but existing scripts using the basic call form work today.
- Fusion graphs terminate at `scatter3`, and gpuArray inputs are gathered so the renderer can access
  dense host memory or a shared WebGPU buffer depending on the build.

## Examples

```matlab
t = linspace(0, 4*pi, 200);
scatter3(cos(t), sin(t), t);
```

## GPU residency
`scatter3` gathers GPU tensors before plotting today. The new shared-device renderer keeps future
implementations zero-copy. Until that lands, expect the builtin to behave like MATLAB: data moves
to the host, rendering completes, and execution returns immediately.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "scatter3",
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
    notes: "Rendering executes outside fusion; gpuArray inputs are gathered prior to plotting.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "scatter3",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "scatter3 terminates fusion graphs and performs host/WebGPU rendering.",
};

#[runtime_builtin(
    name = "scatter3",
    category = "plotting",
    summary = "Render a MATLAB-compatible 3-D scatter plot.",
    keywords = "scatter3,plotting,3d,pointcloud",
    sink = true
)]
pub fn scatter3_builtin(x: Value, y: Value, z: Value, rest: Vec<Value>) -> Result<String, String> {
    let style_args = PointArgs::parse(rest, LineStyleParseOptions::scatter3())?;
    let mut x_input = Some(ScatterInput::from_value(x)?);
    let mut y_input = Some(ScatterInput::from_value(y)?);
    let mut z_input = Some(ScatterInput::from_value(z)?);
    let opts = PlotRenderOptions {
        title: "3-D Scatter",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    render_active_plot(opts, move |figure, axes| {
        let style_args = style_args.clone();
        let point_count = x_input.as_ref().map(|input| input.len()).unwrap_or(0);
        let mut resolved_style = resolve_scatter3_style(point_count, &style_args, "scatter3")?;
        let x_arg = x_input.take().expect("scatter3 x consumed once");
        let y_arg = y_input.take().expect("scatter3 y consumed once");
        let z_arg = z_input.take().expect("scatter3 z consumed once");

        if let (Some(x_gpu), Some(y_gpu), Some(z_gpu)) =
            (x_arg.gpu_handle(), y_arg.gpu_handle(), z_arg.gpu_handle())
        {
            if !resolved_style.requires_cpu {
                match build_scatter3_gpu_plot(x_gpu, y_gpu, z_gpu, &resolved_style) {
                    Ok(plot) => {
                        figure.add_scatter3_plot_on_axes(plot, axes);
                        return Ok(());
                    }
                    Err(err) => {
                        warn!("scatter3 GPU path unavailable: {err}");
                    }
                }
            }
        }

        let (x_tensor, y_tensor, z_tensor) = (
            x_arg.into_tensor("scatter3")?,
            y_arg.into_tensor("scatter3")?,
            z_arg.into_tensor("scatter3")?,
        );
        let (x_vals, y_vals, z_vals) = numeric_triplet(x_tensor, y_tensor, z_tensor, "scatter3")?;
        let scatter = build_scatter3_plot(x_vals, y_vals, z_vals, &mut resolved_style)?;
        figure.add_scatter3_plot_on_axes(scatter, axes);
        Ok(())
    })
}

const DEFAULT_POINT_SIZE: f32 = 6.0;
const DEFAULT_SCATTER3_LABEL: &str = "Data";

fn default_color() -> Vec4 {
    Vec4::new(0.1, 0.6, 0.9, 0.9)
}

#[derive(Clone, Debug)]
struct Scatter3ResolvedStyle {
    uniform_color: Vec4,
    point_size: f32,
    per_point_sizes: Option<Vec<f32>>,
    per_point_colors: Option<Vec<Vec4>>,
    color_values: Option<Vec<f64>>,
    color_limits: Option<(f64, f64)>,
    gpu_sizes: Option<GpuTensorHandle>,
    gpu_colors: Option<PointGpuColor>,
    colormap: ColorMap,
    requires_cpu: bool,
    label: String,
}

fn resolve_scatter3_style(
    point_count: usize,
    args: &PointArgs,
    context: &str,
) -> Result<Scatter3ResolvedStyle, String> {
    let mut style = Scatter3ResolvedStyle {
        uniform_color: default_color(),
        point_size: DEFAULT_POINT_SIZE,
        per_point_sizes: None,
        per_point_colors: None,
        color_values: None,
        color_limits: None,
        gpu_sizes: None,
        gpu_colors: None,
        colormap: ColorMap::Parula,
        requires_cpu: false,
        label: DEFAULT_SCATTER3_LABEL.to_string(),
    };

    if let Some(label) = args.style.label.clone() {
        style.label = label;
    }

    let appearance = &args.style.appearance;
    style.uniform_color = appearance.color;

    if let PointColorArg::Uniform(color) = &args.color {
        style.uniform_color = *color;
    }

    if let Some(marker) = appearance.marker.as_ref() {
        if let Some(size) = marker.size {
            style.point_size = size.max(0.1);
        }
    }

    if let PointSizeArg::Scalar(size) = &args.size {
        style.point_size = (*size).max(0.1);
    }

    if let Some(value) = args.size.value() {
        match value {
            Value::GpuTensor(handle) => {
                validate_gpu_vector_length(handle, point_count, context)?;
                style.gpu_sizes = Some(handle.clone());
            }
            _ => {
                style.per_point_sizes = Some(convert_size_vector(value, point_count, context)?);
            }
        }
    }

    match &args.color {
        PointColorArg::ScalarValues(value) => {
            let scalars = convert_scalar_color_values(value, point_count, context)?;
            let (colors, limits) = map_scalar_values_to_colors(&scalars, style.colormap);
            style.color_values = Some(scalars);
            style.per_point_colors = Some(colors);
            style.color_limits = Some(limits);
        }
        PointColorArg::RgbMatrix(value) => match value {
            Value::GpuTensor(handle) => {
                let components = validate_gpu_color_matrix(handle, point_count, context)?;
                style.gpu_colors = Some(PointGpuColor {
                    handle: handle.clone(),
                    components,
                });
            }
            _ => {
                style.per_point_colors =
                    Some(convert_rgb_color_matrix(value, point_count, context)?);
            }
        },
        _ => {}
    }

    if args.style.appearance.line_style != LineStyle::Solid && args.style.line_style_explicit {
        style.requires_cpu = true;
    }
    if args.style.line_style_order.is_some() {
        style.requires_cpu = true;
    }
    style.requires_cpu |= args.style.requires_cpu_fallback;

    Ok(style)
}

fn build_scatter3_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    style: &mut Scatter3ResolvedStyle,
) -> Result<Scatter3Plot, String> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err("scatter3: X, Y, and Z inputs must have identical lengths".to_string());
    }

    ensure_scatter3_host_metadata(style, x.len())?;

    let points: Vec<Vec3> = x
        .iter()
        .zip(y.iter())
        .zip(z.iter())
        .map(|((x, y), z)| Vec3::new(*x as f32, *y as f32, *z as f32))
        .collect();

    let mut scatter = Scatter3Plot::new(points)
        .map_err(|err| format!("scatter3: {err}"))?
        .with_point_size(style.point_size)
        .with_color(style.uniform_color)
        .with_label(style.label.clone());
    if let Some(sizes) = style.per_point_sizes.take() {
        scatter.set_point_sizes(sizes);
    }
    if let Some(colors) = style.per_point_colors.take() {
        scatter = scatter
            .with_colors(colors)
            .map_err(|e| format!("scatter3: {e}"))?;
    }
    Ok(scatter)
}

enum ScatterInput {
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl ScatterInput {
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = Tensor::try_from(&other).map_err(|e| format!("scatter3: {e}"))?;
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

    fn len(&self) -> usize {
        match self {
            Self::Host(tensor) => tensor.data.len(),
            Self::Gpu(handle) => handle.shape.iter().product(),
        }
    }

    fn into_tensor(self, name: &str) -> Result<Tensor, String> {
        match self {
            Self::Host(t) => Ok(t),
            Self::Gpu(handle) => gather_tensor_from_gpu(handle, name),
        }
    }
}

fn gather_tensor_from_gpu(handle: GpuTensorHandle, name: &str) -> Result<Tensor, String> {
    let value = Value::GpuTensor(handle);
    let gathered = gather_if_needed(&value)?;
    Tensor::try_from(&gathered).map_err(|e| format!("{name}: {e}"))
}

fn build_scatter3_gpu_plot(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    z: &GpuTensorHandle,
    style: &Scatter3ResolvedStyle,
) -> Result<Scatter3Plot, String> {
    let context = runmat_plot::shared_wgpu_context()
        .ok_or_else(|| "scatter3: plotting GPU context unavailable".to_string())?;

    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| "scatter3: unable to export GPU X data".to_string())?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| "scatter3: unable to export GPU Y data".to_string())?;
    let z_ref = runmat_accelerate_api::export_wgpu_buffer(z)
        .ok_or_else(|| "scatter3: unable to export GPU Z data".to_string())?;

    if x_ref.len == 0 {
        return Err("scatter3: empty input tensor".to_string());
    }
    if x_ref.len != y_ref.len || x_ref.len != z_ref.len {
        return Err("scatter3: X, Y, and Z inputs must have identical lengths".to_string());
    }
    if x_ref.precision != y_ref.precision || x_ref.precision != z_ref.precision {
        return Err("scatter3: gpuArray inputs must have matching precision".to_string());
    }
    let point_count = x_ref.len;
    let len_u32 = u32::try_from(point_count)
        .map_err(|_| "scatter3: point count exceeds supported range".to_string())?;
    let scalar = ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64);
    let bounds = build_gpu_bounds(x, y, z)?;
    let extent_hint = scatter3_extent_hint(&bounds);
    let lod_stride = scatter3_lod_stride(len_u32, extent_hint);

    let size_buffer = build_scatter3_size_buffer(style, point_count)?;
    let color_buffer = build_scatter3_color_buffer(style, point_count)?;

    let inputs = Scatter3GpuInputs {
        x_buffer: x_ref.buffer.clone(),
        y_buffer: y_ref.buffer.clone(),
        z_buffer: z_ref.buffer.clone(),
        len: len_u32,
        scalar,
    };
    let params = Scatter3GpuParams {
        color: style.uniform_color,
        point_size: style.point_size,
        sizes: size_buffer,
        colors: color_buffer,
        lod_stride,
    };

    let gpu_vertices = runmat_plot::gpu::scatter3::pack_vertices_from_xyz(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| format!("scatter3: failed to build GPU vertices: {e}"))?;

    let drawn_points = gpu_vertices.vertex_count;
    Ok(Scatter3Plot::from_gpu_buffer(
        gpu_vertices,
        drawn_points,
        style.uniform_color,
        style.point_size,
        bounds,
    )
    .with_label(style.label.clone()))
}

fn build_gpu_bounds(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    z: &GpuTensorHandle,
) -> Result<BoundingBox, String> {
    let (min_x, max_x) = axis_bounds(x, "scatter3")?;
    let (min_y, max_y) = axis_bounds(y, "scatter3")?;
    let (min_z, max_z) = axis_bounds(z, "scatter3")?;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, min_z),
        Vec3::new(max_x, max_y, max_z),
    ))
}

fn scatter3_extent_hint(bounds: &BoundingBox) -> f32 {
    bounds.size().length().max(1.0)
}

fn build_scatter3_size_buffer(
    style: &Scatter3ResolvedStyle,
    point_count: usize,
) -> Result<ScatterAttributeBuffer, String> {
    if let Some(handle) = style.gpu_sizes.as_ref() {
        let exported = runmat_accelerate_api::export_wgpu_buffer(handle)
            .ok_or_else(|| "scatter3: unable to export GPU marker sizes".to_string())?;
        if exported.len != point_count {
            return Err(format!(
                "scatter3: marker size array must have {point_count} elements (got {})",
                exported.len
            ));
        }
        if exported.precision != ProviderPrecision::F32 {
            return Err(
                "scatter3: GPU marker sizes must be single-precision (cast before plotting)"
                    .to_string(),
            );
        }
        return Ok(ScatterAttributeBuffer::Gpu(exported.buffer));
    }
    if let Some(sizes) = style.per_point_sizes.as_ref() {
        if sizes.is_empty() {
            Ok(ScatterAttributeBuffer::None)
        } else {
            Ok(ScatterAttributeBuffer::Host(sizes.clone()))
        }
    } else {
        Ok(ScatterAttributeBuffer::None)
    }
}

fn build_scatter3_color_buffer(
    style: &Scatter3ResolvedStyle,
    point_count: usize,
) -> Result<ScatterColorBuffer, String> {
    if let Some(gpu_color) = style.gpu_colors.as_ref() {
        let exported = runmat_accelerate_api::export_wgpu_buffer(&gpu_color.handle)
            .ok_or_else(|| "scatter3: unable to export GPU color data".to_string())?;
        let expected = point_count * gpu_color.components.stride() as usize;
        if exported.len != expected {
            return Err(format!(
                "scatter3: color array must contain {} elements (got {})",
                expected, exported.len
            ));
        }
        if exported.precision != ProviderPrecision::F32 {
            return Err(
                "scatter3: GPU color arrays must be single-precision (cast before plotting)"
                    .to_string(),
            );
        }
        return Ok(ScatterColorBuffer::Gpu {
            buffer: exported.buffer,
            components: gpu_color.components.stride(),
        });
    }

    if let Some(colors) = style.per_point_colors.as_ref() {
        if colors.is_empty() {
            Ok(ScatterColorBuffer::None)
        } else {
            let data = colors.iter().map(|c| c.to_array()).collect();
            Ok(ScatterColorBuffer::Host(data))
        }
    } else {
        Ok(ScatterColorBuffer::None)
    }
}

fn ensure_scatter3_host_metadata(
    style: &mut Scatter3ResolvedStyle,
    point_count: usize,
) -> Result<(), String> {
    if style.per_point_sizes.is_none() {
        if let Some(handle) = style.gpu_sizes.clone() {
            let tensor = gather_tensor_from_gpu(handle, "scatter3")?;
            let value = Value::Tensor(tensor);
            style.per_point_sizes = Some(convert_size_vector(&value, point_count, "scatter3")?);
        }
    }
    if style.per_point_colors.is_none() {
        if let Some(gpu_color) = style.gpu_colors.as_ref() {
            let tensor = gather_tensor_from_gpu(gpu_color.handle.clone(), "scatter3")?;
            let value = Value::Tensor(tensor);
            style.per_point_colors =
                Some(convert_rgb_color_matrix(&value, point_count, "scatter3")?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::style::LineStyleParseOptions;
    use super::*;
    use runmat_builtins::Value;
    #[ctor::ctor]
    fn init_plot_test_env() {
        crate::builtins::plotting::state::disable_rendering_for_tests();
    }

    fn test_style() -> Scatter3ResolvedStyle {
        Scatter3ResolvedStyle {
            uniform_color: default_color(),
            point_size: DEFAULT_POINT_SIZE,
            per_point_sizes: None,
            per_point_colors: None,
            color_values: None,
            color_limits: None,
            gpu_sizes: None,
            gpu_colors: None,
            colormap: ColorMap::Parula,
            requires_cpu: false,
            label: DEFAULT_SCATTER3_LABEL.to_string(),
        }
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

    fn assert_plotting_unavailable(msg: &str) {
        let lower = msg.to_lowercase();
        assert!(
            lower.contains("plotting is unavailable") || lower.contains("non-main thread"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn build_scatter3_requires_equal_lengths() {
        assert!(build_scatter3_plot(vec![1.0], vec![], vec![1.0], &mut test_style()).is_err());
    }

    #[test]
    fn scatter3_builtin_emits_result_or_backend_error() {
        let out = scatter3_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Vec::new(),
        );
        if let Err(msg) = out {
            assert_plotting_unavailable(&msg);
        }
    }

    #[test]
    fn scatter3_accepts_per_point_sizes() {
        let rest = vec![Value::Tensor(Tensor {
            data: vec![1.0, 2.0],
            shape: vec![2],
            rows: 2,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        })];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter3()).unwrap();
        let style = resolve_scatter3_style(2, &args, "scatter3").expect("style");
        assert!(style.per_point_sizes.is_some());
    }

    #[test]
    fn scatter3_applies_display_name() {
        let rest = vec![
            Value::String("DisplayName".into()),
            Value::String("Cloud A".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter3()).unwrap();
        let mut style = resolve_scatter3_style(2, &args, "scatter3").expect("style");
        let plot = build_scatter3_plot(vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0], &mut style)
            .expect("plot");
        assert_eq!(plot.label.as_deref(), Some("Cloud A"));
    }
}
