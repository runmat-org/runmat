//! MATLAB-compatible `scatter` builtin.

use glam::{Vec3, Vec4};
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::scatter2::{
    Scatter2GpuInputs, Scatter2GpuParams, ScatterAttributeBuffer, ScatterColorBuffer,
};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::scatter::{MarkerStyle, ScatterGpuStyle};
use runmat_plot::plots::surface::ColorMap;
use runmat_plot::plots::LineStyle;
use runmat_plot::plots::ScatterPlot;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;
use std::convert::TryFrom;

use super::common::numeric_pair;
use super::gpu_helpers::axis_bounds;
use super::perf::scatter_target_points;
use super::point::{
    convert_rgb_color_matrix, convert_scalar_color_values, convert_size_vector,
    map_scalar_values_to_colors, validate_gpu_color_matrix, validate_gpu_vector_length, PointArgs,
    PointColorArg, PointGpuColor, PointSizeArg,
};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{LineStyleParseOptions, MarkerColor};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "scatter")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "scatter"
category: "plotting"
keywords: ["scatter", "2-D scatter", "marker plot", "gpuArray"]
summary: "Render MATLAB-compatible 2-D scatter plots."
references:
  - https://www.mathworks.com/help/matlab/ref/scatter.html
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
  unit: "builtins::plotting::scatter::tests"
---

# What does `scatter` do?
`scatter(x, y)` plots paired points in 2-D space. The RunMat implementation mirrors MATLAB's default
styling—circular markers, grid enabled, and axis labels applied to the active figure. Positional size/color
arguments (`S`, `C`), the `'filled'` flag, and name-value pairs for marker appearance are supported.

## Behaviour highlights
- Inputs must contain the same number of elements. Row and column vectors are both accepted.
- Single-precision gpuArray inputs stay on the device when the shared WebGPU renderer is active,
  so browser and native builds avoid a gather step. Other tensors automatically fall back to the
  host path.
- Positional size/color inputs plus `'Marker*'` name-value pairs match MATLAB semantics. Per-point
  size/color vectors currently force the CPU path until the scatter GPU shaders consume those
  attributes directly (tracked separately).

## Examples
```matlab
t = linspace(0, 2*pi, 100);
scatter(cos(t), sin(t));
```

## GPU residency
`scatter` terminates fusion graphs. If the inputs are `single` gpuArrays and the shared plotter
device is active, their buffers are consumed zero-copy by the renderer. Otherwise the tensors are
gathered before plotting, matching MATLAB semantics.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "scatter",
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
    notes: "2-D scatter rendering happens outside fusion; tensors are gathered first.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "scatter",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "scatter performs I/O and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "scatter",
    category = "plotting",
    summary = "Create MATLAB-compatible 2-D scatter plots.",
    keywords = "scatter,plotting,2d,markers",
    sink = true
)]
pub fn scatter_builtin(x: Value, y: Value, rest: Vec<Value>) -> Result<String, String> {
    let style_args = PointArgs::parse(rest, LineStyleParseOptions::scatter())?;
    let mut x_input = Some(ScatterInput::from_value(x)?);
    let mut y_input = Some(ScatterInput::from_value(y)?);
    let opts = PlotRenderOptions {
        title: "Scatter Plot",
        x_label: "X",
        y_label: "Y",
        ..Default::default()
    };
    render_active_plot(opts, move |figure, axes| {
        let style_args = style_args.clone();
        let point_count = x_input.as_ref().map(|input| input.len()).unwrap_or(0);
        let mut resolved_style = resolve_scatter_style(point_count, &style_args, "scatter")?;
        let x_arg = x_input.take().expect("scatter x consumed once");
        let y_arg = y_input.take().expect("scatter y consumed once");

        if !resolved_style.requires_cpu {
            if let (Some(x_gpu), Some(y_gpu)) = (x_arg.gpu_handle(), y_arg.gpu_handle()) {
                match build_scatter_gpu_plot(x_gpu, y_gpu, &resolved_style) {
                    Ok(plot) => {
                        figure.add_scatter_plot_on_axes(plot, axes);
                        return Ok(());
                    }
                    Err(err) => {
                        warn!("scatter GPU path unavailable: {err}");
                    }
                }
            }
        }

        let (x_tensor, y_tensor) = (x_arg.into_tensor("scatter")?, y_arg.into_tensor("scatter")?);
        let (x_data, y_data) = numeric_pair(x_tensor, y_tensor, "scatter")?;
        let scatter = build_scatter_plot(x_data, y_data, &mut resolved_style)?;
        figure.add_scatter_plot_on_axes(scatter, axes);
        Ok(())
    })
}

fn build_scatter_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    style: &mut ScatterResolvedStyle,
) -> Result<ScatterPlot, String> {
    if x.len() != y.len() {
        return Err("scatter: X and Y inputs must share the same length".to_string());
    }
    if x.is_empty() {
        return Err("scatter: inputs cannot be empty".to_string());
    }

    ensure_host_marker_metadata(style, x.len())?;

    let mut scatter = ScatterPlot::new(x, y)
        .map_err(|err| format!("scatter: {err}"))?
        .with_style(style.uniform_color, style.marker_size, style.marker_style)
        .with_label(style.label.clone());
    scatter.colormap = style.colormap;
    scatter.set_edge_color(style.edge_color);
    scatter.set_edge_thickness(style.edge_thickness);
    scatter.set_filled(style.filled);
    scatter.set_edge_color_from_vertex(style.marker_edge_flat);

    if let Some(sizes) = style.per_point_sizes.take() {
        scatter.set_sizes(sizes);
    }
    if let Some(colors) = style.per_point_colors.take() {
        scatter.set_colors(colors);
    }
    if let Some(values) = style.color_values.take() {
        scatter.set_color_values(values, style.color_limits.take());
    }
    Ok(scatter)
}

const DEFAULT_MARKER_SIZE: f32 = 10.0;
const DEFAULT_LINE_WIDTH: f32 = 1.0;
const DEFAULT_SCATTER_LABEL: &str = "Data";

fn default_color() -> Vec4 {
    Vec4::new(0.85, 0.2, 0.2, 0.95)
}

#[derive(Clone, Debug)]
struct ScatterResolvedStyle {
    uniform_color: Vec4,
    edge_color: Vec4,
    edge_thickness: f32,
    marker_style: MarkerStyle,
    marker_size: f32,
    filled: bool,
    per_point_sizes: Option<Vec<f32>>,
    per_point_colors: Option<Vec<Vec4>>,
    color_values: Option<Vec<f64>>,
    color_limits: Option<(f64, f64)>,
    gpu_sizes: Option<GpuTensorHandle>,
    gpu_colors: Option<PointGpuColor>,
    colormap: ColorMap,
    marker_face_flat: bool,
    marker_edge_flat: bool,
    requires_cpu: bool,
    label: String,
}

fn resolve_scatter_style(
    point_count: usize,
    args: &PointArgs,
    context: &str,
) -> Result<ScatterResolvedStyle, String> {
    let mut style = ScatterResolvedStyle {
        uniform_color: default_color(),
        edge_color: default_color(),
        edge_thickness: DEFAULT_LINE_WIDTH,
        marker_style: MarkerStyle::Circle,
        marker_size: DEFAULT_MARKER_SIZE,
        filled: args.filled,
        per_point_sizes: None,
        per_point_colors: None,
        color_values: None,
        color_limits: None,
        gpu_sizes: None,
        gpu_colors: None,
        colormap: ColorMap::Parula,
        marker_face_flat: false,
        marker_edge_flat: false,
        requires_cpu: false,
        label: DEFAULT_SCATTER_LABEL.to_string(),
    };

    if let Some(label) = args.style.label.clone() {
        style.label = label;
    }

    let appearance = &args.style.appearance;
    style.uniform_color = appearance.color;
    style.edge_color = appearance.color;
    style.edge_thickness = appearance.line_width.max(0.1);

    if let PointColorArg::Uniform(color) = &args.color {
        style.uniform_color = *color;
    }

    if appearance.marker.is_none() {
        style.edge_color = style.uniform_color;
    }

    if let Some(marker) = appearance.marker.as_ref() {
        style.marker_style = marker.kind.to_plot_marker();
        if let Some(size) = marker.size {
            style.marker_size = size.max(0.1);
        }
        if matches!(marker.edge_color, MarkerColor::Flat) {
            style.marker_edge_flat = true;
        } else {
            style.edge_color =
                resolve_marker_color(&marker.edge_color, style.edge_color, style.uniform_color);
        }
        match &marker.face_color {
            MarkerColor::Flat => {
                style.marker_face_flat = true;
                style.filled = true;
            }
            MarkerColor::None => {
                style.filled = false;
            }
            _ => {
                let face_color = resolve_marker_color(
                    &marker.face_color,
                    style.uniform_color,
                    style.uniform_color,
                );
                if matches!(marker.face_color, MarkerColor::Color(_) | MarkerColor::Auto) {
                    style.uniform_color = face_color;
                }
            }
        }
    }

    if let PointSizeArg::Scalar(size) = &args.size {
        style.marker_size = (*size).max(0.1);
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

    if style.per_point_colors.is_some() || style.color_values.is_some() {
        style.filled = true;
    }

    if style.marker_face_flat {
        if style.per_point_colors.is_none() && style.gpu_colors.is_none() {
            return Err(format!(
                "{context}: MarkerFaceColor 'flat' requires per-point color data (C argument)"
            ));
        }
        style.filled = true;
    }

    if style.marker_edge_flat && style.per_point_colors.is_none() && style.gpu_colors.is_none() {
        return Err(format!(
            "{context}: MarkerEdgeColor 'flat' requires per-point color data (C argument)"
        ));
    }

    if args.style.appearance.line_style != LineStyle::Solid && args.style.line_style_explicit {
        style.requires_cpu = true;
    }
    style.requires_cpu |= args.style.requires_cpu_fallback;
    if args.style.line_style_order.is_some() {
        style.requires_cpu = true;
    }

    Ok(style)
}

fn resolve_marker_color(marker_color: &MarkerColor, fallback: Vec4, default_base: Vec4) -> Vec4 {
    match marker_color {
        MarkerColor::Auto => fallback,
        MarkerColor::None => Vec4::new(default_base.x, default_base.y, default_base.z, 0.0),
        MarkerColor::Flat => fallback,
        MarkerColor::Color(color) => *color,
    }
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
                let tensor = Tensor::try_from(&other).map_err(|e| format!("scatter: {e}"))?;
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
            Self::Host(tensor) => Ok(tensor),
            Self::Gpu(handle) => gather_tensor_from_gpu(handle, name),
        }
    }
}

fn gather_tensor_from_gpu(handle: GpuTensorHandle, name: &str) -> Result<Tensor, String> {
    let value = Value::GpuTensor(handle);
    let gathered = gather_if_needed(&value)?;
    Tensor::try_from(&gathered).map_err(|e| format!("{name}: {e}"))
}

fn build_scatter_gpu_plot(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    style: &ScatterResolvedStyle,
) -> Result<ScatterPlot, String> {
    let context = runmat_plot::shared_wgpu_context()
        .ok_or_else(|| "scatter: plotting GPU context unavailable".to_string())?;

    let x_ref = runmat_accelerate_api::export_wgpu_buffer(x)
        .ok_or_else(|| "scatter: unable to export GPU X data".to_string())?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y)
        .ok_or_else(|| "scatter: unable to export GPU Y data".to_string())?;

    if x_ref.len == 0 {
        return Err("scatter: empty input tensor".to_string());
    }
    if x_ref.len != y_ref.len {
        return Err("scatter: X and Y inputs must have identical lengths".to_string());
    }
    if x_ref.precision != y_ref.precision {
        return Err("scatter: X and Y gpuArrays must have matching precision".to_string());
    }
    let point_count = x_ref.len;
    let len_u32 = u32::try_from(point_count)
        .map_err(|_| "scatter: point count exceeds supported range".to_string())?;
    let scalar = ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64);
    let lod_stride = scatter_lod_stride(len_u32);

    let size_buffer = build_size_buffer(style, point_count)?;
    let has_sizes = size_buffer.has_data();
    let color_buffer = build_color_buffer(style, point_count)?;
    let has_colors = color_buffer.has_data();

    let inputs = Scatter2GpuInputs {
        x_buffer: x_ref.buffer.clone(),
        y_buffer: y_ref.buffer.clone(),
        len: len_u32,
        scalar,
    };
    let params = Scatter2GpuParams {
        color: style.uniform_color,
        point_size: style.marker_size,
        sizes: size_buffer,
        colors: color_buffer,
        lod_stride,
    };

    let gpu_vertices = runmat_plot::gpu::scatter2::pack_vertices_from_xy(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| format!("scatter: failed to build GPU vertices: {e}"))?;
    let drawn_points = gpu_vertices.vertex_count;

    let bounds = build_gpu_bounds(x, y)?;
    let gpu_style = ScatterGpuStyle {
        color: style.uniform_color,
        edge_color: style.edge_color,
        edge_thickness: style.edge_thickness,
        marker_size: style.marker_size,
        marker_style: style.marker_style,
        filled: style.filled,
        has_per_point_sizes: has_sizes,
        has_per_point_colors: has_colors,
        edge_from_vertex_colors: style.marker_edge_flat,
    };

    let mut scatter = ScatterPlot::from_gpu_buffer(gpu_vertices, drawn_points, bounds, gpu_style)
        .with_label(style.label.clone());
    scatter.colormap = style.colormap;
    scatter.set_edge_color_from_vertex(style.marker_edge_flat);
    if lod_stride == 1 {
        if let Some(values) = style.color_values.as_ref() {
            scatter.color_values = Some(values.clone());
        }
    }
    scatter.color_limits = style.color_limits;
    Ok(scatter)
}

fn scatter_lod_stride(point_count: u32) -> u32 {
    let target = scatter_target_points().max(1);
    if point_count <= target {
        1
    } else {
        (point_count + target - 1) / target
    }
}

fn build_gpu_bounds(x: &GpuTensorHandle, y: &GpuTensorHandle) -> Result<BoundingBox, String> {
    let (min_x, max_x) = axis_bounds(x, "scatter")?;
    let (min_y, max_y) = axis_bounds(y, "scatter")?;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, 0.0),
        Vec3::new(max_x, max_y, 0.0),
    ))
}

fn build_size_buffer(
    style: &ScatterResolvedStyle,
    point_count: usize,
) -> Result<ScatterAttributeBuffer, String> {
    if let Some(handle) = style.gpu_sizes.as_ref() {
        let exported = runmat_accelerate_api::export_wgpu_buffer(handle)
            .ok_or_else(|| "scatter: unable to export GPU marker sizes".to_string())?;
        if exported.len != point_count {
            return Err(format!(
                "scatter: marker size array must have {point_count} elements (got {})",
                exported.len
            ));
        }
        if exported.precision != ProviderPrecision::F32 {
            return Err(
                "scatter: GPU marker sizes must be single-precision (cast before plotting)"
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

fn build_color_buffer(
    style: &ScatterResolvedStyle,
    point_count: usize,
) -> Result<ScatterColorBuffer, String> {
    if let Some(gpu_color) = style.gpu_colors.as_ref() {
        let exported = runmat_accelerate_api::export_wgpu_buffer(&gpu_color.handle)
            .ok_or_else(|| "scatter: unable to export GPU color data".to_string())?;
        let expected = point_count * gpu_color.components.stride() as usize;
        if exported.len != expected {
            return Err(format!(
                "scatter: color array must contain {} elements (got {})",
                expected, exported.len
            ));
        }
        if exported.precision != ProviderPrecision::F32 {
            return Err(
                "scatter: GPU color arrays must be single-precision (cast before plotting)"
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

fn ensure_host_marker_metadata(
    style: &mut ScatterResolvedStyle,
    point_count: usize,
) -> Result<(), String> {
    if style.per_point_sizes.is_none() {
        if let Some(handle) = style.gpu_sizes.clone() {
            let tensor = gather_tensor_from_gpu(handle, "scatter")?;
            let value = Value::Tensor(tensor);
            style.per_point_sizes = Some(convert_size_vector(&value, point_count, "scatter")?);
        }
    }
    if style.per_point_colors.is_none() {
        if let Some(gpu_color) = style.gpu_colors.as_ref() {
            let tensor = gather_tensor_from_gpu(gpu_color.handle.clone(), "scatter")?;
            let value = Value::Tensor(tensor);
            style.per_point_colors =
                Some(convert_rgb_color_matrix(&value, point_count, "scatter")?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::point::{PointArgs, PointColorArg, PointSizeArg};
    use super::super::style::{
        LineAppearance, LineStyleParseOptions, MarkerAppearance, MarkerColor, MarkerKind,
        ParsedLineStyle,
    };
    use super::*;
    use runmat_builtins::Value;
    #[ctor::ctor]
    fn init_plot_test_env() {
        crate::builtins::plotting::state::disable_rendering_for_tests();
    }

    fn test_style() -> ScatterResolvedStyle {
        ScatterResolvedStyle {
            uniform_color: default_color(),
            edge_color: default_color(),
            edge_thickness: DEFAULT_LINE_WIDTH,
            marker_style: MarkerStyle::Circle,
            marker_size: DEFAULT_MARKER_SIZE,
            filled: false,
            per_point_sizes: None,
            per_point_colors: None,
            color_values: None,
            color_limits: None,
            gpu_sizes: None,
            gpu_colors: None,
            colormap: ColorMap::Parula,
            marker_face_flat: false,
            marker_edge_flat: false,
            requires_cpu: false,
            label: DEFAULT_SCATTER_LABEL.to_string(),
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
    fn scatter_requires_equal_lengths() {
        assert!(build_scatter_plot(vec![1.0], vec![], &mut test_style()).is_err());
    }

    #[test]
    fn scatter_builtin_returns_or_reports_backend_status() {
        let out = scatter_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Vec::new(),
        );
        if let Err(msg) = out {
            assert_plotting_unavailable(&msg);
        }
    }

    #[test]
    fn scatter_resolves_marker_style_from_arguments() {
        let rest = vec![
            Value::Num(12.0),
            Value::String("r".into()),
            Value::String("filled".into()),
            Value::String("Marker".into()),
            Value::String("s".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter()).unwrap();
        let style = resolve_scatter_style(3, &args, "scatter").expect("style");
        assert!(style.filled);
        assert_eq!(style.marker_style, MarkerStyle::Square);
        assert_eq!(style.marker_size as i32, 12);
    }

    #[test]
    fn scatter_supports_flat_marker_face_color() {
        let rest = vec![
            Value::Tensor(tensor_from(&[5.0, 5.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::String("Marker".into()),
            Value::String("o".into()),
            Value::String("MarkerFaceColor".into()),
            Value::String("flat".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter()).unwrap();
        let style = resolve_scatter_style(2, &args, "scatter").expect("style");
        assert!(style.marker_face_flat);
        assert!(style.per_point_colors.is_some() || style.gpu_colors.is_some());
    }

    #[test]
    fn scatter_applies_display_name() {
        let rest = vec![
            Value::String("DisplayName".into()),
            Value::String("Series A".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter()).unwrap();
        let mut style = resolve_scatter_style(2, &args, "scatter").expect("style");
        let plot = build_scatter_plot(vec![0.0, 1.0], vec![0.0, 1.0], &mut style).expect("plot");
        assert_eq!(plot.label.as_deref(), Some("Series A"));
    }

    #[test]
    fn scatter_rejects_flat_marker_edge_color() {
        let rest = vec![
            Value::Tensor(tensor_from(&[5.0, 5.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::String("Marker".into()),
            Value::String("o".into()),
            Value::String("MarkerEdgeColor".into()),
            Value::String("flat".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter()).unwrap();
        let err = resolve_scatter_style(2, &args, "scatter").unwrap_err();
        assert!(err.contains("requires per-point color data"));
    }

    #[test]
    fn scatter_accepts_flat_marker_edge_color_when_colors_supplied() {
        let mut appearance = LineAppearance::default();
        appearance.marker = Some(MarkerAppearance {
            kind: MarkerKind::Circle,
            size: None,
            edge_color: MarkerColor::Flat,
            face_color: MarkerColor::Auto,
        });
        let args = PointArgs {
            size: PointSizeArg::Default,
            color: PointColorArg::ScalarValues(Value::Tensor(tensor_from(&[0.1, 0.5]))),
            filled: false,
            style: ParsedLineStyle {
                appearance,
                requires_cpu_fallback: false,
                line_style_explicit: false,
                line_style_order: None,
                label: None,
            },
        };
        let mut style = resolve_scatter_style(2, &args, "scatter").expect("style");
        assert!(style.marker_edge_flat);
        let plot = build_scatter_plot(vec![0.0, 1.0], vec![0.0, 1.0], &mut style).expect("plot");
        assert!(plot.edge_color_from_vertex_colors);
    }
}
