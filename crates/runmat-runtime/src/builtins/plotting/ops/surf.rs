//! MATLAB-compatible `surf` builtin.

use glam::Vec3;
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::{numeric_vector, tensor_to_surface_grid, SurfaceDataInput};
use super::gpu_helpers::axis_bounds;
use super::perf::compute_surface_lod;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use std::convert::TryFrom;
use std::sync::Arc;

use crate::BuiltinResult;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "surf",
        builtin_path = "crate::builtins::plotting::surf"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
const BUILTIN_NAME: &str = "surf";

#[allow(dead_code)]
pub const DOC_MD: &str = r#"---
title: "surf"
category: "plotting"
keywords: ["surf", "surface plot", "3-D plotting", "gpuArray"]
summary: "Render MATLAB-compatible 3-D surface plots."
references:
  - https://www.mathworks.com/help/matlab/ref/surf.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Single-precision gpuArray height maps render zero-copy via the shared WebGPU context; other inputs are gathered before plotting."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::plotting::surf::tests"
---

# What does `surf` do?
`surf(X, Y, Z)` draws a shaded surface where `X` and `Y` provide axis coordinates and `Z` provides
heights. RunMat expects `X` and `Y` to be vectors defining the grid axes (rows × columns), and `Z`
to contain `numel(X) * numel(Y)` elements stored in column-major order, matching MATLAB tensors.
When `Z` is a single-precision gpuArray and the shared WebGPU renderer is active, the surface
geometry stays on the device and feeds a compute shader that emits renderer-ready vertices.

## Behaviour highlights
- Axis vectors must be non-empty and `Z` must contain exactly `length(X) * length(Y)` elements.
- Single-precision gpuArray height maps stream directly into the renderer; other precisions gather
  to host memory before plotting.
- Surfaces default to the Parula colormap with smooth shading and lighting enabled.

## Examples
```matlab
x = linspace(-2, 2, 50);
y = linspace(-2, 2, 50);
z = meshgrid(x, y);
surf(x, y, sin(x)' * cos(y));
```
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::surf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "surf",
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
    notes: "Surface rendering runs on the host/WebGPU pipeline; single-precision gpuArray inputs stay on device.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::surf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "surf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "surf terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "surf",
    category = "plotting",
    summary = "Render a MATLAB-compatible surface plot.",
    keywords = "surf,plotting,3d,surface",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::surf"
)]
pub fn surf_builtin(x: Value, y: Value, z: Value, rest: Vec<Value>) -> crate::BuiltinResult<String> {
    let x_tensor = Tensor::try_from(&x)
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("surf: {e}")))?;
    let y_tensor = Tensor::try_from(&y)
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("surf: {e}")))?;
    let x_axis = numeric_vector(x_tensor);
    let y_axis = numeric_vector(y_tensor);
    let mut x_axis = Some(x_axis);
    let mut y_axis = Some(y_axis);
    let mut z_input = Some(SurfaceDataInput::from_value(z, "surf")?);
    let style = Arc::new(parse_surface_style_args(
        "surf",
        &rest,
        SurfaceStyleDefaults::new(
            ColorMap::Parula,
            ShadingMode::Smooth,
            false,
            1.0,
            false,
            true,
        ),
    )?);
    let opts = PlotRenderOptions {
        title: "Surface Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: false,
        ..Default::default()
    };
    let rendered = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let x_axis_vec = x_axis.take().expect("surf: X axis consumed once");
        let y_axis_vec = y_axis.take().expect("surf: Y axis consumed once");
        let z_arg = z_input.take().expect("surf: Z consumed once");

        if let Some(z_gpu) = z_arg.gpu_handle() {
            let style = Arc::clone(&style);
            match build_surface_gpu_plot(BUILTIN_NAME, &x_axis_vec, &y_axis_vec, z_gpu) {
                Ok(mut surface) => {
                    style.apply_to_plot(&mut surface);
                    figure.add_surface_plot_on_axes(surface, axes);
                    return Ok(());
                }
                Err(err) => {
                    warn!("surf GPU path unavailable: {err}");
                }
            }
        }

        let grid = tensor_to_surface_grid(
            z_arg.into_tensor(BUILTIN_NAME)?,
            x_axis_vec.len(),
            y_axis_vec.len(),
            BUILTIN_NAME,
        )?;
        let mut surface = build_surface(x_axis_vec, y_axis_vec, grid)?;
        let style = Arc::clone(&style);
        style.apply_to_plot(&mut surface);
        figure.add_surface_plot_on_axes(surface, axes);
        Ok(())
    })?;
    Ok(rendered)
}

pub(crate) fn build_surface(
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    z_grid: Vec<Vec<f64>>,
) -> BuiltinResult<SurfacePlot> {
    if x_axis.is_empty() || y_axis.is_empty() {
        return Err(plotting_error(
            BUILTIN_NAME,
            "surf: axis vectors must be non-empty",
        ));
    }

    let surface = SurfacePlot::new(x_axis, y_axis, z_grid)
        .map_err(|err| plotting_error(BUILTIN_NAME, format!("surf: {err}")))?
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::Smooth);
    Ok(surface)
}

pub(crate) fn build_surface_gpu_plot(
    name: &'static str,
    x_axis: &[f64],
    y_axis: &[f64],
    z: &GpuTensorHandle,
) -> BuiltinResult<SurfacePlot> {
    if x_axis.is_empty() || y_axis.is_empty() {
        return Err(plotting_error(name, format!("{name}: axis vectors must be non-empty")));
    }

    let context = runmat_plot::shared_wgpu_context().ok_or_else(|| {
        plotting_error(name, format!("{name}: plotting GPU context unavailable"))
    })?;

    let z_ref = runmat_accelerate_api::export_wgpu_buffer(z).ok_or_else(|| {
        plotting_error(name, format!("{name}: unable to export GPU Z data"))
    })?;

    let expected_len = x_axis
        .len()
        .checked_mul(y_axis.len())
        .ok_or_else(|| plotting_error(name, format!("{name}: grid dimensions overflowed")))?;
    if z_ref.len as usize != expected_len {
        return Err(plotting_error(
            name,
            format!(
                "{name}: Z must contain exactly {} elements ({}×{})",
                expected_len,
                x_axis.len(),
                y_axis.len()
            ),
        ));
    }
    let (min_z, max_z) = axis_bounds(z, name)?;
    let min_x = x_axis
        .iter()
        .fold(f32::INFINITY, |acc, &val| acc.min(val as f32));
    let max_x = x_axis
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &val| acc.max(val as f32));
    let min_y = y_axis
        .iter()
        .fold(f32::INFINITY, |acc, &val| acc.min(val as f32));
    let max_y = y_axis
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &val| acc.max(val as f32));
    let bounds = runmat_plot::core::scene::BoundingBox::new(
        Vec3::new(min_x, min_y, min_z),
        Vec3::new(max_x, max_y, max_z),
    );
    let extent_hint = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();

    let x_axis_f32: Vec<f32> = x_axis.iter().map(|&v| v as f32).collect();
    let y_axis_f32: Vec<f32> = y_axis.iter().map(|&v| v as f32).collect();
    let color_table = build_color_lut(ColorMap::Parula, 512, 1.0);

    let inputs = runmat_plot::gpu::surface::SurfaceGpuInputs {
        x_axis: &x_axis_f32,
        y_axis: &y_axis_f32,
        z_buffer: z_ref.buffer.clone(),
        color_table: &color_table,
        x_len: x_axis.len() as u32,
        y_len: y_axis.len() as u32,
        scalar: ScalarType::from_is_f64(z_ref.precision == ProviderPrecision::F64),
    };
    let lod = compute_surface_lod(x_axis.len(), y_axis.len(), extent_hint);
    let params = runmat_plot::gpu::surface::SurfaceGpuParams {
        min_z,
        max_z,
        alpha: 1.0,
        flatten_z: false,
        x_stride: lod.stride_x,
        y_stride: lod.stride_y,
        lod_x_len: lod.lod_x_len,
        lod_y_len: lod.lod_y_len,
    };

    let gpu_vertices = runmat_plot::gpu::surface::pack_surface_vertices(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| plotting_error(name, format!("{name}: failed to build GPU vertices: {e}")))?;

    let vertex_count = lod.vertex_count();
    let mut surface = SurfacePlot::from_gpu_buffer(
        x_axis.to_vec(),
        y_axis.to_vec(),
        gpu_vertices,
        vertex_count,
        bounds,
    );
    surface.colormap = ColorMap::Parula;
    surface.shading_mode = ShadingMode::Smooth;
    Ok(surface)
}

pub(crate) fn build_color_lut(colormap: ColorMap, samples: usize, alpha: f32) -> Vec<[f32; 4]> {
    let clamped = samples.max(2);
    (0..clamped)
        .map(|i| {
            let t = i as f32 / (clamped as f32 - 1.0);
            let rgb = colormap.map_value(t);
            [rgb.x, rgb.y, rgb.z, alpha]
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;

    fn setup_plot_tests() {
        ensure_plot_test_env();
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn surf_requires_matching_grid() {
        setup_plot_tests();
        let res = surf_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0])),
            Value::Tensor(Tensor {
                data: vec![0.0],
                shape: vec![1],
                rows: 1,
                cols: 1,
                dtype: runmat_builtins::NumericDType::F64,
            }),
            Vec::new(),
        );
        assert!(res.is_err());
    }
}
