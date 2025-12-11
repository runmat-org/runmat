//! MATLAB-compatible `mesh` builtin.

use log::warn;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::common::{numeric_vector, tensor_to_surface_grid, SurfaceDataInput};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use super::surf::build_surface_gpu_plot;
use std::sync::Arc;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "mesh",
        wasm_path = "crate::builtins::plotting::ops::mesh"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "mesh"
category: "plotting"
keywords: ["mesh", "wireframe", "3-D plotting", "gpuArray"]
summary: "Render MATLAB-compatible mesh (wireframe) plots."
references:
  - https://www.mathworks.com/help/matlab/ref/mesh.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["single"]
  broadcasting: "none"
  notes: "Single-precision gpuArray height maps stream directly into the shared WebGPU renderer; other inputs gather before plotting."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::plotting::mesh::tests"
---

# What does `mesh` do?
`mesh(X, Y, Z)` draws a wireframe surface. RunMat reuses the `SurfacePlot` renderer with wireframe
mode enabled and no fill, matching MATLAB's default mesh aesthetics.

## Behaviour highlights
- `X` and `Y` are axis vectors; `Z` must contain `length(X) * length(Y)` values in column-major order.
- Surfaces default to a Turbo colormap with `wireframe = true` and faceted shading.
- Single-precision gpuArray height maps stream directly into the shared WebGPU renderer; other inputs gather automatically.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::plotting::ops::mesh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mesh",
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
    notes: "Wireframe rendering happens on the host/WebGPU path.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::plotting::ops::mesh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mesh",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "mesh terminates fusion graphs.",
};

#[runtime_builtin(
    name = "mesh",
    category = "plotting",
    summary = "Render a MATLAB-compatible wireframe surface.",
    keywords = "mesh,wireframe,surface,plotting",
    sink = true,
    wasm_path = "crate::builtins::plotting::ops::mesh"
)]
pub fn mesh_builtin(x: Tensor, y: Tensor, z: Value, rest: Vec<Value>) -> Result<String, String> {
    let x_axis = numeric_vector(x);
    let y_axis = numeric_vector(y);
    let mut x_axis = Some(x_axis);
    let mut y_axis = Some(y_axis);
    let mut z_input = Some(SurfaceDataInput::from_value(z, "mesh")?);
    let style = Arc::new(parse_surface_style_args(
        "mesh",
        &rest,
        SurfaceStyleDefaults::new(
            ColorMap::Turbo,
            ShadingMode::Faceted,
            true,
            1.0,
            false,
            true,
        ),
    )?);
    let opts = PlotRenderOptions {
        title: "Mesh Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: false,
        ..Default::default()
    };
    render_active_plot(opts, move |figure, axes| {
        let x_axis_vec = x_axis.take().expect("mesh data consumed once");
        let y_axis_vec = y_axis.take().expect("mesh data consumed once");
        let z_arg = z_input.take().expect("mesh data consumed once");

        if let Some(z_gpu) = z_arg.gpu_handle() {
            let style = Arc::clone(&style);
            match build_surface_gpu_plot(&x_axis_vec, &y_axis_vec, z_gpu) {
                Ok(surface_gpu) => {
                    let mut surface = surface_gpu
                        .with_colormap(ColorMap::Turbo)
                        .with_wireframe(true)
                        .with_shading(ShadingMode::Faceted);
                    style.apply_to_plot(&mut surface);
                    figure.add_surface_plot_on_axes(surface, axes);
                    return Ok(());
                }
                Err(err) => {
                    warn!("mesh GPU path unavailable: {err}");
                }
            }
        }

        let grid = tensor_to_surface_grid(
            z_arg.into_tensor("mesh")?,
            x_axis_vec.len(),
            y_axis_vec.len(),
        )?;
        let mut surface = build_mesh_surface(x_axis_vec, y_axis_vec, grid)?;
        let style = Arc::clone(&style);
        style.apply_to_plot(&mut surface);
        figure.add_surface_plot_on_axes(surface, axes);
        Ok(())
    })
}

pub(crate) fn build_mesh_surface(
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    z_grid: Vec<Vec<f64>>,
) -> Result<SurfacePlot, String> {
    if x_axis.is_empty() || y_axis.is_empty() {
        return Err("mesh: axis vectors must be non-empty".to_string());
    }

    let surface = SurfacePlot::new(x_axis, y_axis, z_grid)
        .map_err(|err| format!("mesh: {err}"))?
        .with_colormap(ColorMap::Turbo)
        .with_wireframe(true)
        .with_shading(ShadingMode::Faceted);
    Ok(surface)
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
    fn mesh_requires_matching_grid() {
        let res = mesh_builtin(
            tensor_from(&[0.0]),
            tensor_from(&[0.0, 1.0]),
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
