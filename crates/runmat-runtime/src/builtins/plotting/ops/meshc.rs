//! MATLAB-compatible `meshc` builtin (mesh with contour).

use log::warn;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};

use super::common::{numeric_vector, tensor_to_surface_grid, SurfaceDataInput};
use super::contour::{
    build_contour_gpu_plot, build_contour_plot, default_level_count, ContourLevelSpec,
};
use super::mesh::build_mesh_surface;
use super::state::{render_active_plot, PlotRenderOptions};
use super::surf::build_surface_gpu_plot;

#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "meshc"
category: "plotting"
keywords: ["meshc", "mesh", "contour", "gpuArray"]
summary: "Render a wireframe surface with contour lines projected onto the base plane."
references:
  - https://www.mathworks.com/help/matlab/ref/meshc.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["single"]
  broadcasting: "none"
  notes: "Single-precision gpuArrays reuse the shared WebGPU context; other inputs gather."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::plotting::meshc::tests"
---

# What does `meshc` do?
`meshc(X, Y, Z)` draws a wireframe mesh and overlays contour lines on the XY plane.

## GPU behaviour
- Single-precision gpuArrays stream directly into the shared WebGPU renderer.
- Double precision falls back to host gathers until SHADER_F64 is available.
"#;

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("meshc", DOC_MD);

#[runtime_builtin(
    name = "meshc",
    category = "plotting",
    summary = "Render a MATLAB-compatible mesh with contour overlay.",
    keywords = "meshc,plotting,mesh,contour",
    sink = true
)]
pub fn meshc_builtin(x: Tensor, y: Tensor, z: Value) -> Result<String, String> {
    let x_axis = numeric_vector(x);
    let y_axis = numeric_vector(y);
    let mut x_axis = Some(x_axis);
    let mut y_axis = Some(y_axis);
    let mut z_input = Some(SurfaceDataInput::from_value(z, "meshc")?);
    let opts = PlotRenderOptions {
        title: "Mesh with Contours",
        x_label: "X",
        y_label: "Y",
        axis_equal: false,
        ..Default::default()
    };
    let level_spec = ContourLevelSpec::Count(default_level_count());
    render_active_plot(opts, move |figure, axes| {
        let level_spec = level_spec.clone();
        let x_vec = x_axis.take().expect("meshc X consumed once");
        let y_vec = y_axis.take().expect("meshc Y consumed once");
        let z_arg = z_input.take().expect("meshc Z consumed once");

        if let Some(z_gpu) = z_arg.gpu_handle() {
            match build_surface_gpu_plot(&x_vec, &y_vec, z_gpu) {
                Ok(surface_gpu) => {
                    let mut surface = surface_gpu.with_wireframe(true);
                    surface.shading_mode = ShadingMode::Faceted;
                    let base_z = surface.bounds().min.z;
                    match build_contour_gpu_plot(
                        &x_vec,
                        &y_vec,
                        z_gpu,
                        ColorMap::Turbo,
                        base_z,
                        &level_spec,
                    ) {
                        Ok(contour) => {
                            figure.add_surface_plot_on_axes(surface, axes);
                            figure.add_contour_plot_on_axes(contour, axes);
                            return Ok(());
                        }
                        Err(err) => warn!("meshc contour GPU path unavailable: {err}"),
                    }
                }
                Err(err) => warn!("meshc surface GPU path unavailable: {err}"),
            }
        }

        let grid = tensor_to_surface_grid(z_arg.into_tensor("meshc")?, x_vec.len(), y_vec.len())?;
        let mut surface = build_mesh_surface(x_vec.clone(), y_vec.clone(), grid.clone())?;
        surface.shading_mode = ShadingMode::Faceted;
        let base_z = surface.bounds().min.z;
        let contour =
            build_contour_plot(&x_vec, &y_vec, &grid, ColorMap::Turbo, base_z, &level_spec)?;

        figure.add_surface_plot_on_axes(surface, axes);
        figure.add_contour_plot_on_axes(contour, axes);
        Ok(())
    })
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
    fn meshc_requires_matching_grid() {
        let res = meshc_builtin(
            tensor_from(&[0.0]),
            tensor_from(&[0.0, 1.0]),
            Value::Tensor(Tensor {
                data: vec![0.0],
                shape: vec![1],
                rows: 1,
                cols: 1,
                dtype: runmat_builtins::NumericDType::F64,
            }),
        );
        assert!(res.is_err());
    }
}
