//! MATLAB-compatible `meshc` builtin (mesh with contour).

use log::warn;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};

use super::common::{numeric_vector, tensor_to_surface_grid, SurfaceDataInput};
use super::contour::{
    build_contour_gpu_plot, build_contour_plot, default_level_count, ContourLevelSpec,
    ContourLineColor,
};
use super::mesh::build_mesh_surface;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use super::surf::build_surface_gpu_plot;
use std::sync::Arc;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "meshc",
        builtin_path = "crate::builtins::plotting::meshc"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
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

#[runtime_builtin(
    name = "meshc",
    category = "plotting",
    summary = "Render a MATLAB-compatible mesh with contour overlay.",
    keywords = "meshc,plotting,mesh,contour",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::meshc"
)]
pub fn meshc_builtin(x: Tensor, y: Tensor, z: Value, rest: Vec<Value>) -> Result<String, String> {
    let x_axis = numeric_vector(x);
    let y_axis = numeric_vector(y);
    let mut x_axis = Some(x_axis);
    let mut y_axis = Some(y_axis);
    let mut z_input = Some(SurfaceDataInput::from_value(z, "meshc")?);
    let style = Arc::new(parse_surface_style_args(
        "meshc",
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

        let style = Arc::clone(&style);
        let contour_map = style.colormap;
        if let Some(z_gpu) = z_arg.gpu_handle() {
            match build_surface_gpu_plot(&x_vec, &y_vec, z_gpu) {
                Ok(surface_gpu) => {
                    let mut surface = surface_gpu.with_wireframe(true);
                    style.apply_to_plot(&mut surface);
                    let base_z = surface.bounds().min.z;
                    match build_contour_gpu_plot(
                        &x_vec,
                        &y_vec,
                        z_gpu,
                        contour_map,
                        base_z,
                        &level_spec,
                        &ContourLineColor::Auto,
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
        style.apply_to_plot(&mut surface);
        let base_z = surface.bounds().min.z;
        let contour = build_contour_plot(
            &x_vec,
            &y_vec,
            &grid,
            contour_map,
            base_z,
            &level_spec,
            &ContourLineColor::Auto,
        )?;

        figure.add_surface_plot_on_axes(surface, axes);
        figure.add_contour_plot_on_axes(contour, axes);
        Ok(())
    })
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
    fn meshc_requires_matching_grid() {
        setup_plot_tests();
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
            Vec::new(),
        );
        assert!(res.is_err());
    }
}
