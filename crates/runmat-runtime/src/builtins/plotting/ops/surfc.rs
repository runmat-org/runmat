//! MATLAB-compatible `surfc` builtin (surface with contour).

use log::warn;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};

use super::common::{numeric_vector, tensor_to_surface_grid, SurfaceDataInput};
use super::contour::{
    build_contour_gpu_plot, build_contour_plot, default_level_count, ContourLevelSpec,
    ContourLineColor,
};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, SurfaceStyleDefaults};
use super::surf::{build_surface, build_surface_gpu_plot};
use std::sync::Arc;

use crate::RuntimeControlFlow;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "surfc",
        builtin_path = "crate::builtins::plotting::surfc"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
const BUILTIN_NAME: &str = "surfc";

#[allow(dead_code)]
pub const DOC_MD: &str = r#"---
title: "surfc"
category: "plotting"
keywords: ["surfc", "surface", "contour", "gpuArray"]
summary: "Render a shaded surface with contour lines on the base plane."
references:
  - https://www.mathworks.com/help/matlab/ref/surfc.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["single"]
  broadcasting: "none"
  notes: "Single-precision gpuArray inputs stay on the device via the shared WebGPU renderer."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::plotting::surfc::tests"
---

# What does `surfc` do?
`surfc(X, Y, Z)` draws a shaded surface and overlays contour lines projected onto the XY plane.
RunMat reuses the same surface renderer as `surf` and complements it with GPU-generated iso-lines.

## GPU behaviour
- Single-precision gpuArrays stream directly into the shared WebGPU renderer.
- Double-precision data falls back to the CPU path until SHADER_F64 is available.
"#;

#[runtime_builtin(
    name = "surfc",
    category = "plotting",
    summary = "Render a MATLAB-compatible surface with contour overlay.",
    keywords = "surfc,plotting,surface,contour",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::surfc"
)]
pub fn surfc_builtin(x: Tensor, y: Tensor, z: Value, rest: Vec<Value>) -> crate::BuiltinResult<String> {
    let x_axis = numeric_vector(x);
    let y_axis = numeric_vector(y);
    let mut x_axis = Some(x_axis);
    let mut y_axis = Some(y_axis);
    let mut z_input = Some(SurfaceDataInput::from_value(z, "surfc")?);
    let style = Arc::new(parse_surface_style_args(
        "surfc",
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
        title: "Surface with Contours",
        x_label: "X",
        y_label: "Y",
        axis_equal: false,
        ..Default::default()
    };
    let level_spec = ContourLevelSpec::Count(default_level_count());
    let rendered = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let level_spec = level_spec.clone();
        let x_vec = x_axis.take().expect("surfc X consumed once");
        let y_vec = y_axis.take().expect("surfc Y consumed once");
        let z_arg = z_input.take().expect("surfc Z consumed once");

        let style = Arc::clone(&style);
        let contour_map = style.colormap;
        if let Some(z_gpu) = z_arg.gpu_handle() {
            match build_surface_gpu_plot(BUILTIN_NAME, &x_vec, &y_vec, z_gpu) {
                Ok(mut surface) => {
                    let base_z = surface.bounds().min.z;
                    style.apply_to_plot(&mut surface);
                    match build_contour_gpu_plot(
                        BUILTIN_NAME,
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
                        Err(RuntimeControlFlow::Suspend(pending)) => {
                            return Err(RuntimeControlFlow::Suspend(pending));
                        }
                        Err(RuntimeControlFlow::Error(err)) => {
                            warn!("surfc contour GPU path unavailable: {err}");
                        }
                    }
                }
                Err(RuntimeControlFlow::Suspend(pending)) => {
                    return Err(RuntimeControlFlow::Suspend(pending));
                }
                Err(RuntimeControlFlow::Error(err)) => {
                    warn!("surfc surface GPU path unavailable: {err}");
                }
            }
        }

        let grid = tensor_to_surface_grid(
            z_arg.into_tensor(BUILTIN_NAME)?,
            x_vec.len(),
            y_vec.len(),
            BUILTIN_NAME,
        )?;
        let mut surface = build_surface(x_vec.clone(), y_vec.clone(), grid.clone())?;
        style.apply_to_plot(&mut surface);
        let base_z = surface.bounds().min.z;
        let contour = build_contour_plot(
            BUILTIN_NAME,
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
    })?;
    Ok(rendered)
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
    fn surfc_requires_matching_grid() {
        setup_plot_tests();
        let res = surfc_builtin(
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
