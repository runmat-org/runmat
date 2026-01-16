//! MATLAB-compatible `contourf` builtin (filled contour plot).

use log::warn;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_plot::plots::ColorMap;

use super::common::tensor_to_surface_grid;
use super::contour::{
    build_contour_fill_gpu_plot, build_contour_fill_plot, build_contour_gpu_plot,
    build_contour_plot, parse_contour_args, ContourArgs, ContourLineColor,
};
use super::state::{render_active_plot, PlotRenderOptions};

#[runtime_builtin(
    name = "contourf",
    category = "plotting",
    summary = "Render MATLAB-compatible filled contour plots.",
    keywords = "contourf,plotting,filled,contour",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::contourf"
)]
pub fn contourf_builtin(first: Value, rest: Vec<Value>) -> Result<String, String> {
    let mut args = Some(parse_contour_args("contourf", first, rest)?);
    let opts = PlotRenderOptions {
        title: "Filled Contour Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    render_active_plot(opts, move |figure, axes| {
        let ContourArgs {
            name,
            x_axis,
            y_axis,
            z_input,
            level_spec,
            line_color,
        } = args.take().expect("contourf args consumed once");
        let color_map = ColorMap::Parula;
        let base_z = 0.0;

        if let Some(handle) = z_input.gpu_handle() {
            match build_contour_fill_gpu_plot(
                &x_axis,
                &y_axis,
                handle,
                color_map,
                base_z,
                &level_spec,
            ) {
                Ok(fill_plot) => {
                    figure.add_contour_fill_plot_on_axes(fill_plot, axes);
                    if !matches!(line_color, ContourLineColor::None) {
                        if let Ok(contours) = build_contour_gpu_plot(
                            &x_axis,
                            &y_axis,
                            handle,
                            color_map,
                            base_z,
                            &level_spec,
                            &line_color,
                        ) {
                            figure.add_contour_plot_on_axes(contours, axes);
                        } else {
                            warn!(
                                "contourf contour overlay unavailable: failed to build contour lines"
                            );
                        }
                    }
                    return Ok(());
                }
                Err(err) => warn!("contourf GPU path unavailable: {err}"),
            }
        }

        let grid = tensor_to_surface_grid(z_input.into_tensor(name)?, x_axis.len(), y_axis.len())?;
        let fill_plot =
            build_contour_fill_plot(&x_axis, &y_axis, &grid, color_map, base_z, &level_spec)?;
        figure.add_contour_fill_plot_on_axes(fill_plot, axes);
        if !matches!(line_color, ContourLineColor::None) {
            if let Ok(contours) = build_contour_plot(
                &x_axis,
                &y_axis,
                &grid,
                color_map,
                base_z,
                &level_spec,
                &line_color,
            ) {
                figure.add_contour_plot_on_axes(contours, axes);
            } else {
                warn!("contourf overlay contour unavailable: failed to build contour lines");
            }
        }
        Ok(())
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use runmat_builtins::Tensor;

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
    fn contourf_requires_matching_grid() {
        setup_plot_tests();
        let res = contourf_builtin(
            Value::Tensor(tensor_from(&[0.0])),
            vec![
                Value::Tensor(tensor_from(&[0.0, 1.0])),
                Value::Tensor(tensor_from(&[0.0, 1.0])),
            ],
        );
        assert!(res.is_err());
    }
}
