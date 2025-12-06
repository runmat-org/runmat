//! Modern plotting builtin functions using the runmat-plot system. When the `gui`
//! feature is disabled (as it is for wasm builds today) the plotting builtins are
//! still registered but return a descriptive error so scripts fail gracefully.

#[cfg(feature = "gui")]
mod native {
    use runmat_builtins::{Tensor, Value};
    use runmat_macros::runtime_builtin;
    use runmat_plot::plots::figure::PlotElement;
    use runmat_plot::plots::*;
    use std::convert::TryInto;
    use std::env;

    #[derive(Debug, Clone, Copy)]
    enum PlottingMode {
        Auto,
        Interactive,
        Static,
        Jupyter,
    }

    fn get_plotting_mode() -> PlottingMode {
        if let Ok(mode) = env::var("RUSTMAT_PLOT_MODE") {
            match mode.to_lowercase().as_str() {
                "gui" => PlottingMode::Interactive,
                "headless" => PlottingMode::Static,
                "jupyter" => PlottingMode::Jupyter,
                _ => PlottingMode::Auto,
            }
        } else {
            PlottingMode::Auto
        }
    }

    fn execute_plot(mut figure: Figure) -> Result<String, String> {
        match get_plotting_mode() {
            PlottingMode::Interactive => interactive_export(&mut figure),
            PlottingMode::Static => static_export(&mut figure, "plot.png"),
            PlottingMode::Jupyter => jupyter_export(&mut figure),
            PlottingMode::Auto => {
                if env::var("JPY_PARENT_PID").is_ok() || env::var("JUPYTER_RUNTIME_DIR").is_ok() {
                    jupyter_export(&mut figure)
                } else {
                    interactive_export(&mut figure)
                }
            }
        }
    }

    fn interactive_export(figure: &mut Figure) -> Result<String, String> {
        let figure_clone = figure.clone();
        match runmat_plot::show_interactive_platform_optimal(figure_clone) {
            Ok(result) => Ok(result),
            Err(e) => Err(format!(
                "Interactive plotting failed: {e}. Please check GPU/GUI system setup."
            )),
        }
    }

    fn static_export(figure: &mut Figure, filename: &str) -> Result<String, String> {
        if figure.is_empty() {
            return Err("No plots found in figure to export".to_string());
        }

        match figure.get_plot_mut(0) {
            Some(PlotElement::Line(line_plot)) => {
                let x_data: Vec<f64> = line_plot.x_data.to_vec();
                let y_data: Vec<f64> = line_plot.y_data.to_vec();

                runmat_plot::plot_line(
                    &x_data,
                    &y_data,
                    filename,
                    runmat_plot::PlotOptions::default(),
                )
                .map_err(|e| format!("Plot export failed: {e}"))?;

                Ok(format!("Plot saved to {filename}"))
            }
            Some(PlotElement::Scatter(scatter_plot)) => {
                let x_data: Vec<f64> = scatter_plot.x_data.to_vec();
                let y_data: Vec<f64> = scatter_plot.y_data.to_vec();

                runmat_plot::plot_scatter(
                    &x_data,
                    &y_data,
                    filename,
                    runmat_plot::PlotOptions::default(),
                )
                .map_err(|e| format!("Plot export failed: {e}"))?;

                Ok(format!("Scatter plot saved to {filename}"))
            }
            Some(PlotElement::Bar(bar_chart)) => {
                let values: Vec<f64> = bar_chart.values.to_vec();

                runmat_plot::plot_bar(
                    &bar_chart.labels,
                    &values,
                    filename,
                    runmat_plot::PlotOptions::default(),
                )
                .map_err(|e| format!("Plot export failed: {e}"))?;

                Ok(format!("Bar chart saved to {filename}"))
            }
            Some(PlotElement::Histogram(histogram)) => {
                let data: Vec<f64> = histogram.data.to_vec();

                runmat_plot::plot_histogram(
                    &data,
                    histogram.bins,
                    filename,
                    runmat_plot::PlotOptions::default(),
                )
                .map_err(|e| format!("Plot export failed: {e}"))?;

                Ok(format!("Histogram saved to {filename}"))
            }
            Some(PlotElement::PointCloud(point_cloud)) => {
                let x_data: Vec<f64> = point_cloud
                    .positions
                    .iter()
                    .map(|pos| pos.x as f64)
                    .collect();
                let y_data: Vec<f64> = point_cloud
                    .positions
                    .iter()
                    .map(|pos| pos.y as f64)
                    .collect();

                runmat_plot::plot_scatter(
                    &x_data,
                    &y_data,
                    filename,
                    runmat_plot::PlotOptions::default(),
                )
                .map_err(|e| format!("Point cloud export failed: {e}"))?;

                Ok(format!("Point cloud (2D projection) saved to {filename}"))
            }
            None => Err("No plots found in figure to export".to_string()),
        }
    }

    #[cfg(feature = "jupyter")]
    fn jupyter_export(figure: &mut Figure) -> Result<String, String> {
        use runmat_plot::jupyter::JupyterBackend;
        let mut backend = JupyterBackend::new();
        backend.display_figure(figure)
    }

    #[cfg(not(feature = "jupyter"))]
    fn jupyter_export(_figure: &mut Figure) -> Result<String, String> {
        Err("Jupyter feature not enabled".to_string())
    }

    fn extract_numeric_vector(matrix: &Tensor) -> Vec<f64> {
        matrix.data.clone()
    }

    #[runtime_builtin(name = "plot", sink = true)]
    fn plot_builtin(x: Tensor, y: Tensor) -> Result<String, String> {
        let x_data = extract_numeric_vector(&x);
        let y_data = extract_numeric_vector(&y);

        if x_data.len() != y_data.len() {
            return Err("X and Y data must have the same length".to_string());
        }

        let line_plot = LinePlot::new(x_data.clone(), y_data.clone())
            .map_err(|e| format!("Failed to create line plot: {e}"))?
            .with_label("Data")
            .with_style(glam::Vec4::new(0.0, 0.4, 0.8, 1.0), 3.0, LineStyle::Solid);

        let mut figure = Figure::new()
            .with_title("Plot")
            .with_labels("X", "Y")
            .with_grid(true);

        figure.add_line_plot(line_plot);

        execute_plot(figure)
    }

    #[runtime_builtin(name = "scatter", sink = true)]
    fn scatter_builtin(x: Tensor, y: Tensor) -> Result<String, String> {
        let x_data = extract_numeric_vector(&x);
        let y_data = extract_numeric_vector(&y);

        if x_data.len() != y_data.len() {
            return Err("X and Y data must have the same length".to_string());
        }

        let scatter_plot = ScatterPlot::new(x_data.clone(), y_data.clone())
            .map_err(|e| format!("Failed to create scatter plot: {e}"))?
            .with_label("Data")
            .with_style(
                glam::Vec4::new(0.8, 0.2, 0.2, 1.0),
                glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
                8.0,
                MarkerShape::Circle,
            );

        let mut figure = Figure::new()
            .with_title("Scatter Plot")
            .with_labels("X", "Y")
            .with_grid(true);

        figure.add_scatter_plot(scatter_plot);

        execute_plot(figure)
    }

    #[runtime_builtin(name = "bar", sink = true)]
    fn bar_builtin(values: Tensor) -> Result<String, String> {
        let values = extract_numeric_vector(&values);
        let categories: Vec<String> = (0..values.len()).map(|i| format!("C{}", i + 1)).collect();

        let bar_chart = BarChart::new(
            categories.clone(),
            values.clone(),
            glam::Vec4::new(0.2, 0.6, 0.8, 1.0),
        )
        .map_err(|e| format!("Failed to create bar chart: {e}"))?;

        let mut figure = Figure::new()
            .with_title("Bar Chart")
            .with_labels("Categories", "Values")
            .with_grid(true);

        figure.add_bar_chart(bar_chart);

        execute_plot(figure)
    }

    #[runtime_builtin(name = "hist", sink = true)]
    fn hist_builtin(data: Tensor, rest: Vec<Value>) -> Result<String, String> {
        let data = extract_numeric_vector(&data);
        let bins = rest
            .first()
            .and_then(|v| v.clone().try_into().ok())
            .unwrap_or_else(|| (data.len() as f64).sqrt() as usize);

        let histogram = HistogramPlot::new(data.clone(), bins.max(1))
            .map_err(|e| format!("Failed to create histogram: {e}"))?;

        let mut figure = Figure::new()
            .with_title("Histogram")
            .with_labels("Bins", "Frequency")
            .with_grid(true);

        figure.add_histogram(histogram);

        execute_plot(figure)
    }

    #[runtime_builtin(name = "surf", sink = true)]
    fn surf_builtin(x: Tensor, y: Tensor, z: Tensor) -> Result<String, String> {
        let x_data = extract_numeric_vector(&x);
        let y_data = extract_numeric_vector(&y);
        let z_data = extract_numeric_vector(&z);

        if x_data.len() != y_data.len() || x_data.len() != z_data.len() {
            return Err("X, Y, and Z must have the same number of elements".to_string());
        }

        let mut surface = SurfacePlot::new()
            .with_color_palette(ColorPalette::Turbo)
            .with_opacity(0.9);

        for (&x_val, (&y_val, &z_val)) in x_data.iter().zip(y_data.iter().zip(z_data.iter())) {
            surface.add_vertex(glam::Vec3::new(x_val as f32, y_val as f32, z_val as f32));
        }

        let mut figure = Figure::new()
            .with_title("Surface Plot")
            .with_labels("X", "Y")
            .with_grid(true);

        figure.add_surface(surface);

        execute_plot(figure)
    }

    #[runtime_builtin(name = "scatter3", sink = true)]
    fn scatter3_builtin(x: Tensor, y: Tensor, z: Tensor) -> Result<String, String> {
        let x_data = extract_numeric_vector(&x);
        let y_data = extract_numeric_vector(&y);
        let z_data = extract_numeric_vector(&z);

        if x_data.len() != y_data.len() || x_data.len() != z_data.len() {
            return Err("X, Y, and Z must have the same number of elements".to_string());
        }

        let points: Vec<glam::Vec3> = x_data
            .iter()
            .zip(y_data.iter())
            .zip(z_data.iter())
            .map(|((x, y), z)| glam::Vec3::new(*x as f32, *y as f32, *z as f32))
            .collect();

        let mut point_cloud = PointCloud::new(points)
            .with_point_size(6.0)
            .with_color(glam::Vec4::new(0.1, 0.7, 0.3, 1.0));

        point_cloud.enable_depth_test(true);

        let mut figure = Figure::new()
            .with_title("3D Scatter Plot")
            .with_labels("X", "Y")
            .with_grid(true);

        figure.add_point_cloud(point_cloud);

        execute_plot(figure)
    }

    #[runtime_builtin(name = "mesh", sink = true)]
    fn mesh_builtin(x: Tensor, y: Tensor, z: Tensor) -> Result<String, String> {
        let x_data = extract_numeric_vector(&x);
        let y_data = extract_numeric_vector(&y);
        let z_data = extract_numeric_vector(&z);

        if x_data.len() != y_data.len() || x_data.len() != z_data.len() {
            return Err("X, Y, and Z must have the same number of elements".to_string());
        }

        let mesh = MeshPlot::new(x_data, y_data, z_data)
            .map_err(|e| format!("Failed to create mesh plot: {e}"))?;

        let mut figure = Figure::new()
            .with_title("Mesh Plot")
            .with_labels("X", "Y")
            .with_grid(true);

        figure.add_mesh(mesh);

        execute_plot(figure)
    }
}

#[cfg(feature = "gui")]
pub use native::*;

#[cfg(all(target_arch = "wasm32", not(feature = "gui")))]
mod wasm_web {
    use runmat_builtins::{Tensor, Value};
    use runmat_macros::runtime_builtin;
    use serde_json::json;
    use wasm_bindgen::JsValue;

    const EVENT_NAME: &str = "runmat:plot";

    #[runtime_builtin(name = "plot", sink = true)]
    fn plot_builtin(x: Tensor, y: Tensor) -> Result<String, String> {
        let (x_data, y_data) = extract_pair(x, y)?;
        emit_plot_event(
            "line2d",
            json!({ "x": x_data, "y": y_data, "label": "Data" }),
        )
    }

    #[runtime_builtin(name = "scatter", sink = true)]
    fn scatter_builtin(x: Tensor, y: Tensor) -> Result<String, String> {
        let (x_data, y_data) = extract_pair(x, y)?;
        emit_plot_event(
            "scatter2d",
            json!({ "x": x_data, "y": y_data, "marker": { "shape": "circle", "size": 6 } }),
        )
    }

    #[runtime_builtin(name = "bar", sink = true)]
    fn bar_builtin(values: Tensor) -> Result<String, String> {
        let values = extract_vector(&values)?;
        let labels: Vec<String> = (0..values.len()).map(|i| format!("C{}", i + 1)).collect();
        emit_plot_event("bar", json!({ "labels": labels, "values": values }))
    }

    #[runtime_builtin(name = "hist", sink = true)]
    fn hist_builtin(data: Tensor, rest: Vec<Value>) -> Result<String, String> {
        let data = extract_vector(&data)?;
        let bins = rest
            .first()
            .and_then(|value| {
                let numeric: f64 = value.clone().try_into().ok()?;
                Some(numeric.max(1.0) as usize)
            })
            .unwrap_or_else(|| (data.len() as f64).sqrt() as usize);
        emit_plot_event("histogram", json!({ "data": data, "bins": bins }))
    }

    #[runtime_builtin(name = "surf", sink = true)]
    fn surf_builtin(x: Tensor, y: Tensor, z: Tensor) -> Result<String, String> {
        let (x_data, y_data, z_data) = extract_triplet(x, y, z)?;
        emit_plot_event(
            "surface3d",
            json!({ "x": x_data, "y": y_data, "z": z_data, "palette": "turbo" }),
        )
    }

    #[runtime_builtin(name = "scatter3", sink = true)]
    fn scatter3_builtin(x: Tensor, y: Tensor, z: Tensor) -> Result<String, String> {
        let (x_data, y_data, z_data) = extract_triplet(x, y, z)?;
        emit_plot_event(
            "scatter3d",
            json!({ "x": x_data, "y": y_data, "z": z_data, "size": 5.0 }),
        )
    }

    #[runtime_builtin(name = "mesh", sink = true)]
    fn mesh_builtin(x: Tensor, y: Tensor, z: Tensor) -> Result<String, String> {
        let (x_data, y_data, z_data) = extract_triplet(x, y, z)?;
        emit_plot_event("mesh3d", json!({ "x": x_data, "y": y_data, "z": z_data }))
    }

    fn extract_vector(tensor: &Tensor) -> Result<Vec<f64>, String> {
        Ok(tensor.data.clone())
    }

    fn extract_pair(x: Tensor, y: Tensor) -> Result<(Vec<f64>, Vec<f64>), String> {
        let x_data = extract_vector(&x)?;
        let y_data = extract_vector(&y)?;
        if x_data.len() != y_data.len() {
            return Err("X and Y data must have the same length".to_string());
        }
        Ok((x_data, y_data))
    }

    fn extract_triplet(
        x: Tensor,
        y: Tensor,
        z: Tensor,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
        let x_data = extract_vector(&x)?;
        let y_data = extract_vector(&y)?;
        let z_data = extract_vector(&z)?;
        if x_data.len() != y_data.len() || x_data.len() != z_data.len() {
            return Err("X, Y, and Z must have the same number of elements".to_string());
        }
        Ok((x_data, y_data, z_data))
    }

    fn emit_plot_event(kind: &str, data: serde_json::Value) -> Result<String, String> {
        let payload = json!({
            "version": 1,
            "kind": kind,
            "data": data
        });
        let json = payload.to_string();
        let detail = JsValue::from_str(&json);
        if let Some(window) = web_sys::window() {
            let mut init = web_sys::CustomEventInit::new();
            init.set_detail(&detail);
            let event = web_sys::CustomEvent::new_with_event_init_dict(EVENT_NAME, &init)
                .map_err(|err| format!("Failed to construct plot event: {err:?}"))?;
            window
                .dispatch_event(&event)
                .map_err(|err| format!("Plot event dispatch failed: {err:?}"))?;
        } else {
            log::warn!("runmat plot event dropped (window unavailable): {json}");
        }
        Ok(format!("Plot '{kind}' dispatched"))
    }
}

#[cfg(all(target_arch = "wasm32", not(feature = "gui")))]
pub use wasm_web::*;

#[cfg(all(not(target_arch = "wasm32"), not(feature = "gui")))]
mod stub {
    use runmat_builtins::{Tensor, Value};
    use runmat_macros::runtime_builtin;

    fn plotting_unavailable() -> Result<String, String> {
        Err("Plotting is unavailable in this build".to_string())
    }

    #[runtime_builtin(name = "plot", sink = true)]
    fn plot_builtin(_x: Tensor, _y: Tensor) -> Result<String, String> {
        plotting_unavailable()
    }

    #[runtime_builtin(name = "scatter", sink = true)]
    fn scatter_builtin(_x: Tensor, _y: Tensor) -> Result<String, String> {
        plotting_unavailable()
    }

    #[runtime_builtin(name = "bar", sink = true)]
    fn bar_builtin(_values: Tensor) -> Result<String, String> {
        plotting_unavailable()
    }

    #[runtime_builtin(name = "hist", sink = true)]
    fn hist_builtin(_data: Tensor, _rest: Vec<Value>) -> Result<String, String> {
        plotting_unavailable()
    }

    #[runtime_builtin(name = "surf", sink = true)]
    fn surf_builtin(_x: Tensor, _y: Tensor, _z: Tensor) -> Result<String, String> {
        plotting_unavailable()
    }

    #[runtime_builtin(name = "scatter3", sink = true)]
    fn scatter3_builtin(_x: Tensor, _y: Tensor, _z: Tensor) -> Result<String, String> {
        plotting_unavailable()
    }

    #[runtime_builtin(name = "mesh", sink = true)]
    fn mesh_builtin(_x: Tensor, _y: Tensor, _z: Tensor) -> Result<String, String> {
        plotting_unavailable()
    }
}

#[cfg(all(not(target_arch = "wasm32"), not(feature = "gui")))]
#[allow(unused_imports)]
pub use stub::*;
