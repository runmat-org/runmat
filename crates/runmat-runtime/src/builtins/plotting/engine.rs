use runmat_plot::plots::Figure;

use super::common::ERR_PLOTTING_UNAVAILABLE;

pub fn render_figure(figure: Figure) -> Result<String, String> {
    #[cfg(feature = "gui")]
    {
        return native::render(figure);
    }

    #[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
    {
        return super::web::render_web_canvas(figure);
    }

    let _ = figure;
    Err(ERR_PLOTTING_UNAVAILABLE.to_string())
}

#[cfg(feature = "gui")]
mod native {
    use super::*;
    use runmat_plot::plots::Figure;
    use std::env;

    #[derive(Debug, Clone, Copy)]
    enum PlottingMode {
        Auto,
        Interactive,
        Static,
        Jupyter,
    }

    pub fn render(mut figure: Figure) -> Result<String, String> {
        match detect_mode() {
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

    fn detect_mode() -> PlottingMode {
        if let Ok(mode) = env::var("RUSTMAT_PLOT_MODE") {
            match mode.to_lowercase().as_str() {
                "gui" => PlottingMode::Interactive,
                "headless" | "static" => PlottingMode::Static,
                "jupyter" => PlottingMode::Jupyter,
                _ => PlottingMode::Auto,
            }
        } else {
            PlottingMode::Auto
        }
    }

    fn interactive_export(figure: &mut Figure) -> Result<String, String> {
        let figure_clone = figure.clone();
        runmat_plot::show_interactive_platform_optimal(figure_clone).map_err(|e| {
            format!("Interactive plotting failed: {e}. Please check GPU/GUI system setup.")
        })
    }

    fn static_export(figure: &mut Figure, filename: &str) -> Result<String, String> {
        if figure.is_empty() {
            return Err("No plots found in figure to export".to_string());
        }
        runmat_plot::show_plot_unified(figure.clone(), Some(filename))
            .map(|_| format!("Plot saved to {filename}"))
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
}
