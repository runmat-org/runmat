use runmat_plot::plots::Figure;

#[cfg(not(any(feature = "gui", all(target_arch = "wasm32", feature = "plot-web"))))]
use super::common::ERR_PLOTTING_UNAVAILABLE;
use super::state::{clone_figure, FigureHandle};
use thiserror::Error;

#[cfg(feature = "plot-core")]
use crate::builtins::common::map_control_flow_with_builtin;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[derive(Debug, Error)]
#[allow(dead_code)]
enum PlottingBackendError {
    #[error("interactive backend error: {0}")]
    Interactive(String),
    #[error("static backend error: {0}")]
    Static(String),
    #[error("jupyter backend error: {0}")]
    Jupyter(String),
    #[error("export initialization error: {0}")]
    ImageExportInit(String),
    #[error("export render error: {0}")]
    ImageExport(String),
}

fn engine_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).build()
}

fn engine_error_with_source(
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    build_runtime_error(message).with_source(source).build()
}

pub fn render_figure(handle: FigureHandle, figure: Figure) -> BuiltinResult<String> {
    #[cfg(feature = "gui")]
    {
        native::render(handle, figure)
    }

    #[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
    {
        super::web::render_web_canvas(handle.as_u32(), figure)
    }

    #[cfg(not(any(feature = "gui", all(target_arch = "wasm32", feature = "plot-web"))))]
    {
        let _ = handle;
        let _ = figure;
        Err(engine_error(ERR_PLOTTING_UNAVAILABLE))
    }
}

#[cfg(feature = "plot-core")]
pub async fn render_figure_png_bytes(
    mut figure: Figure,
    width: u32,
    height: u32,
) -> BuiltinResult<Vec<u8>> {
    use runmat_plot::export::image::{ImageExportSettings, ImageExporter};

    let mut settings = ImageExportSettings::default();
    if width > 0 {
        settings.width = width;
    }
    if height > 0 {
        settings.height = height;
    }

    let exporter = ImageExporter::with_settings(settings)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export initialization failed.",
                PlottingBackendError::ImageExportInit(err),
            )
        })?;
    exporter.render_png_bytes(&mut figure).await.map_err(|err| {
        engine_error_with_source(
            "Plot export failed.",
            PlottingBackendError::ImageExport(err),
        )
    })
}

#[cfg(feature = "plot-core")]
pub async fn render_figure_snapshot(
    handle: FigureHandle,
    width: u32,
    height: u32,
) -> BuiltinResult<Vec<u8>> {
    const SNAPSHOT_CONTEXT: &str = "renderFigureImage";
    let figure = clone_figure(handle).ok_or_else(|| {
        map_control_flow_with_builtin(
            engine_error(format!("figure handle {} does not exist", handle.as_u32())),
            SNAPSHOT_CONTEXT,
        )
    })?;
    render_figure_png_bytes(figure, width, height)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, SNAPSHOT_CONTEXT))
}

#[cfg(feature = "gui")]
pub(crate) mod native {
    use super::super::state::{install_figure_observer, FigureEventKind, FigureEventView};
    use super::*;
    use once_cell::sync::OnceCell;
    use runmat_plot::plots::Figure;
    use std::env;
    use std::sync::Arc;

    static FIGURE_EVENT_BRIDGE: OnceCell<()> = OnceCell::new();

    #[derive(Debug, Clone, Copy)]
    enum PlottingMode {
        Auto,
        Interactive,
        Static,
        Jupyter,
    }

    pub fn render(handle: FigureHandle, mut figure: Figure) -> BuiltinResult<String> {
        ensure_figure_event_bridge();
        match detect_mode() {
            PlottingMode::Interactive => interactive_export(handle, &mut figure),
            PlottingMode::Static => static_export(&mut figure, "plot.png"),
            PlottingMode::Jupyter => jupyter_export(&mut figure),
            PlottingMode::Auto => {
                if env::var("JPY_PARENT_PID").is_ok() || env::var("JUPYTER_RUNTIME_DIR").is_ok() {
                    jupyter_export(&mut figure)
                } else {
                    interactive_export(handle, &mut figure)
                }
            }
        }
    }

    fn detect_mode() -> PlottingMode {
        if let Ok(mode) = env::var("RUNMAT_PLOT_MODE") {
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

    fn interactive_export(handle: FigureHandle, figure: &mut Figure) -> BuiltinResult<String> {
        let figure_clone = figure.clone();
        runmat_plot::render_interactive_with_handle(handle.as_u32(), figure_clone).map_err(|err| {
            engine_error_with_source(
                "Interactive plotting failed. Please check GPU/GUI system setup.",
                PlottingBackendError::Interactive(err),
            )
        })
    }

    fn static_export(figure: &mut Figure, filename: &str) -> BuiltinResult<String> {
        if figure.is_empty() {
            return Err(engine_error("No plots found in figure to export"));
        }
        runmat_plot::show_plot_unified(figure.clone(), Some(filename))
            .map(|_| format!("Plot saved to {filename}"))
            .map_err(|err| {
                engine_error_with_source("Plot export failed.", PlottingBackendError::Static(err))
            })
    }

    #[cfg(feature = "jupyter")]
    fn jupyter_export(figure: &mut Figure) -> BuiltinResult<String> {
        use runmat_plot::jupyter::JupyterBackend;
        let mut backend = JupyterBackend::new();
        backend.display_figure(figure).map_err(|err| {
            engine_error_with_source(
                "Jupyter plotting failed.",
                PlottingBackendError::Jupyter(err),
            )
        })
    }

    #[cfg(not(feature = "jupyter"))]
    fn jupyter_export(_figure: &mut Figure) -> BuiltinResult<String> {
        Err(engine_error("Jupyter feature not enabled"))
    }

    fn ensure_figure_event_bridge() {
        FIGURE_EVENT_BRIDGE.get_or_init(|| {
            let observer: Arc<dyn for<'a> Fn(FigureEventView<'a>) + Send + Sync> =
                Arc::new(|event: FigureEventView<'_>| {
                    if let FigureEventKind::Closed = event.kind {
                        runmat_plot::gui::lifecycle::request_close(event.handle.as_u32());
                    }
                });
            let _ = install_figure_observer(observer);
        });
    }
}
