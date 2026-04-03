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
    build_runtime_error(message)
        .with_identifier("RunMat:plot:EngineError")
        .build()
}

fn engine_error_with_source(
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:plot:EngineError")
        .with_source(source)
        .build()
}

#[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
pub fn render_figure(handle: FigureHandle, figure: Figure) -> BuiltinResult<String> {
    #[cfg(feature = "gui")]
    {
        native::render(handle, figure)
    }

    #[cfg(not(feature = "gui"))]
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

    let mut exporter = ImageExporter::with_settings(settings)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export initialization failed.",
                PlottingBackendError::ImageExportInit(err),
            )
        })?;
    exporter.set_theme_config(super::web::current_plot_theme_config());
    exporter.render_png_bytes(&mut figure).await.map_err(|err| {
        engine_error_with_source(
            "Plot export failed.",
            PlottingBackendError::ImageExport(err),
        )
    })
}

#[cfg(feature = "plot-core")]
pub async fn render_figure_rgba_bytes(
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

    let mut exporter = ImageExporter::with_settings(settings)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export initialization failed.",
                PlottingBackendError::ImageExportInit(err),
            )
        })?;
    exporter.set_theme_config(super::web::current_plot_theme_config());
    exporter
        .render_rgba_bytes(&mut figure)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export failed.",
                PlottingBackendError::ImageExport(err),
            )
        })
}

#[cfg(feature = "plot-core")]
pub async fn render_figure_png_bytes_with_camera(
    mut figure: Figure,
    width: u32,
    height: u32,
    camera: &runmat_plot::core::Camera,
) -> BuiltinResult<Vec<u8>> {
    use runmat_plot::export::image::{ImageExportSettings, ImageExporter};

    let mut settings = ImageExportSettings::default();
    if width > 0 {
        settings.width = width;
    }
    if height > 0 {
        settings.height = height;
    }

    let mut exporter = ImageExporter::with_settings(settings)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export initialization failed.",
                PlottingBackendError::ImageExportInit(err),
            )
        })?;
    exporter.set_theme_config(super::web::current_plot_theme_config());
    exporter
        .render_png_bytes_with_camera(&mut figure, camera)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export failed.",
                PlottingBackendError::ImageExport(err),
            )
        })
}

#[cfg(feature = "plot-core")]
pub async fn render_figure_rgba_bytes_with_camera(
    mut figure: Figure,
    width: u32,
    height: u32,
    camera: &runmat_plot::core::Camera,
) -> BuiltinResult<Vec<u8>> {
    use runmat_plot::export::image::{ImageExportSettings, ImageExporter};

    let mut settings = ImageExportSettings::default();
    if width > 0 {
        settings.width = width;
    }
    if height > 0 {
        settings.height = height;
    }

    let mut exporter = ImageExporter::with_settings(settings)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export initialization failed.",
                PlottingBackendError::ImageExportInit(err),
            )
        })?;
    exporter.set_theme_config(super::web::current_plot_theme_config());
    exporter
        .render_rgba_bytes_with_camera(&mut figure, camera)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export failed.",
                PlottingBackendError::ImageExport(err),
            )
        })
}

#[cfg(feature = "plot-core")]
pub async fn render_figure_png_bytes_with_axes_cameras(
    mut figure: Figure,
    width: u32,
    height: u32,
    axes_cameras: &[runmat_plot::core::Camera],
) -> BuiltinResult<Vec<u8>> {
    use runmat_plot::export::image::{ImageExportSettings, ImageExporter};

    let mut settings = ImageExportSettings::default();
    if width > 0 {
        settings.width = width;
    }
    if height > 0 {
        settings.height = height;
    }

    let mut exporter = ImageExporter::with_settings(settings)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export initialization failed.",
                PlottingBackendError::ImageExportInit(err),
            )
        })?;
    exporter.set_theme_config(super::web::current_plot_theme_config());
    exporter
        .render_png_bytes_with_axes_cameras(&mut figure, axes_cameras)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export failed.",
                PlottingBackendError::ImageExport(err),
            )
        })
}

#[cfg(feature = "plot-core")]
pub async fn render_figure_rgba_bytes_with_axes_cameras(
    mut figure: Figure,
    width: u32,
    height: u32,
    axes_cameras: &[runmat_plot::core::Camera],
) -> BuiltinResult<Vec<u8>> {
    use runmat_plot::export::image::{ImageExportSettings, ImageExporter};

    let mut settings = ImageExportSettings::default();
    if width > 0 {
        settings.width = width;
    }
    if height > 0 {
        settings.height = height;
    }

    let mut exporter = ImageExporter::with_settings(settings)
        .await
        .map_err(|err| {
            engine_error_with_source(
                "Plot export initialization failed.",
                PlottingBackendError::ImageExportInit(err),
            )
        })?;
    exporter.set_theme_config(super::web::current_plot_theme_config());
    exporter
        .render_rgba_bytes_with_axes_cameras(&mut figure, axes_cameras)
        .await
        .map_err(|err| {
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
    textmark: Option<String>,
) -> BuiltinResult<Vec<u8>> {
    const SNAPSHOT_CONTEXT: &str = "renderFigureImage";
    log::debug!(
        "runmat-runtime: render_figure_snapshot.start handle={} width={} height={} textmark={}",
        handle.as_u32(),
        width,
        height,
        textmark.as_deref().unwrap_or("")
    );
    let figure = clone_figure(handle).ok_or_else(|| {
        map_control_flow_with_builtin(
            engine_error(format!("figure handle {} does not exist", handle.as_u32())),
            SNAPSHOT_CONTEXT,
        )
    })?;
    log::debug!(
        "runmat-runtime: render_figure_snapshot.figure_cloned handle={} axes={} elements={}",
        handle.as_u32(),
        figure.axes_metadata.len(),
        figure.statistics().total_plots
    );
    let bytes = runmat_plot::export::native_surface::render_figure_png_bytes_interactive_and_theme_and_textmark(
        figure,
        width,
        height,
        super::web::current_plot_theme_config(),
        textmark.as_deref(),
    )
    .await
    .map_err(|err| {
        log::warn!(
            "runmat-runtime: render_figure_snapshot.failed handle={} width={} height={} error={}",
            handle.as_u32(),
            width,
            height,
            err
        );
        map_control_flow_with_builtin(
            engine_error_with_source(
                format!("Plot export failed: {err}"),
                PlottingBackendError::ImageExport(err),
            ),
            SNAPSHOT_CONTEXT,
        )
    })?;
    log::debug!(
        "runmat-runtime: render_figure_snapshot.ok handle={} bytes={}",
        handle.as_u32(),
        bytes.len()
    );
    Ok(bytes)
}

#[cfg(feature = "plot-core")]
pub async fn render_figure_snapshot_with_camera_state(
    handle: FigureHandle,
    width: u32,
    height: u32,
    camera_state: super::web::PlotSurfaceCameraState,
    textmark: Option<String>,
) -> BuiltinResult<Vec<u8>> {
    const SNAPSHOT_CONTEXT: &str = "renderFigureImage";
    log::debug!(
        "runmat-runtime: render_figure_snapshot_with_camera_state.start handle={} width={} height={} axes={}",
        handle.as_u32(),
        width,
        height,
        camera_state.axes.len()
    );
    let figure = clone_figure(handle).ok_or_else(|| {
        map_control_flow_with_builtin(
            engine_error(format!("figure handle {} does not exist", handle.as_u32())),
            SNAPSHOT_CONTEXT,
        )
    })?;

    let axes_cameras: Vec<runmat_plot::core::Camera> = camera_state
        .axes
        .iter()
        .map(surface_plot_camera_to_core_camera)
        .collect();

    if axes_cameras.is_empty() {
        let bytes = runmat_plot::export::native_surface::render_figure_png_bytes_interactive_and_theme_and_textmark(
            figure,
            width,
            height,
            super::web::current_plot_theme_config(),
            textmark.as_deref(),
        )
        .await
            .map_err(|err| {
                log::warn!(
                    "runmat-runtime: render_figure_snapshot_with_camera_state.fallback_failed handle={} error={}",
                    handle.as_u32(),
                    err
                );
                map_control_flow_with_builtin(
                    engine_error_with_source(
                        format!("Plot export failed: {err}"),
                        PlottingBackendError::ImageExport(err),
                    ),
                    SNAPSHOT_CONTEXT,
                )
            })?;
        log::debug!(
            "runmat-runtime: render_figure_snapshot_with_camera_state.fallback_ok handle={} bytes={}",
            handle.as_u32(),
            bytes.len()
        );
        return Ok(bytes);
    }

    let bytes = runmat_plot::export::native_surface::render_figure_png_bytes_interactive_with_axes_cameras_and_theme_and_textmark(
        figure,
        width,
        height,
        &axes_cameras,
        super::web::current_plot_theme_config(),
        textmark.as_deref(),
    )
    .await
    .map_err(|err| {
        log::warn!(
            "runmat-runtime: render_figure_snapshot_with_camera_state.failed handle={} axes={} error={}",
            handle.as_u32(),
            axes_cameras.len(),
            err
        );
        map_control_flow_with_builtin(
            engine_error_with_source(
                format!("Plot export failed: {err}"),
                PlottingBackendError::ImageExport(err),
            ),
            SNAPSHOT_CONTEXT,
        )
    })?;
    log::debug!(
        "runmat-runtime: render_figure_snapshot_with_camera_state.ok handle={} bytes={} axes={}",
        handle.as_u32(),
        bytes.len(),
        axes_cameras.len()
    );
    Ok(bytes)
}

#[cfg(feature = "plot-core")]
fn surface_plot_camera_to_core_camera(
    state: &super::web::PlotCameraState,
) -> runmat_plot::core::Camera {
    let mut camera = runmat_plot::core::Camera::new();
    camera.position = glam::Vec3::new(state.position[0], state.position[1], state.position[2]);
    camera.target = glam::Vec3::new(state.target[0], state.target[1], state.target[2]);
    camera.up = glam::Vec3::new(state.up[0], state.up[1], state.up[2]);
    camera.zoom = state.zoom;
    camera.aspect_ratio = state.aspect_ratio.max(0.000_1);
    camera.projection = match state.projection {
        super::web::PlotCameraProjection::Perspective { fov, near, far } => {
            runmat_plot::core::camera::ProjectionType::Perspective {
                fov,
                near: near.max(1.0e-6),
                far: far.max(near + 1.0e-6),
            }
        }
        super::web::PlotCameraProjection::Orthographic {
            left,
            right,
            bottom,
            top,
            near,
            far,
        } => runmat_plot::core::camera::ProjectionType::Orthographic {
            left,
            right,
            bottom,
            top,
            near,
            far,
        },
    };
    camera.mark_dirty();
    camera
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

#[cfg(all(test, feature = "plot-core"))]
mod tests {
    use super::render_figure_snapshot;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::state::{
        clear_figure, current_figure_handle, reset_hold_state_for_run, PlotTestLockGuard,
    };
    use crate::builtins::plotting::subplot::subplot_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::title::title_builtin;
    use crate::builtins::plotting::xlabel::xlabel_builtin;
    use crate::builtins::plotting::ylabel::ylabel_builtin;
    use futures::executor::block_on;
    use runmat_builtins::{Tensor, Value};

    fn setup_plot_tests() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
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
    fn render_figure_snapshot_supports_margin_style_two_axes_lines() {
        let _guard = setup_plot_tests();
        let x_mm: Vec<f64> = (-30..=30).map(|i| i as f64).collect();
        let y_mm: Vec<f64> = (-25..=25).map(|i| i as f64).collect();
        let centerline: Vec<f64> = x_mm
            .iter()
            .map(|x| 25.0 + 18.0 * (-(x / 11.0).powi(2)).exp())
            .collect();
        let vertical: Vec<f64> = y_mm
            .iter()
            .map(|y| 25.0 + 20.0 * (-(y / 9.0).powi(2)).exp())
            .collect();

        subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(1.0)).expect("subplot 1");
        block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&x_mm)),
            Value::Tensor(tensor_from(&centerline)),
        ]))
        .expect("left plot");
        title_builtin(vec![Value::String("Centerline slice".into())]).expect("left title");
        xlabel_builtin(vec![Value::String("x (mm)".into())]).expect("left xlabel");
        ylabel_builtin(vec![Value::String("temperature (C)".into())]).expect("left ylabel");

        subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).expect("subplot 2");
        block_on(plot_builtin(vec![
            Value::Tensor(tensor_from(&y_mm)),
            Value::Tensor(tensor_from(&vertical)),
        ]))
        .expect("right plot");
        title_builtin(vec![Value::String("Vertical slice through source".into())])
            .expect("right title");
        xlabel_builtin(vec![Value::String("y (mm)".into())]).expect("right xlabel");
        ylabel_builtin(vec![Value::String("temperature (C)".into())]).expect("right ylabel");

        let handle = current_figure_handle();
        let bytes =
            block_on(render_figure_snapshot(handle, 1280, 720, None)).expect("snapshot render");
        assert!(bytes.len() > 1000, "expected nontrivial PNG payload");
    }
}
