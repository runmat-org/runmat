#[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
use super::common::ERR_PLOTTING_UNAVAILABLE;
use runmat_plot::plots::Figure;

use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow};

fn web_error(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message).build().into()
}

fn web_error_with_source(
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeControlFlow {
    build_runtime_error(message).with_source(source).build().into()
}

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
pub(crate) mod wasm {
    use super::*;
    use log::debug;
    use runmat_plot::web::WebRenderer;
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;

    runmat_thread_local! {
        static WEB_RENDERERS: RefCell<HashMap<u32, WebRenderer>> = RefCell::new(HashMap::new());
        static DEFAULT_RENDERER: RefCell<Option<WebRenderer>> = RefCell::new(None);
    }

    pub fn install_web_renderer(renderer: WebRenderer) -> BuiltinResult<()> {
        DEFAULT_RENDERER.with(|slot| {
            *slot.borrow_mut() = Some(renderer);
        });
        Ok(())
    }

    pub fn install_web_renderer_for_handle(
        handle: u32,
        renderer: WebRenderer,
    ) -> BuiltinResult<()> {
        WEB_RENDERERS.with(|slot| {
            slot.borrow_mut().insert(handle, renderer);
        });
        Ok(())
    }

    pub fn detach_web_renderer(handle: u32) {
        WEB_RENDERERS.with(|slot| {
            slot.borrow_mut().remove(&handle);
        });
    }

    pub fn detach_default_renderer() {
        DEFAULT_RENDERER.with(|slot| {
            slot.borrow_mut().take();
        });
    }

    pub fn web_renderer_ready() -> bool {
        WEB_RENDERERS.with(|slot| !slot.borrow().is_empty())
            || DEFAULT_RENDERER.with(|slot| slot.borrow().is_some())
    }

    pub fn render_web_canvas(handle: u32, figure: Figure) -> BuiltinResult<String> {
        with_renderer(handle, |renderer| renderer.render_figure(figure))
            .map(|_| "Plot rendered to canvas".to_string())
    }

    pub fn resize_web_renderer(handle: u32, width: u32, height: u32) -> BuiltinResult<()> {
        with_renderer(handle, |renderer| renderer.resize_surface(width, height)).map(|_| ())
    }

    pub fn render_current_scene(handle: u32) -> BuiltinResult<()> {
        debug!("plot-web: render_current_scene(handle={handle})");
        with_renderer(handle, |renderer| renderer.render_current_scene()).map(|_| ())
    }

    fn with_renderer<F, R>(handle: u32, f: F) -> BuiltinResult<R>
    where
        F: FnOnce(&mut WebRenderer) -> Result<R, runmat_plot::web::WebRendererError>,
    {
        WEB_RENDERERS.with(|map_cell| {
            let mut map = map_cell.borrow_mut();
            if let Some(renderer) = map.get_mut(&handle) {
                return f(renderer)
                    .map_err(|err| web_error_with_source("Plotting failed.", err));
            }
            drop(map);
            DEFAULT_RENDERER.with(|default_cell| {
                let mut default = default_cell.borrow_mut();
                let renderer = default.as_mut().ok_or_else(|| {
                    web_error(
                        "RunMat plotting canvas not registered. Call registerPlotCanvas() before plotting.",
                    )
                })?;
                f(renderer).map_err(|err| web_error_with_source("Plotting failed.", err))
            })
        })
    }

    // expose type to outer module
    pub(super) use runmat_plot::web::WebRenderer as RendererType;
}

#[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
pub(crate) mod wasm {
    use super::*;

    pub struct RendererPlaceholder;

    pub fn install_web_renderer(_renderer: RendererPlaceholder) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub fn install_web_renderer_for_handle(
        _handle: u32,
        _renderer: RendererPlaceholder,
    ) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub fn detach_web_renderer(_handle: u32) {}

    pub fn detach_default_renderer() {}

    pub fn web_renderer_ready() -> bool {
        false
    }

    #[allow(dead_code)]
    pub fn render_web_canvas(_handle: u32, _figure: Figure) -> BuiltinResult<String> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) use RendererPlaceholder as RendererType;

    pub fn resize_web_renderer(_handle: u32, _width: u32, _height: u32) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub fn render_current_scene(_handle: u32) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }
}

pub use wasm::render_current_scene;
pub use wasm::resize_web_renderer;
pub use wasm::web_renderer_ready;

pub fn install_web_renderer(renderer: wasm::RendererType) -> BuiltinResult<()> {
    wasm::install_web_renderer(renderer)
}

pub fn install_web_renderer_for_handle(
    handle: u32,
    renderer: wasm::RendererType,
) -> BuiltinResult<()> {
    wasm::install_web_renderer_for_handle(handle, renderer)
}

pub fn detach_web_renderer(handle: u32) {
    wasm::detach_web_renderer(handle)
}

pub fn detach_default_web_renderer() {
    wasm::detach_default_renderer()
}

#[cfg_attr(
    not(all(target_arch = "wasm32", feature = "plot-web")),
    allow(dead_code)
)]
pub(crate) fn render_web_canvas(handle: u32, figure: Figure) -> BuiltinResult<String> {
    wasm::render_web_canvas(handle, figure)
}
