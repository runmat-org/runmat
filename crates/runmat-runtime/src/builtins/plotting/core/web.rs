#[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
use super::common::ERR_PLOTTING_UNAVAILABLE;
use runmat_plot::plots::Figure;

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
pub(crate) mod wasm {
    use super::*;
    use runmat_plot::web::WebRenderer;
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        static WEB_RENDERERS: RefCell<HashMap<u32, WebRenderer>> = RefCell::new(HashMap::new());
        static DEFAULT_RENDERER: RefCell<Option<WebRenderer>> = RefCell::new(None);
    }

    pub fn install_web_renderer(renderer: WebRenderer) -> Result<(), String> {
        DEFAULT_RENDERER.with(|slot| {
            *slot.borrow_mut() = Some(renderer);
        });
        Ok(())
    }

    pub fn install_web_renderer_for_handle(
        handle: u32,
        renderer: WebRenderer,
    ) -> Result<(), String> {
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

    pub fn render_web_canvas(handle: u32, figure: Figure) -> Result<String, String> {
        with_renderer(handle, |renderer| renderer.render_figure(figure))
            .map(|_| "Plot rendered to canvas".to_string())
    }

    fn with_renderer<F, R>(handle: u32, f: F) -> Result<R, String>
    where
        F: FnOnce(&mut WebRenderer) -> Result<R, runmat_plot::web::WebRendererError>,
    {
        WEB_RENDERERS.with(|map_cell| {
            let mut map = map_cell.borrow_mut();
            if let Some(renderer) = map.get_mut(&handle) {
                return f(renderer).map_err(|err| format!("Plotting failed: {err}"));
            }
            drop(map);
            DEFAULT_RENDERER.with(|default_cell| {
                let mut default = default_cell.borrow_mut();
                let renderer = default.as_mut().ok_or_else(|| {
                    "RunMat plotting canvas not registered. Call registerPlotCanvas() before plotting."
                        .to_string()
                })?;
                f(renderer).map_err(|err| format!("Plotting failed: {err}"))
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

    pub fn install_web_renderer(_renderer: RendererPlaceholder) -> Result<(), String> {
        Err(ERR_PLOTTING_UNAVAILABLE.to_string())
    }

    pub fn install_web_renderer_for_handle(
        _handle: u32,
        _renderer: RendererPlaceholder,
    ) -> Result<(), String> {
        Err(ERR_PLOTTING_UNAVAILABLE.to_string())
    }

    pub fn detach_web_renderer(_handle: u32) {}

    pub fn detach_default_renderer() {}

    pub fn web_renderer_ready() -> bool {
        false
    }

    #[allow(dead_code)]
    pub fn render_web_canvas(_handle: u32, _figure: Figure) -> Result<String, String> {
        Err(ERR_PLOTTING_UNAVAILABLE.to_string())
    }

    pub(super) use RendererPlaceholder as RendererType;
}

pub use wasm::web_renderer_ready;

pub fn install_web_renderer(renderer: wasm::RendererType) -> Result<(), String> {
    wasm::install_web_renderer(renderer)
}

pub fn install_web_renderer_for_handle(
    handle: u32,
    renderer: wasm::RendererType,
) -> Result<(), String> {
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
pub(crate) fn render_web_canvas(handle: u32, figure: Figure) -> Result<String, String> {
    wasm::render_web_canvas(handle, figure)
}
