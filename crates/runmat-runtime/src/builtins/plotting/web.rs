use super::common::ERR_PLOTTING_UNAVAILABLE;
use runmat_plot::plots::Figure;

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
mod wasm {
    use super::*;
    use runmat_plot::web::WebRenderer;
    use std::cell::RefCell;

    thread_local! {
        static WEB_RENDERER: RefCell<Option<WebRenderer>> = RefCell::new(None);
    }

    pub fn install_web_renderer(renderer: WebRenderer) -> Result<(), String> {
        WEB_RENDERER.with(|slot| {
            *slot.borrow_mut() = Some(renderer);
        });
        Ok(())
    }

    pub fn web_renderer_ready() -> bool {
        WEB_RENDERER.with(|slot| slot.borrow().is_some())
    }

    pub fn render_web_canvas(mut figure: Figure) -> Result<String, String> {
        with_renderer(|renderer| renderer.render_figure(figure))
            .map(|_| "Plot rendered to canvas".to_string())
    }

    fn with_renderer<F, R>(f: F) -> Result<R, String>
    where
        F: FnOnce(&mut WebRenderer) -> Result<R, runmat_plot::web::WebRendererError>,
    {
        WEB_RENDERER.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let renderer = borrow.as_mut().ok_or_else(|| {
                "RunMat plotting canvas not registered. Call registerPlotCanvas() before plotting."
                    .to_string()
            })?;
            f(renderer).map_err(|err| format!("Plotting failed: {err}"))
        })
    }

    // expose type to outer module
    pub(super) use runmat_plot::web::WebRenderer as RendererType;
}

#[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
mod wasm {
    use super::*;

    pub struct RendererPlaceholder;

    pub fn install_web_renderer(_renderer: RendererPlaceholder) -> Result<(), String> {
        Err(ERR_PLOTTING_UNAVAILABLE.to_string())
    }

    pub fn web_renderer_ready() -> bool {
        false
    }

    #[allow(dead_code)]
    pub fn render_web_canvas(_figure: Figure) -> Result<String, String> {
        Err(ERR_PLOTTING_UNAVAILABLE.to_string())
    }

    pub(super) use RendererPlaceholder as RendererType;
}

pub use wasm::web_renderer_ready;

pub fn install_web_renderer(renderer: wasm::RendererType) -> Result<(), String> {
    wasm::install_web_renderer(renderer)
}

#[cfg_attr(
    not(all(target_arch = "wasm32", feature = "plot-web")),
    allow(dead_code)
)]
pub(crate) fn render_web_canvas(figure: Figure) -> Result<String, String> {
    wasm::render_web_canvas(figure)
}
