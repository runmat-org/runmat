#[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
use super::common::ERR_PLOTTING_UNAVAILABLE;
use runmat_plot::plots::Figure;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn web_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).build()
}

#[allow(dead_code)]
fn web_error_with_source(
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    build_runtime_error(message).with_source(source).build()
}

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
pub(crate) mod wasm {
    use super::*;
    use log::debug;
    use runmat_plot::web::WebRenderer;
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use crate::builtins::plotting::state::{clone_figure, current_figure_revision, FigureHandle};

    runmat_thread_local! {
        static SURFACES: RefCell<HashMap<u32, SurfaceEntry>> = RefCell::new(HashMap::new());
    }

    struct SurfaceEntry {
        renderer: WebRenderer,
        bound_handle: Option<u32>,
        last_revision: Option<u64>,
    }

    pub(super) fn install_surface_impl(surface_id: u32, renderer: WebRenderer) -> BuiltinResult<()> {
        SURFACES.with(|slot| {
            slot.borrow_mut().insert(
                surface_id,
                SurfaceEntry {
                    renderer,
                    bound_handle: None,
                    last_revision: None,
                },
            );
        });
        SURFACES.with(|slot| {
            let keys: Vec<u32> = slot.borrow().keys().copied().collect();
            debug!("plot-web: installed surface surface_id={surface_id} (active_surfaces={keys:?})");
        });
        Ok(())
    }

    pub(super) fn detach_surface_impl(surface_id: u32) {
        SURFACES.with(|slot| {
            slot.borrow_mut().remove(&surface_id);
        });
        SURFACES.with(|slot| {
            let keys: Vec<u32> = slot.borrow().keys().copied().collect();
            debug!("plot-web: detached surface surface_id={surface_id} (active_surfaces={keys:?})");
        });
    }

    pub fn web_renderer_ready() -> bool {
        SURFACES.with(|slot| !slot.borrow().is_empty())
    }

    pub fn render_web_canvas(handle: u32, figure: Figure) -> BuiltinResult<String> {
        // If nothing is currently bound to this handle, try to claim the lowest-id unbound surface.
        let needs_autobind = SURFACES.with(|slot| {
            let map = slot.borrow();
            !map.values().any(|entry| entry.bound_handle == Some(handle))
        });
        if needs_autobind {
            let maybe_unbound_surface = SURFACES.with(|slot| {
                let map = slot.borrow();
                map.iter()
                    .filter_map(|(surface_id, entry)| {
                        if entry.bound_handle.is_none() {
                            Some(*surface_id)
                        } else {
                            None
                        }
                    })
                    .min()
            });
            if let Some(surface_id) = maybe_unbound_surface {
                // Bind without forcing a full re-prime here; the render below will set last_revision.
                let _ = bind_surface_to_figure_impl(surface_id, handle);
            }
        }

        let mut rendered_any = false;
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            for (_surface_id, entry) in map.iter_mut() {
                if entry.bound_handle != Some(handle) {
                    continue;
                }
                rendered_any = true;
                // Figure was just mutated; always (re)load render data for this surface.
                entry
                    .renderer
                    .render_figure(figure.clone())
                    .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
                let rev = current_figure_revision(FigureHandle::from(handle));
                entry.last_revision = rev;
                entry
                    .renderer
                    .render_current_scene()
                    .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
            }
            Ok::<(), RuntimeError>(())
        })
        .map_err(|err| err)?;
        if !rendered_any {
            // It's valid to update a figure when no surfaces are bound to it yet (or no surfaces
            // exist at all). The figure state is still updated + emitted via figure events, and
            // hosts can present it later by binding a surface to this handle.
            return Ok("Plot updated (no bound surfaces)".to_string());
        }
        Ok("Plot rendered to surface".to_string())
    }

    pub(super) fn resize_surface_impl(
        surface_id: u32,
        width: u32,
        height: u32,
        pixels_per_point: f32,
    ) -> BuiltinResult<()> {
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            let entry = map.get_mut(&surface_id).ok_or_else(|| {
                web_error(format!(
                    "Plotting surface {surface_id} not registered. Call createPlotSurface() first."
                ))
            })?;
            entry.renderer.set_pixels_per_point(pixels_per_point);
            entry
                .renderer
                .resize_surface(width, height)
                .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
            Ok(())
        })
    }

    pub(super) fn bind_surface_to_figure_impl(surface_id: u32, handle: u32) -> BuiltinResult<()> {
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            let entry = map.get_mut(&surface_id).ok_or_else(|| {
                web_error(format!(
                    "Plotting surface {surface_id} not registered. Call createPlotSurface() first."
                ))
            })?;
            entry.bound_handle = Some(handle);
            // Force a re-prime on next present.
            entry.last_revision = None;
            Ok(())
        })
    }

    pub(super) fn present_figure_on_surface_impl(surface_id: u32, handle: u32) -> BuiltinResult<()> {
        // "Better" path: only invalidate cached render data if the handle actually changes.
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            let entry = map.get_mut(&surface_id).ok_or_else(|| {
                web_error(format!(
                    "Plotting surface {surface_id} not registered. Call createPlotSurface() first."
                ))
            })?;
            if entry.bound_handle != Some(handle) {
                entry.bound_handle = Some(handle);
                entry.last_revision = None;
            }
            Ok::<(), RuntimeError>(())
        })?;
        present_surface_impl(surface_id)
    }

    pub(super) fn present_surface_impl(surface_id: u32) -> BuiltinResult<()> {
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            let entry = map.get_mut(&surface_id).ok_or_else(|| {
                web_error(format!(
                    "Plotting surface {surface_id} not registered. Call createPlotSurface() first."
                ))
            })?;
            let handle = entry.bound_handle.ok_or_else(|| {
                web_error("Plotting surface is not bound to a figure handle. Call bindSurfaceToFigure().")
            })?;
            // "Better" path: only re-prime render data when the figure revision changed.
            let current_rev = current_figure_revision(FigureHandle::from(handle));
            if entry.last_revision != current_rev {
                let figure = clone_figure(FigureHandle::from(handle)).ok_or_else(|| {
                    web_error(format!("figure handle {handle} does not exist"))
                })?;
                entry
                    .renderer
                    .render_figure(figure)
                    .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
                entry.last_revision = current_rev;
            }
            entry
                .renderer
                .render_current_scene()
                .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
            Ok(())
        })
    }

    pub fn render_current_scene(handle: u32) -> BuiltinResult<()> {
        debug!("plot-web: render_current_scene(handle={handle})");
        // Render any surfaces that are currently bound to this handle.
        let surface_ids: Vec<u32> = SURFACES.with(|slot| {
            slot.borrow()
                .iter()
                .filter_map(|(surface_id, entry)| {
                    if entry.bound_handle == Some(handle) {
                        Some(*surface_id)
                    } else {
                        None
                    }
                })
                .collect()
        });
        if surface_ids.is_empty() {
            // No bound surfaces; nothing to do.
            return Ok(());
        }
        for surface_id in surface_ids {
            // Use caching logic in present_surface so we avoid re-priming unless revision changed.
            present_surface_impl(surface_id)?;
        }
        Ok(())
    }

    // expose type to outer module
    pub(super) use runmat_plot::web::WebRenderer as RendererType;
}

#[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
pub(crate) mod wasm {
    use super::*;

    pub struct RendererPlaceholder;

    pub(super) fn install_surface_impl(
        _surface_id: u32,
        _renderer: RendererPlaceholder,
    ) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) fn detach_surface_impl(_surface_id: u32) {}

    pub fn web_renderer_ready() -> bool {
        false
    }

    #[allow(dead_code)]
    pub fn render_web_canvas(_handle: u32, _figure: Figure) -> BuiltinResult<String> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) use RendererPlaceholder as RendererType;

    pub(super) fn resize_surface_impl(
        _surface_id: u32,
        _width: u32,
        _height: u32,
        _pixels_per_point: f32,
    ) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub fn render_current_scene(_handle: u32) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) fn bind_surface_to_figure_impl(_surface_id: u32, _handle: u32) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) fn present_surface_impl(_surface_id: u32) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) fn present_figure_on_surface_impl(_surface_id: u32, _handle: u32) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }
}

pub use wasm::render_current_scene;
pub use wasm::web_renderer_ready;

pub fn install_surface(surface_id: u32, renderer: wasm::RendererType) -> BuiltinResult<()> {
    wasm::install_surface_impl(surface_id, renderer)
}

pub fn detach_surface(surface_id: u32) {
    wasm::detach_surface_impl(surface_id)
}

pub fn resize_surface(
    surface_id: u32,
    width: u32,
    height: u32,
    pixels_per_point: f32,
) -> BuiltinResult<()> {
    wasm::resize_surface_impl(surface_id, width, height, pixels_per_point)
}

pub fn bind_surface_to_figure(surface_id: u32, handle: u32) -> BuiltinResult<()> {
    wasm::bind_surface_to_figure_impl(surface_id, handle)
}

pub fn present_surface(surface_id: u32) -> BuiltinResult<()> {
    wasm::present_surface_impl(surface_id)
}

pub fn present_figure_on_surface(surface_id: u32, handle: u32) -> BuiltinResult<()> {
    wasm::present_figure_on_surface_impl(surface_id, handle)
}

#[cfg_attr(
    not(all(target_arch = "wasm32", feature = "plot-web")),
    allow(dead_code)
)]
pub(crate) fn render_web_canvas(handle: u32, figure: Figure) -> BuiltinResult<String> {
    wasm::render_web_canvas(handle, figure)
}
