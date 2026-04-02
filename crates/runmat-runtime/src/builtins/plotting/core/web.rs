#[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
use super::common::ERR_PLOTTING_UNAVAILABLE;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlotSurfaceCameraState {
    pub active_axes: usize,
    pub axes: Vec<PlotCameraState>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlotCameraState {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub zoom: f32,
    pub aspect_ratio: f32,
    pub projection: PlotCameraProjection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PlotCameraProjection {
    Perspective {
        fov: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    },
}

fn web_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:plot:WebError")
        .build()
}

#[allow(dead_code)]
fn web_error_with_source(
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:plot:WebError")
        .with_source(source)
        .build()
}

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
pub(crate) mod wasm {
    use super::*;
    use crate::builtins::plotting::state::{clone_figure, current_figure_revision, FigureHandle};
    use log::debug;
    use runmat_plot::core::PlotEvent;
    use runmat_plot::styling::PlotThemeConfig;
    use runmat_plot::web::WebRenderer;
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;

    runmat_thread_local! {
        static SURFACES: RefCell<HashMap<u32, SurfaceEntry>> = RefCell::new(HashMap::new());
        static ACTIVE_THEME: RefCell<PlotThemeConfig> = RefCell::new(PlotThemeConfig::default());
    }

    struct SurfaceEntry {
        renderer: WebRenderer,
        bound_handle: Option<u32>,
        last_revision: Option<u64>,
    }

    pub(super) fn install_surface_impl(
        surface_id: u32,
        mut renderer: WebRenderer,
    ) -> BuiltinResult<()> {
        ACTIVE_THEME.with(|theme| {
            renderer.set_theme_config(theme.borrow().clone());
        });
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
            debug!(
                "plot-web: installed surface surface_id={surface_id} (active_surfaces={keys:?})"
            );
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

    pub(super) fn clear_closed_figure_surfaces_impl(handle: u32) -> BuiltinResult<()> {
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            for entry in map.values_mut() {
                if entry.bound_handle == Some(handle) {
                    entry.bound_handle = None;
                    entry.last_revision = None;
                    entry
                        .renderer
                        .clear_surface()
                        .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
                }
            }
            Ok(())
        })
    }

    pub fn web_renderer_ready() -> bool {
        SURFACES.with(|slot| !slot.borrow().is_empty())
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

    pub(super) fn set_theme_config_impl(theme: PlotThemeConfig) -> BuiltinResult<()> {
        debug!(
            "plot-web: runtime set_theme_config_impl variant={:?} custom_colors={}",
            theme.variant,
            theme.custom_colors.is_some()
        );
        ACTIVE_THEME.with(|slot| {
            *slot.borrow_mut() = theme.clone();
        });
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            debug!("plot-web: applying theme to {} surfaces", map.len());
            for entry in map.values_mut() {
                entry.renderer.set_theme_config(theme.clone());
                if let Some(handle) = entry.bound_handle {
                    if let Some(figure) = clone_figure(FigureHandle::from(handle)) {
                        entry
                            .renderer
                            .render_figure(figure)
                            .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
                    }
                }
            }
            Ok(())
        })
    }

    pub(super) fn present_figure_on_surface_impl(
        surface_id: u32,
        handle: u32,
    ) -> BuiltinResult<()> {
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
                web_error(
                    "Plotting surface is not bound to a figure handle. Call bindSurfaceToFigure().",
                )
            })?;
            // "Better" path: only re-prime render data when the figure revision changed.
            let current_rev = current_figure_revision(FigureHandle::from(handle));
            if entry.last_revision != current_rev {
                let figure = clone_figure(FigureHandle::from(handle))
                    .ok_or_else(|| web_error(format!("figure handle {handle} does not exist")))?;
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

    pub(super) fn handle_surface_event_impl(
        surface_id: u32,
        event: PlotEvent,
    ) -> BuiltinResult<()> {
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            let entry = map.get_mut(&surface_id).ok_or_else(|| {
                web_error(format!(
                    "Plotting surface {surface_id} not registered. Call createPlotSurface() first."
                ))
            })?;
            match &event {
                PlotEvent::MousePress { .. }
                | PlotEvent::MouseRelease { .. }
                | PlotEvent::MouseWheel { .. } => {
                    debug!("plot-web: surface_event(surface_id={surface_id}, event={event:?})");
                }
                PlotEvent::MouseMove { .. } | PlotEvent::Resize { .. } => {}
                PlotEvent::KeyPress { .. } | PlotEvent::KeyRelease { .. } => {}
            }
            // If no figure was ever rendered, there's nothing to manipulate.
            // Still accept the event (no-op) so the host doesn't have to special-case.
            let _ = entry.renderer.handle_event(event);
            // Camera interactions should re-render immediately, without requiring a figure revision bump.
            entry
                .renderer
                .render_current_scene()
                .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
            Ok(())
        })
    }

    pub(super) fn fit_surface_extents_impl(surface_id: u32) -> BuiltinResult<()> {
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            let entry = map.get_mut(&surface_id).ok_or_else(|| {
                web_error(format!(
                    "Plotting surface {surface_id} not registered. Call createPlotSurface() first."
                ))
            })?;
            entry.renderer.fit_extents();
            entry
                .renderer
                .render_current_scene()
                .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
            Ok(())
        })
    }

    pub(super) fn reset_surface_camera_impl(surface_id: u32) -> BuiltinResult<()> {
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            let entry = map.get_mut(&surface_id).ok_or_else(|| {
                web_error(format!(
                    "Plotting surface {surface_id} not registered. Call createPlotSurface() first."
                ))
            })?;
            entry.renderer.reset_camera_position();
            entry
                .renderer
                .render_current_scene()
                .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
            Ok(())
        })
    }

    pub(super) fn get_surface_camera_state_impl(
        surface_id: u32,
    ) -> BuiltinResult<PlotSurfaceCameraState> {
        SURFACES.with(|slot| {
            let map = slot.borrow();
            let entry = map.get(&surface_id).ok_or_else(|| {
                web_error(format!(
                    "Plotting surface {surface_id} not registered. Call createPlotSurface() first."
                ))
            })?;
            Ok(convert_camera_state(entry.renderer.camera_state()))
        })
    }

    pub(super) fn set_surface_camera_state_impl(
        surface_id: u32,
        state: PlotSurfaceCameraState,
    ) -> BuiltinResult<()> {
        SURFACES.with(|slot| {
            let mut map = slot.borrow_mut();
            let entry = map.get_mut(&surface_id).ok_or_else(|| {
                web_error(format!(
                    "Plotting surface {surface_id} not registered. Call createPlotSurface() first."
                ))
            })?;
            entry
                .renderer
                .set_camera_state(&convert_camera_state_back(state));
            entry
                .renderer
                .render_current_scene()
                .map_err(|err| web_error(format!("Plotting failed: {err}")))?;
            Ok(())
        })
    }

    pub fn render_current_scene(handle: u32) -> BuiltinResult<()> {
        debug!("plot-web: render_current_scene(handle={handle})");
        // If nothing is currently bound to this handle, try to claim the lowest-id unbound
        // surface. This ensures `drawnow()` / `pause()` can present even if the host hasn't
        // explicitly bound a surface yet.
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
                // Bind without forcing a full re-prime here; present_surface will set last_revision.
                let _ = bind_surface_to_figure_impl(surface_id, handle);
            }
        }

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

    pub(super) fn clear_closed_figure_surfaces_impl(_handle: u32) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub fn web_renderer_ready() -> bool {
        false
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

    pub(super) fn present_figure_on_surface_impl(
        _surface_id: u32,
        _handle: u32,
    ) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) fn fit_surface_extents_impl(_surface_id: u32) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) fn reset_surface_camera_impl(_surface_id: u32) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) fn get_surface_camera_state_impl(
        _surface_id: u32,
    ) -> BuiltinResult<PlotSurfaceCameraState> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) fn set_surface_camera_state_impl(
        _surface_id: u32,
        _state: PlotSurfaceCameraState,
    ) -> BuiltinResult<()> {
        Err(web_error(ERR_PLOTTING_UNAVAILABLE))
    }

    pub(super) fn set_theme_config_impl(
        _theme: runmat_plot::styling::PlotThemeConfig,
    ) -> BuiltinResult<()> {
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

pub fn clear_closed_figure_surfaces(handle: u32) -> BuiltinResult<()> {
    wasm::clear_closed_figure_surfaces_impl(handle)
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

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
pub fn handle_plot_surface_event(
    surface_id: u32,
    event: runmat_plot::core::PlotEvent,
) -> BuiltinResult<()> {
    wasm::handle_surface_event_impl(surface_id, event)
}

pub fn present_figure_on_surface(surface_id: u32, handle: u32) -> BuiltinResult<()> {
    wasm::present_figure_on_surface_impl(surface_id, handle)
}

pub fn fit_surface_extents(surface_id: u32) -> BuiltinResult<()> {
    wasm::fit_surface_extents_impl(surface_id)
}

pub fn reset_surface_camera(surface_id: u32) -> BuiltinResult<()> {
    wasm::reset_surface_camera_impl(surface_id)
}

pub fn get_surface_camera_state(surface_id: u32) -> BuiltinResult<PlotSurfaceCameraState> {
    wasm::get_surface_camera_state_impl(surface_id)
}

pub fn set_surface_camera_state(
    surface_id: u32,
    state: PlotSurfaceCameraState,
) -> BuiltinResult<()> {
    wasm::set_surface_camera_state_impl(surface_id, state)
}

pub fn set_plot_theme_config(theme: runmat_plot::styling::PlotThemeConfig) -> BuiltinResult<()> {
    wasm::set_theme_config_impl(theme)
}

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
fn convert_camera_state(state: runmat_plot::web::PlotSurfaceCameraState) -> PlotSurfaceCameraState {
    PlotSurfaceCameraState {
        active_axes: state.active_axes,
        axes: state
            .axes
            .into_iter()
            .map(|camera| PlotCameraState {
                position: camera.position,
                target: camera.target,
                up: camera.up,
                zoom: camera.zoom,
                aspect_ratio: camera.aspect_ratio,
                projection: match camera.projection {
                    runmat_plot::web::PlotCameraProjection::Perspective { fov, near, far } => {
                        PlotCameraProjection::Perspective { fov, near, far }
                    }
                    runmat_plot::web::PlotCameraProjection::Orthographic {
                        left,
                        right,
                        bottom,
                        top,
                        near,
                        far,
                    } => PlotCameraProjection::Orthographic {
                        left,
                        right,
                        bottom,
                        top,
                        near,
                        far,
                    },
                },
            })
            .collect(),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
fn convert_camera_state_back(
    state: PlotSurfaceCameraState,
) -> runmat_plot::web::PlotSurfaceCameraState {
    runmat_plot::web::PlotSurfaceCameraState {
        active_axes: state.active_axes,
        axes: state
            .axes
            .into_iter()
            .map(|camera| runmat_plot::web::PlotCameraState {
                position: camera.position,
                target: camera.target,
                up: camera.up,
                zoom: camera.zoom,
                aspect_ratio: camera.aspect_ratio,
                projection: match camera.projection {
                    PlotCameraProjection::Perspective { fov, near, far } => {
                        runmat_plot::web::PlotCameraProjection::Perspective { fov, near, far }
                    }
                    PlotCameraProjection::Orthographic {
                        left,
                        right,
                        bottom,
                        top,
                        near,
                        far,
                    } => runmat_plot::web::PlotCameraProjection::Orthographic {
                        left,
                        right,
                        bottom,
                        top,
                        near,
                        far,
                    },
                },
            })
            .collect(),
    }
}

// No render_web_canvas wrapper; web presentation is surface-driven.
