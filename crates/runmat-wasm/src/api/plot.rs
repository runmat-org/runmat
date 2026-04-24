use std::sync::Arc;

use glam::Vec2;
use js_sys::{Reflect, Uint8Array};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use runmat_accelerate_api::{AccelContextHandle, AccelContextKind};
use runmat_plot::{
    event::{
        FigureEvent as PlotFigureEvent, FigureEventKind as PlotFigureEventKind, FigureSnapshot,
    },
    styling::{
        config::{
            CustomColorConfig, GridConfig, InteractionConfig, LayoutConfig, TypographyConfig,
        },
        PlotThemeConfig, ThemeVariant,
    },
    web::{WebCanvas, WebRenderer, WebRendererOptions},
    SharedWgpuContext,
};
use runmat_runtime::builtins::plotting::{
    bind_surface_to_figure as runtime_bind_surface_to_figure,
    clear_closed_figure_surfaces as runtime_clear_closed_figure_surfaces,
    clear_figure as runtime_clear_figure, close_figure as runtime_close_figure,
    configure_subplot as runtime_configure_subplot,
    current_axes_state as runtime_current_axes_state,
    current_figure_handle as runtime_current_figure_handle,
    detach_surface as runtime_detach_surface, fit_surface_extents as runtime_fit_surface_extents,
    get_surface_camera_state as runtime_get_surface_camera_state,
    handle_plot_surface_event as runtime_handle_plot_surface_event,
    install_figure_observer as runtime_install_figure_observer,
    install_surface as runtime_install_surface, new_figure_handle as runtime_new_figure_handle,
    present_figure_on_surface as runtime_present_figure_on_surface,
    present_surface as runtime_present_surface,
    render_current_scene as runtime_render_current_scene,
    render_figure_snapshot as runtime_render_figure_snapshot,
    render_figure_snapshot_with_camera_state as runtime_render_figure_snapshot_with_camera_state,
    reset_surface_camera as runtime_reset_surface_camera, resize_surface as runtime_resize_surface,
    select_figure as runtime_select_figure, set_hold as runtime_set_hold,
    set_plot_theme_config as runtime_set_plot_theme_config,
    set_surface_camera_state as runtime_set_surface_camera_state,
    web_renderer_ready as runtime_plot_renderer_ready, FigureAxesState, FigureError,
    FigureEventKind, FigureEventView, FigureHandle, HoldMode, PlotSurfaceCameraState,
};
use runmat_runtime::RuntimeError;

use crate::runtime::logging::init_logging_once;
use crate::runtime::state::{
    figure_event_callback, replace_figure_event_callback, FIGURE_EVENT_OBSERVER,
    LEGACY_FIGURE_SURFACES, LEGACY_PLOT_SURFACE_ID, PLOT_SURFACE_NEXT_ID,
};
use crate::wire::errors::{
    init_error_with_details, js_error, runtime_error_payload, runtime_error_to_js, InitErrorCode,
};

#[cfg(target_arch = "wasm32")]
pub(crate) fn ensure_figure_event_bridge() {
    FIGURE_EVENT_OBSERVER.get_or_init(|| {
        let observer: Arc<dyn for<'a> Fn(FigureEventView<'a>) + Send + Sync> =
            Arc::new(emit_js_figure_event);
        let _ = runtime_install_figure_observer(observer);
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = registerPlotCanvas)]
pub async fn register_plot_canvas(canvas: JsValue) -> Result<(), JsValue> {
    let canvas = parse_web_canvas(canvas)?;
    let surface_id = PLOT_SURFACE_NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    install_surface_renderer(surface_id, canvas)
        .await
        .map_err(|err| {
            init_error_with_details(
                InitErrorCode::PlotCanvas,
                "Failed to register plot canvas",
                Some(err),
            )
        })?;
    LEGACY_PLOT_SURFACE_ID.with(|slot| {
        slot.borrow_mut().replace(surface_id);
    });
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = registerFigureCanvas)]
pub async fn register_figure_canvas(handle: u32, canvas: JsValue) -> Result<(), JsValue> {
    let canvas = parse_web_canvas(canvas)?;
    let surface_id = PLOT_SURFACE_NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    install_surface_renderer(surface_id, canvas)
        .await
        .map_err(|err| {
            init_error_with_details(
                InitErrorCode::PlotCanvas,
                "Failed to register figure canvas",
                Some(err),
            )
        })?;
    runtime_bind_surface_to_figure(surface_id, handle).map_err(|err| runtime_error_to_js(&err))?;
    LEGACY_FIGURE_SURFACES.with(|slot| {
        slot.borrow_mut().insert(handle, surface_id);
    });
    let _ = runtime_render_current_scene(handle);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = deregisterPlotCanvas)]
pub fn deregister_plot_canvas() {
    let surface_id = LEGACY_PLOT_SURFACE_ID.with(|slot| slot.borrow_mut().take());
    if let Some(id) = surface_id {
        runtime_detach_surface(id);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = deregisterFigureCanvas)]
pub fn deregister_figure_canvas(handle: u32) {
    let surface_id = LEGACY_FIGURE_SURFACES.with(|slot| slot.borrow_mut().remove(&handle));
    if let Some(id) = surface_id {
        runtime_detach_surface(id);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = resizeFigureCanvas)]
pub fn resize_figure_canvas(handle: u32, width: u32, height: u32) -> Result<(), JsValue> {
    let surface_id = LEGACY_FIGURE_SURFACES.with(|slot| slot.borrow().get(&handle).copied());
    let Some(surface_id) = surface_id else {
        return Err(js_error("Figure canvas not registered"));
    };
    runtime_resize_surface(surface_id, width.max(1), height.max(1), 1.0)
        .map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = renderCurrentFigureScene)]
pub fn render_current_figure_scene(handle: u32) -> Result<(), JsValue> {
    runtime_render_current_scene(handle).map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createPlotSurface)]
pub async fn create_plot_surface(canvas: JsValue) -> Result<u32, JsValue> {
    let canvas = parse_web_canvas(canvas)?;
    let surface_id = PLOT_SURFACE_NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    install_surface_renderer(surface_id, canvas).await?;
    Ok(surface_id)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = destroyPlotSurface)]
pub fn destroy_plot_surface(surface_id: u32) {
    runtime_detach_surface(surface_id);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = resizePlotSurface)]
pub fn resize_plot_surface(
    surface_id: u32,
    width: u32,
    height: u32,
    pixels_per_point: f32,
) -> Result<(), JsValue> {
    runtime_resize_surface(surface_id, width.max(1), height.max(1), pixels_per_point)
        .map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bindSurfaceToFigure)]
pub fn bind_surface_to_figure(surface_id: u32, handle: u32) -> Result<(), JsValue> {
    runtime_bind_surface_to_figure(surface_id, handle).map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = presentSurface)]
pub fn present_surface(surface_id: u32) -> Result<(), JsValue> {
    runtime_present_surface(surface_id).map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = presentFigureOnSurface)]
pub fn present_figure_on_surface(surface_id: u32, handle: u32) -> Result<(), JsValue> {
    runtime_present_figure_on_surface(surface_id, handle).map_err(|err| runtime_error_to_js(&err))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlotSurfaceEventPayload {
    kind: String,
    x: f32,
    y: f32,
    #[serde(default)]
    dx: f32,
    #[serde(default)]
    dy: f32,
    #[serde(default)]
    button: i32,
    #[serde(default)]
    wheel_delta_x: f32,
    #[serde(default)]
    wheel_delta_y: f32,
    #[serde(default)]
    wheel_delta_mode: u32,
    #[serde(default)]
    shift_key: bool,
    #[serde(default)]
    ctrl_key: bool,
    #[serde(default)]
    alt_key: bool,
    #[serde(default)]
    meta_key: bool,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = handlePlotSurfaceEvent)]
pub fn handle_plot_surface_event(surface_id: u32, event: JsValue) -> Result<(), JsValue> {
    use runmat_plot::core::interaction::Modifiers as PlotModifiers;
    use runmat_plot::core::interaction::MouseButton as PlotMouseButton;
    use runmat_plot::core::PlotEvent;

    let payload: PlotSurfaceEventPayload =
        serde_wasm_bindgen::from_value(event).map_err(|err| js_error(&err.to_string()))?;
    let position = Vec2::new(payload.x, payload.y);
    let delta = Vec2::new(payload.dx, payload.dy);
    let button = match payload.button {
        2 => PlotMouseButton::Right,
        1 => PlotMouseButton::Middle,
        _ => PlotMouseButton::Left,
    };
    let modifiers = PlotModifiers {
        shift: payload.shift_key,
        ctrl: payload.ctrl_key,
        alt: payload.alt_key,
        meta: payload.meta_key,
    };

    let event = match payload.kind.as_str() {
        "mouseDown" => PlotEvent::MousePress {
            position,
            button,
            modifiers,
        },
        "mouseUp" => PlotEvent::MouseRelease {
            position,
            button,
            modifiers,
        },
        "mouseMove" => PlotEvent::MouseMove {
            position,
            delta,
            modifiers,
        },
        "wheel" => {
            let mut wheel_delta_x = payload.wheel_delta_x;
            let mut wheel_delta_y = payload.wheel_delta_y;
            match payload.wheel_delta_mode {
                0 => {
                    wheel_delta_x /= 100.0;
                    wheel_delta_y /= 100.0;
                }
                1 => {}
                2 => {
                    wheel_delta_x *= 10.0;
                    wheel_delta_y *= 10.0;
                }
                _ => {}
            }
            wheel_delta_x = -wheel_delta_x;
            wheel_delta_y = -wheel_delta_y;
            PlotEvent::MouseWheel {
                position,
                delta: Vec2::new(wheel_delta_x, wheel_delta_y),
                modifiers,
            }
        }
        other => {
            let message = format!("Unknown plot event kind '{other}'");
            return Err(js_error(&message));
        }
    };

    runtime_handle_plot_surface_event(surface_id, event).map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "fitPlotSurfaceExtents")]
pub fn fit_plot_surface_extents(surface_id: u32) -> Result<(), JsValue> {
    runtime_fit_surface_extents(surface_id).map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "resetPlotSurfaceCamera")]
pub fn reset_plot_surface_camera(surface_id: u32) -> Result<(), JsValue> {
    runtime_reset_surface_camera(surface_id).map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "getPlotSurfaceCameraState")]
pub fn get_plot_surface_camera_state(surface_id: u32) -> Result<JsValue, JsValue> {
    let state =
        runtime_get_surface_camera_state(surface_id).map_err(|err| runtime_error_to_js(&err))?;
    serde_wasm_bindgen::to_value(&state).map_err(|err| js_error(&err.to_string()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "setPlotSurfaceCameraState")]
pub fn set_plot_surface_camera_state(surface_id: u32, state: JsValue) -> Result<(), JsValue> {
    let parsed: PlotSurfaceCameraState =
        serde_wasm_bindgen::from_value(state).map_err(|err| js_error(&err.to_string()))?;
    runtime_set_surface_camera_state(surface_id, parsed).map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Deserialize)]
struct PlotThemeConfigPatch {
    variant: Option<ThemeVariant>,
    custom_colors: Option<CustomColorConfig>,
    typography: Option<TypographyConfig>,
    layout: Option<LayoutConfig>,
    grid: Option<GridConfig>,
    interaction: Option<InteractionConfig>,
}

#[cfg(target_arch = "wasm32")]
fn normalize_plot_theme_config(value: JsValue) -> Result<PlotThemeConfig, JsValue> {
    if let Ok(full) = serde_wasm_bindgen::from_value::<PlotThemeConfig>(value.clone()) {
        return Ok(full);
    }
    let patch: PlotThemeConfigPatch = serde_wasm_bindgen::from_value(value)
        .map_err(|err| js_error(&format!("Invalid plot theme config payload: {err}")))?;
    let mut normalized = PlotThemeConfig::default();
    if let Some(variant) = patch.variant {
        normalized.variant = variant;
    }
    if let Some(custom_colors) = patch.custom_colors {
        normalized.custom_colors = Some(custom_colors);
    }
    if let Some(typography) = patch.typography {
        normalized.typography = typography;
    }
    if let Some(layout) = patch.layout {
        normalized.layout = layout;
    }
    if let Some(grid) = patch.grid {
        normalized.grid = grid;
    }
    if let Some(interaction) = patch.interaction {
        normalized.interaction = interaction;
    }
    Ok(normalized)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "setPlotThemeConfig")]
pub fn set_plot_theme_config(theme: JsValue) -> Result<(), JsValue> {
    let normalized = normalize_plot_theme_config(theme)?;
    log::info!(
        "plot-web: setPlotThemeConfig variant={:?} custom_colors={}",
        normalized.variant,
        normalized.custom_colors.is_some()
    );
    normalized
        .validate()
        .map_err(|err| js_error(&format!("Invalid plot theme config: {err}")))?;
    runtime_set_plot_theme_config(normalized).map_err(|err| runtime_error_to_js(&err))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = newFigureHandle)]
pub fn wasm_new_figure_handle() -> u32 {
    runtime_new_figure_handle().as_u32()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = selectFigure)]
pub fn wasm_select_figure(handle: u32) {
    runtime_select_figure(FigureHandle::from(handle));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = currentFigureHandle)]
pub fn wasm_current_figure_handle() -> u32 {
    runtime_current_figure_handle().as_u32()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = configureSubplot)]
pub fn wasm_configure_subplot(rows: u32, cols: u32, index: u32) -> Result<(), JsValue> {
    runtime_configure_subplot(rows as usize, cols as usize, index as usize)
        .map_err(figure_error_to_js)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = setHoldMode)]
pub fn wasm_set_hold_mode(mode: JsValue) -> Result<bool, JsValue> {
    let parsed = parse_hold_mode(mode)?;
    Ok(runtime_set_hold(parsed))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = clearFigure)]
pub fn wasm_clear_figure(handle: JsValue) -> Result<u32, JsValue> {
    let target = parse_optional_handle(handle)?;
    let cleared = runtime_clear_figure(target).map_err(figure_error_to_js)?;
    Ok(cleared.as_u32())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = closeFigure)]
pub fn wasm_close_figure(handle: JsValue) -> Result<u32, JsValue> {
    let target = parse_optional_handle(handle)?;
    let closed = runtime_close_figure(target).map_err(figure_error_to_js)?;
    Ok(closed.as_u32())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = currentAxesInfo)]
pub fn wasm_current_axes_info() -> JsValue {
    axes_state_to_js(runtime_current_axes_state())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = renderFigureImage)]
pub async fn wasm_render_figure_image(
    handle: JsValue,
    width: Option<u32>,
    height: Option<u32>,
    textmark: Option<String>,
) -> Result<Uint8Array, JsValue> {
    const DEFAULT_PREVIEW_WIDTH: u32 = 1280;
    const DEFAULT_PREVIEW_HEIGHT: u32 = 720;
    let _ = shared_webgpu_context();
    let target = parse_optional_handle(handle)?.unwrap_or_else(runtime_current_figure_handle);
    let normalized_width = width.unwrap_or(0).max(1);
    let normalized_height = height.unwrap_or(0).max(1);
    let (render_width, render_height) = if normalized_width == 1 && normalized_height == 1 {
        (DEFAULT_PREVIEW_WIDTH, DEFAULT_PREVIEW_HEIGHT)
    } else {
        (normalized_width, normalized_height)
    };
    log::debug!(
        "RunMat wasm: renderFigureImage start handle={} width={} height={} textmark={}",
        target.as_u32(),
        render_width,
        render_height,
        textmark.as_deref().unwrap_or("")
    );
    let bytes = runtime_render_figure_snapshot(target, render_width, render_height, textmark)
        .await
        .map_err(runtime_flow_to_js)?;
    log::debug!(
        "RunMat wasm: renderFigureImage ok handle={} bytes={}",
        target.as_u32(),
        bytes.len()
    );
    Ok(Uint8Array::from(bytes.as_slice()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = renderFigureImageWithTextmark)]
pub async fn wasm_render_figure_image_with_textmark(
    handle: JsValue,
    width: Option<u32>,
    height: Option<u32>,
    textmark: Option<String>,
) -> Result<Uint8Array, JsValue> {
    wasm_render_figure_image(handle, width, height, textmark).await
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = renderFigureImageWithCameraState)]
pub async fn wasm_render_figure_image_with_camera_state(
    handle: JsValue,
    width: Option<u32>,
    height: Option<u32>,
    camera_state: JsValue,
    textmark: Option<String>,
) -> Result<Uint8Array, JsValue> {
    const DEFAULT_PREVIEW_WIDTH: u32 = 1280;
    const DEFAULT_PREVIEW_HEIGHT: u32 = 720;
    let _ = shared_webgpu_context();
    let target = parse_optional_handle(handle)?.unwrap_or_else(runtime_current_figure_handle);
    let normalized_width = width.unwrap_or(0).max(1);
    let normalized_height = height.unwrap_or(0).max(1);
    let (render_width, render_height) = if normalized_width == 1 && normalized_height == 1 {
        (DEFAULT_PREVIEW_WIDTH, DEFAULT_PREVIEW_HEIGHT)
    } else {
        (normalized_width, normalized_height)
    };
    let parsed: PlotSurfaceCameraState =
        serde_wasm_bindgen::from_value(camera_state).map_err(|err| js_error(&err.to_string()))?;
    let bytes = runtime_render_figure_snapshot_with_camera_state(
        target,
        render_width,
        render_height,
        parsed,
        textmark,
    )
    .await
    .map_err(runtime_flow_to_js)?;
    Ok(Uint8Array::from(bytes.as_slice()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = plotRendererReady)]
pub fn plot_renderer_ready() -> bool {
    runtime_plot_renderer_ready()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = onFigureEvent)]
pub fn on_figure_event(callback: JsValue) -> Result<(), JsValue> {
    ensure_figure_event_bridge();
    if callback.is_null() || callback.is_undefined() {
        replace_figure_event_callback(None);
        return Ok(());
    }
    let func = callback
        .dyn_ref::<js_sys::Function>()
        .ok_or_else(|| js_error("onFigureEvent expects a Function or null"))?
        .clone();
    replace_figure_event_callback(Some(func));
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn parse_hold_mode(value: JsValue) -> Result<HoldMode, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(HoldMode::Toggle);
    }
    if let Some(flag) = value.as_bool() {
        return Ok(if flag { HoldMode::On } else { HoldMode::Off });
    }
    if let Some(text) = value.as_string() {
        let normalized = text.trim().to_ascii_lowercase();
        return match normalized.as_str() {
            "on" | "holdon" => Ok(HoldMode::On),
            "off" | "holdoff" => Ok(HoldMode::Off),
            "toggle" | "switch" => Ok(HoldMode::Toggle),
            other => Err(js_error(&format!(
                "Unsupported hold mode '{other}'. Expected 'on', 'off', or 'toggle'."
            ))),
        };
    }
    Err(js_error(
        "setHoldMode expects a string ('on'|'off'|'toggle') or a boolean",
    ))
}

#[cfg(target_arch = "wasm32")]
fn parse_optional_handle(value: JsValue) -> Result<Option<FigureHandle>, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    if let Some(num) = value.as_f64() {
        if !num.is_finite() || num <= 0.0 {
            return Err(js_error("Figure handles must be positive numbers"));
        }
        return Ok(Some(FigureHandle::from(num.round() as u32)));
    }
    Err(js_error(
        "Figure handles must be numeric or left undefined for the active figure",
    ))
}

#[cfg(target_arch = "wasm32")]
fn figure_error_to_js(err: FigureError) -> JsValue {
    let payload = js_sys::Object::new();
    let message = err.to_string();
    let code = match err {
        FigureError::InvalidHandle(handle) => {
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("handle"),
                &JsValue::from(handle),
            );
            "InvalidHandle"
        }
        FigureError::InvalidSubplotGrid { rows, cols } => {
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("rows"),
                &JsValue::from(rows as u32),
            );
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("cols"),
                &JsValue::from(cols as u32),
            );
            "InvalidSubplotGrid"
        }
        FigureError::InvalidSubplotIndex { rows, cols, index } => {
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("rows"),
                &JsValue::from(rows as u32),
            );
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("cols"),
                &JsValue::from(cols as u32),
            );
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("index"),
                &JsValue::from(index as u32),
            );
            "InvalidSubplotIndex"
        }
        FigureError::InvalidAxesHandle => "InvalidAxesHandle",
        FigureError::InvalidPlotObjectHandle => "InvalidPlotObjectHandle",
        FigureError::RenderFailure { source } => {
            let details = source.to_string();
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("details"),
                &JsValue::from_str(details.as_str()),
            );
            "RenderFailure"
        }
    };
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("code"),
        &JsValue::from_str(code),
    );
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("message"),
        &JsValue::from_str(&message),
    );
    JsValue::from(payload)
}

#[cfg(target_arch = "wasm32")]
fn runtime_flow_to_js(err: RuntimeError) -> JsValue {
    serde_wasm_bindgen::to_value(&runtime_error_payload(&err, None))
        .unwrap_or_else(|_| js_error(err.message()))
}

#[cfg(target_arch = "wasm32")]
fn axes_state_to_js(state: FigureAxesState) -> JsValue {
    let payload = js_sys::Object::new();
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("handle"),
        &JsValue::from(state.handle.as_u32()),
    );
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("axesRows"),
        &JsValue::from(state.rows as u32),
    );
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("axesCols"),
        &JsValue::from(state.cols as u32),
    );
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("activeIndex"),
        &JsValue::from(state.active_index as u32),
    );
    payload.into()
}

#[cfg(target_arch = "wasm32")]
fn emit_js_figure_event(event: FigureEventView<'_>) {
    if let FigureEventKind::Closed = event.kind {
        let handle = event.handle.as_u32();
        let _ = runtime_clear_closed_figure_surfaces(handle);
        let surface_id = LEGACY_FIGURE_SURFACES.with(|slot| slot.borrow_mut().remove(&handle));
        if let Some(id) = surface_id {
            runtime_detach_surface(id);
        }
    }
    if let Some(cb) = figure_event_callback() {
        let payload = convert_event_view(event);
        let js_value = serde_wasm_bindgen::to_value(&payload).unwrap_or(JsValue::UNDEFINED);
        let _ = cb.call1(&JsValue::NULL, &js_value);
    }
}

#[cfg(target_arch = "wasm32")]
fn convert_event_view(view: FigureEventView<'_>) -> PlotFigureEvent {
    PlotFigureEvent {
        handle: view.handle.as_u32(),
        kind: match view.kind {
            FigureEventKind::Created => PlotFigureEventKind::Created,
            FigureEventKind::Updated => PlotFigureEventKind::Updated,
            FigureEventKind::Cleared => PlotFigureEventKind::Cleared,
            FigureEventKind::Closed => PlotFigureEventKind::Closed,
        },
        figure: view.figure.map(FigureSnapshot::capture),
    }
}

#[cfg(target_arch = "wasm32")]
async fn install_surface_renderer(surface_id: u32, canvas: WebCanvas) -> Result<(), JsValue> {
    init_logging_once();
    let options = WebRendererOptions {
        enable_overlay: true,
        ..WebRendererOptions::default()
    };
    let canvas_kind = match &canvas {
        WebCanvas::Html(_) => "html",
        WebCanvas::Offscreen(_) => "offscreen",
    };
    log::debug!(
        "plot-web: install_surface_renderer(surface_id={surface_id}, canvas_kind={})",
        canvas_kind
    );
    let renderer = match shared_webgpu_context() {
        Some(shared) => {
            log::debug!(
                "plot-web: install_surface_renderer using shared context surface_id={} canvas_kind={}",
                surface_id,
                canvas_kind
            );
            WebRenderer::with_shared_context(canvas.clone(), options.clone(), shared).await
        }
        None => {
            log::debug!(
                "plot-web: install_surface_renderer using dedicated context surface_id={} canvas_kind={}",
                surface_id,
                canvas_kind
            );
            WebRenderer::new(canvas, options).await
        }
    }
    .map_err(|err| js_error(&format!("Failed to initialize plot renderer: {err}")))?;
    log::debug!("plot-web: install_surface_renderer renderer_ready surface_id={surface_id}");
    runtime_install_surface(surface_id, renderer)
        .map_err(|err| js_error(&format!("Failed to register plot surface: {err}")))?;
    log::debug!("plot-web: install_surface_renderer registered surface_id={surface_id}");
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn parse_web_canvas(canvas: JsValue) -> Result<WebCanvas, JsValue> {
    if canvas.is_null() || canvas.is_undefined() {
        return Err(js_error("Canvas is required"));
    }
    if let Ok(html) = canvas.clone().dyn_into::<web_sys::HtmlCanvasElement>() {
        return Ok(WebCanvas::Html(html));
    }
    if let Ok(offscreen) = canvas.clone().dyn_into::<web_sys::OffscreenCanvas>() {
        return Ok(WebCanvas::Offscreen(offscreen));
    }
    Err(js_error("Expected an HTMLCanvasElement or OffscreenCanvas"))
}

#[cfg(target_arch = "wasm32")]
fn shared_webgpu_context() -> Option<SharedWgpuContext> {
    if let Some(ctx) = runmat_plot::shared_wgpu_context() {
        log::debug!("plot-web: shared_webgpu_context: using existing runmat_plot shared context");
        return Some(ctx);
    }

    let api_provider_present = runmat_accelerate_api::provider().is_some();
    log::debug!(
        "plot-web: shared_webgpu_context: no plot context installed yet (api_provider_present={})",
        api_provider_present
    );

    let handle = match runmat_accelerate_api::export_context(AccelContextKind::Plotting) {
        Some(handle) => handle,
        None => {
            log::debug!(
                "plot-web: shared_webgpu_context: export_context(Plotting) returned None (api_provider_present={})",
                api_provider_present
            );
            return None;
        }
    };
    match handle {
        AccelContextHandle::Wgpu(ctx) => {
            let shared = SharedWgpuContext {
                instance: ctx.instance.clone(),
                device: ctx.device,
                queue: ctx.queue,
                adapter: ctx.adapter,
                adapter_info: ctx.adapter_info.clone(),
                limits: ctx.limits,
                features: ctx.features,
            };
            log::debug!("plot-web: shared_webgpu_context: installed shared context from accelerate provider (adapter={:?})", shared.adapter_info);
            runmat_plot::install_shared_wgpu_context(shared.clone());
            Some(shared)
        }
    }
}
