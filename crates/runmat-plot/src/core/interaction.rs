//! Event handling and user interaction for interactive plots
//!
//! Manages mouse, keyboard, and touch input for plot navigation
//! and data interaction.

use glam::Vec2;

/// Keyboard modifier keys captured alongside pointer events.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

/// Event types for plot interaction
#[derive(Debug, Clone)]
pub enum PlotEvent {
    MousePress {
        position: Vec2,
        button: MouseButton,
        modifiers: Modifiers,
    },
    MouseRelease {
        position: Vec2,
        button: MouseButton,
        modifiers: Modifiers,
    },
    MouseMove {
        position: Vec2,
        delta: Vec2,
        buttons: u32,
        modifiers: Modifiers,
    },
    /// Mouse wheel / trackpad scroll.
    ///
    /// `position` is the pointer location in the same coordinate space as other mouse events.
    MouseWheel {
        position: Vec2,
        delta: Vec2,
        modifiers: Modifiers,
    },
    KeyPress {
        key: KeyCode,
    },
    KeyRelease {
        key: KeyCode,
    },
    Resize {
        width: u32,
        height: u32,
    },
}

/// Mouse button enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Key code enumeration (subset for now)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCode {
    Escape,
    Space,
    Enter,
    Tab,
    Backspace,
    Delete,
    Home,
    End,
    PageUp,
    PageDown,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    // Add more as needed
}

/// Event handler trait for plot interaction
pub trait EventHandler {
    fn handle_event(&mut self, event: PlotEvent) -> bool;
}

#[cfg(feature = "egui-overlay")]
pub fn egui_raw_input_from_plot_events(
    screen_rect: egui::Rect,
    pixels_per_point: f32,
    events: Vec<PlotEvent>,
) -> egui::RawInput {
    let ppp = pixels_per_point.max(0.5);
    let mut egui_events = Vec::with_capacity(events.len());
    for event in events {
        match event {
            PlotEvent::MousePress {
                position,
                button,
                modifiers,
            } => {
                egui_events.push(egui::Event::PointerMoved(egui_pos(position, ppp)));
                egui_events.push(egui::Event::PointerButton {
                    pos: egui_pos(position, ppp),
                    button: egui_button(button),
                    pressed: true,
                    modifiers: egui_modifiers(modifiers),
                });
            }
            PlotEvent::MouseRelease {
                position,
                button,
                modifiers,
            } => {
                egui_events.push(egui::Event::PointerMoved(egui_pos(position, ppp)));
                egui_events.push(egui::Event::PointerButton {
                    pos: egui_pos(position, ppp),
                    button: egui_button(button),
                    pressed: false,
                    modifiers: egui_modifiers(modifiers),
                });
            }
            PlotEvent::MouseMove {
                position, delta, ..
            } => {
                egui_events.push(egui::Event::PointerMoved(egui_pos(position, ppp)));
                if delta.length_squared() > f32::EPSILON {
                    egui_events.push(egui::Event::MouseMoved(egui::vec2(
                        delta.x / ppp,
                        delta.y / ppp,
                    )));
                }
            }
            PlotEvent::MouseWheel {
                position,
                delta,
                modifiers,
            } => {
                egui_events.push(egui::Event::PointerMoved(egui_pos(position, ppp)));
                egui_events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(delta.x * 80.0, delta.y * 80.0),
                    modifiers: egui_wheel_modifiers(modifiers),
                });
                if modifiers.ctrl || modifiers.meta {
                    egui_events.push(egui::Event::Zoom((1.0 + delta.y * 0.08).clamp(0.2, 5.0)));
                }
            }
            PlotEvent::Resize { .. }
            | PlotEvent::KeyPress { .. }
            | PlotEvent::KeyRelease { .. } => {}
        }
    }

    egui::RawInput {
        screen_rect: Some(screen_rect),
        events: egui_events,
        viewports: std::iter::once((
            egui::ViewportId::ROOT,
            egui::ViewportInfo {
                native_pixels_per_point: Some(ppp),
                inner_rect: Some(screen_rect),
                outer_rect: Some(screen_rect),
                focused: Some(true),
                ..Default::default()
            },
        ))
        .collect(),
        ..Default::default()
    }
}

#[cfg(all(test, feature = "egui-overlay"))]
mod egui_input_tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn modified_wheel_keeps_platform_zoom_presentation() {
        let input = egui_raw_input_from_plot_events(
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 100.0)),
            1.0,
            vec![PlotEvent::MouseWheel {
                position: Vec2::new(12.0, 18.0),
                delta: Vec2::new(0.0, 1.0),
                modifiers: Modifiers {
                    ctrl: true,
                    ..Modifiers::default()
                },
            }],
        );
        assert_eq!(input.events.len(), 3);
        assert!(matches!(input.events[0], egui::Event::PointerMoved(_)));
        assert!(matches!(
            input.events[1],
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                ..
            }
        ));
        assert!(
            matches!(input.events[2], egui::Event::Zoom(factor) if (factor - 1.08).abs() < 1e-6)
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(matches!(
                input.events[1],
                egui::Event::MouseWheel {
                    modifiers: egui::Modifiers {
                        ctrl: false,
                        mac_cmd: false,
                        command: false,
                        ..
                    },
                    ..
                }
            ));
            let mut observed = None;
            let context = egui::Context::default();
            let _ = context.run(input, |ctx| {
                observed = Some(ctx.input(|state| state.zoom_delta()))
            });
            assert!((observed.expect("zoom factor") - 1.08).abs() < 1e-6);
        }

        #[cfg(target_arch = "wasm32")]
        assert!(matches!(
            input.events[1],
            egui::Event::MouseWheel {
                modifiers: egui::Modifiers { ctrl: true, .. },
                ..
            }
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_modified_wheel_uses_original_zoom_clamps() {
        for (delta_y, expected) in [(-1.0, 0.92), (-100.0, 0.2), (100.0, 5.0)] {
            let input = egui_raw_input_from_plot_events(
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 100.0)),
                1.0,
                vec![PlotEvent::MouseWheel {
                    position: Vec2::ZERO,
                    delta: Vec2::new(0.0, delta_y),
                    modifiers: Modifiers {
                        meta: true,
                        ..Modifiers::default()
                    },
                }],
            );
            let mut observed = None;
            let context = egui::Context::default();
            let _ = context.run(input, |ctx| {
                observed = Some(ctx.input(|state| state.zoom_delta()))
            });
            assert!((observed.expect("zoom factor") - expected).abs() < 1e-6);
        }
    }
}

#[cfg(feature = "egui-overlay")]
fn egui_pos(position: Vec2, pixels_per_point: f32) -> egui::Pos2 {
    egui::pos2(position.x / pixels_per_point, position.y / pixels_per_point)
}

#[cfg(feature = "egui-overlay")]
fn egui_modifiers(value: Modifiers) -> egui::Modifiers {
    egui::Modifiers {
        alt: value.alt,
        ctrl: value.ctrl,
        shift: value.shift,
        mac_cmd: value.meta,
        command: value.meta || value.ctrl,
    }
}

#[cfg(all(feature = "egui-overlay", not(target_arch = "wasm32")))]
fn egui_wheel_modifiers(_value: Modifiers) -> egui::Modifiers {
    // Native's old Scroll event carried no modifiers. Keep its plain scrolling
    // (including Shift) and separately emit the original clamped Zoom event.
    egui::Modifiers::NONE
}

#[cfg(all(feature = "egui-overlay", target_arch = "wasm32"))]
fn egui_wheel_modifiers(value: Modifiers) -> egui::Modifiers {
    egui_modifiers(value)
}

#[cfg(feature = "egui-overlay")]
fn egui_button(button: MouseButton) -> egui::PointerButton {
    match button {
        MouseButton::Left => egui::PointerButton::Primary,
        MouseButton::Right => egui::PointerButton::Secondary,
        MouseButton::Middle => egui::PointerButton::Middle,
    }
}
