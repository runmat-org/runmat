//! Event handling and user interaction for interactive plots
//! 
//! Manages mouse, keyboard, and touch input for plot navigation
//! and data interaction.

use glam::Vec2;

/// Event types for plot interaction
#[derive(Debug, Clone)]
pub enum PlotEvent {
    MousePress { position: Vec2, button: MouseButton },
    MouseRelease { position: Vec2, button: MouseButton },
    MouseMove { position: Vec2, delta: Vec2 },
    MouseWheel { delta: f32 },
    KeyPress { key: KeyCode },
    KeyRelease { key: KeyCode },
    Resize { width: u32, height: u32 },
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