//! GUI components for interactive plotting
//! 
//! Provides interactive controls, widgets, and GUI layout
//! for desktop plotting applications.

#[cfg(feature = "gui")]
pub mod window;
#[cfg(feature = "gui")]
pub mod controls;
#[cfg(feature = "gui")]
pub mod widgets;

#[cfg(feature = "gui")]
pub use window::*;
#[cfg(feature = "gui")]
pub use controls::*;
#[cfg(feature = "gui")]
pub use widgets::*;