//! Modern styling and theming system for RustMat plots
//!
//! Provides professional themes, typography, and layout systems that integrate
//! seamlessly with RustMat's configuration system.

pub mod config;
pub mod theme;

pub use config::{validate_theme_config, PlotThemeConfig, ThemeVariant};
pub use theme::{Layout, ModernDarkTheme, Typography};
