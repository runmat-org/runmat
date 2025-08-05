//! Modern styling and theming system for RustMat plots
//!
//! Provides professional themes, typography, and layout systems that integrate
//! seamlessly with RustMat's configuration system.

pub mod theme;
pub mod config;

pub use theme::{ModernDarkTheme, Typography, Layout};
pub use config::{PlotThemeConfig, ThemeVariant, validate_theme_config};