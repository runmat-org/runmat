//! Theme configuration system that integrates with RustMat's config
//!
//! This module provides the configuration structures that RustMat can import
//! and use in its main configuration system, while keeping the plotting
//! library in control of its own theming.

use super::theme::{Layout, ModernDarkTheme, Typography};
use glam::Vec4;
use serde::{Deserialize, Serialize};

/// Theme variants available in the plotting system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThemeVariant {
    /// Modern dark theme with green accents (default)
    ModernDark,
    /// Classic light theme (MATLAB-style)
    ClassicLight,
    /// High contrast theme for accessibility
    HighContrast,
    /// Custom theme (loads from user config)
    Custom,
}

impl Default for ThemeVariant {
    fn default() -> Self {
        Self::ModernDark
    }
}

/// Complete plotting theme configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlotThemeConfig {
    /// Theme variant to use
    pub variant: ThemeVariant,

    /// Typography settings
    pub typography: TypographyConfig,

    /// Layout and spacing settings
    pub layout: LayoutConfig,

    /// Custom color overrides (when variant is Custom)
    pub custom_colors: Option<CustomColorConfig>,

    /// Grid settings
    pub grid: GridConfig,

    /// Animation and interaction settings
    pub interaction: InteractionConfig,
}

/// Typography configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypographyConfig {
    /// Font sizes
    pub title_font_size: f32,
    pub subtitle_font_size: f32,
    pub axis_label_font_size: f32,
    pub tick_label_font_size: f32,
    pub legend_font_size: f32,

    /// Font families
    pub title_font_family: String,
    pub body_font_family: String,
    pub monospace_font_family: String,

    /// Typography features
    pub enable_antialiasing: bool,
    pub enable_subpixel_rendering: bool,
}

impl Default for TypographyConfig {
    fn default() -> Self {
        let typography = Typography::default();
        Self {
            title_font_size: typography.title_font_size,
            subtitle_font_size: typography.subtitle_font_size,
            axis_label_font_size: typography.axis_label_font_size,
            tick_label_font_size: typography.tick_label_font_size,
            legend_font_size: typography.legend_font_size,
            title_font_family: typography.title_font_family,
            body_font_family: typography.body_font_family,
            monospace_font_family: typography.monospace_font_family,
            enable_antialiasing: true,
            enable_subpixel_rendering: true,
        }
    }
}

/// Layout and spacing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Margins and padding
    pub plot_padding: f32,
    pub title_margin: f32,
    pub axis_margin: f32,
    pub legend_margin: f32,

    /// Line widths
    pub grid_line_width: f32,
    pub axis_line_width: f32,
    pub data_line_width: f32,

    /// Point and marker sizes
    pub point_size: f32,
    pub marker_size: f32,

    /// Layout features
    pub auto_adjust_margins: bool,
    pub maintain_aspect_ratio: bool,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        let layout = Layout::default();
        Self {
            plot_padding: layout.plot_padding,
            title_margin: layout.title_margin,
            axis_margin: layout.axis_margin,
            legend_margin: layout.legend_margin,
            grid_line_width: layout.grid_line_width,
            axis_line_width: layout.axis_line_width,
            data_line_width: layout.data_line_width,
            point_size: layout.point_size,
            marker_size: 6.0,
            auto_adjust_margins: true,
            maintain_aspect_ratio: false,
        }
    }
}

/// Custom color configuration (used when variant is Custom)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomColorConfig {
    /// Background colors (as hex strings for easy configuration)
    pub background_primary: String,
    pub background_secondary: String,
    pub plot_background: String,

    /// Text colors
    pub text_primary: String,
    pub text_secondary: String,

    /// Accent colors
    pub accent_primary: String,
    pub accent_secondary: String,

    /// Grid and axis colors
    pub grid_major: String,
    pub grid_minor: String,
    pub axis_color: String,

    /// Data series colors
    pub data_colors: Vec<String>,
}

impl Default for CustomColorConfig {
    fn default() -> Self {
        Self {
            background_primary: "#141619".to_string(),
            background_secondary: "#1f2329".to_string(),
            plot_background: "#1a1d21".to_string(),
            text_primary: "#f2f4f7".to_string(),
            text_secondary: "#bfc7d1".to_string(),
            accent_primary: "#59c878".to_string(),
            accent_secondary: "#47a661".to_string(),
            grid_major: "#404449".to_string(),
            grid_minor: "#33373c".to_string(),
            axis_color: "#a6adb7".to_string(),
            data_colors: vec![
                "#59c878".to_string(), // Green
                "#40a5d6".to_string(), // Blue
                "#f28c40".to_string(), // Orange
                "#bf59d6".to_string(), // Purple
                "#f2c040".to_string(), // Yellow
                "#d95973".to_string(), // Pink
                "#40d6bf".to_string(), // Turquoise
                "#a6bf59".to_string(), // Lime
            ],
        }
    }
}

/// Grid display configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Grid visibility
    pub show_major_grid: bool,
    pub show_minor_grid: bool,

    /// Grid styling
    pub major_grid_alpha: f32,
    pub minor_grid_alpha: f32,

    /// Grid spacing
    pub auto_grid_spacing: bool,
    pub major_grid_divisions: u32,
    pub minor_grid_subdivisions: u32,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            show_major_grid: true,
            show_minor_grid: true,
            major_grid_alpha: 0.6,
            minor_grid_alpha: 0.3,
            auto_grid_spacing: true,
            major_grid_divisions: 5,
            minor_grid_subdivisions: 5,
        }
    }
}

/// Interaction and animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionConfig {
    /// Animation settings
    pub enable_animations: bool,
    pub animation_duration_ms: u32,
    pub animation_easing: String,

    /// Interaction settings
    pub enable_zoom: bool,
    pub enable_pan: bool,
    pub enable_selection: bool,
    pub enable_tooltips: bool,

    /// Performance settings
    pub max_fps: u32,
    pub enable_vsync: bool,
    pub enable_gpu_acceleration: bool,
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            enable_animations: true,
            animation_duration_ms: 300,
            animation_easing: "ease_out".to_string(),
            enable_zoom: true,
            enable_pan: true,
            enable_selection: true,
            enable_tooltips: true,
            max_fps: 60,
            enable_vsync: true,
            enable_gpu_acceleration: true,
        }
    }
}

impl PlotThemeConfig {
    /// Create a theme instance from this configuration
    pub fn build_theme(&self) -> Box<dyn PlotTheme> {
        match self.variant {
            ThemeVariant::ModernDark => Box::new(ModernDarkTheme::default()),
            ThemeVariant::ClassicLight => Box::new(ClassicLightTheme::default()),
            ThemeVariant::HighContrast => Box::new(HighContrastTheme::default()),
            ThemeVariant::Custom => {
                if let Some(custom) = &self.custom_colors {
                    Box::new(CustomTheme::from_config(custom))
                } else {
                    Box::new(ModernDarkTheme::default())
                }
            }
        }
    }

    /// Validate this configuration
    pub fn validate(&self) -> Result<(), String> {
        validate_theme_config(self)
    }

    /// Get the active typography settings
    pub fn get_typography(&self) -> Typography {
        Typography {
            title_font_size: self.typography.title_font_size,
            subtitle_font_size: self.typography.subtitle_font_size,
            axis_label_font_size: self.typography.axis_label_font_size,
            tick_label_font_size: self.typography.tick_label_font_size,
            legend_font_size: self.typography.legend_font_size,
            title_font_family: self.typography.title_font_family.clone(),
            body_font_family: self.typography.body_font_family.clone(),
            monospace_font_family: self.typography.monospace_font_family.clone(),
        }
    }

    /// Get the active layout settings
    pub fn get_layout(&self) -> Layout {
        Layout {
            plot_padding: self.layout.plot_padding,
            title_margin: self.layout.title_margin,
            axis_margin: self.layout.axis_margin,
            legend_margin: self.layout.legend_margin,
            grid_line_width: self.layout.grid_line_width,
            axis_line_width: self.layout.axis_line_width,
            data_line_width: self.layout.data_line_width,
            point_size: self.layout.point_size,
        }
    }
}

/// Trait for theme implementations
pub trait PlotTheme {
    fn get_background_color(&self) -> Vec4;
    fn get_text_color(&self) -> Vec4;
    fn get_accent_color(&self) -> Vec4;
    fn get_grid_color(&self) -> Vec4;
    fn get_axis_color(&self) -> Vec4;
    fn get_data_color(&self, index: usize) -> Vec4;
    fn apply_to_egui(&self, ctx: &egui::Context);
}

impl PlotTheme for ModernDarkTheme {
    fn get_background_color(&self) -> Vec4 {
        self.background_primary
    }
    fn get_text_color(&self) -> Vec4 {
        self.text_primary
    }
    fn get_accent_color(&self) -> Vec4 {
        self.accent_primary
    }
    fn get_grid_color(&self) -> Vec4 {
        self.grid_major
    }
    fn get_axis_color(&self) -> Vec4 {
        self.axis_color
    }
    fn get_data_color(&self, index: usize) -> Vec4 {
        self.get_data_color(index)
    }
    fn apply_to_egui(&self, ctx: &egui::Context) {
        self.apply_to_egui(ctx)
    }
}

/// Classic light theme (MATLAB-style)
#[derive(Debug, Clone)]
pub struct ClassicLightTheme {
    pub background_color: Vec4,
    pub text_color: Vec4,
    pub accent_color: Vec4,
    pub grid_color: Vec4,
    pub axis_color: Vec4,
    pub data_colors: Vec<Vec4>,
}

impl Default for ClassicLightTheme {
    fn default() -> Self {
        Self {
            background_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            text_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            accent_color: Vec4::new(0.0, 0.5, 1.0, 1.0),
            grid_color: Vec4::new(0.8, 0.8, 0.8, 0.8),
            axis_color: Vec4::new(0.3, 0.3, 0.3, 1.0),
            data_colors: vec![
                Vec4::new(0.0, 0.5, 1.0, 1.0), // Blue
                Vec4::new(1.0, 0.5, 0.0, 1.0), // Orange
                Vec4::new(0.5, 0.8, 0.2, 1.0), // Green
                Vec4::new(0.8, 0.2, 0.8, 1.0), // Magenta
                Vec4::new(1.0, 0.8, 0.0, 1.0), // Yellow
                Vec4::new(0.2, 0.8, 0.8, 1.0), // Cyan
                Vec4::new(0.8, 0.2, 0.2, 1.0), // Red
            ],
        }
    }
}

impl PlotTheme for ClassicLightTheme {
    fn get_background_color(&self) -> Vec4 {
        self.background_color
    }
    fn get_text_color(&self) -> Vec4 {
        self.text_color
    }
    fn get_accent_color(&self) -> Vec4 {
        self.accent_color
    }
    fn get_grid_color(&self) -> Vec4 {
        self.grid_color
    }
    fn get_axis_color(&self) -> Vec4 {
        self.axis_color
    }
    fn get_data_color(&self, index: usize) -> Vec4 {
        self.data_colors[index % self.data_colors.len()]
    }
    fn apply_to_egui(&self, ctx: &egui::Context) {
        ctx.set_visuals(egui::Visuals::light());
    }
}

/// High contrast theme for accessibility
#[derive(Debug, Clone)]
pub struct HighContrastTheme {
    pub background_color: Vec4,
    pub text_color: Vec4,
    pub accent_color: Vec4,
    pub grid_color: Vec4,
    pub axis_color: Vec4,
    pub data_colors: Vec<Vec4>,
}

impl Default for HighContrastTheme {
    fn default() -> Self {
        Self {
            background_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            text_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            accent_color: Vec4::new(1.0, 1.0, 0.0, 1.0),
            grid_color: Vec4::new(0.5, 0.5, 0.5, 1.0),
            axis_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            data_colors: vec![
                Vec4::new(1.0, 1.0, 0.0, 1.0), // Yellow
                Vec4::new(0.0, 1.0, 1.0, 1.0), // Cyan
                Vec4::new(1.0, 0.0, 1.0, 1.0), // Magenta
                Vec4::new(1.0, 1.0, 1.0, 1.0), // White
                Vec4::new(1.0, 0.5, 0.0, 1.0), // Orange
                Vec4::new(0.5, 1.0, 0.5, 1.0), // Light green
            ],
        }
    }
}

impl PlotTheme for HighContrastTheme {
    fn get_background_color(&self) -> Vec4 {
        self.background_color
    }
    fn get_text_color(&self) -> Vec4 {
        self.text_color
    }
    fn get_accent_color(&self) -> Vec4 {
        self.accent_color
    }
    fn get_grid_color(&self) -> Vec4 {
        self.grid_color
    }
    fn get_axis_color(&self) -> Vec4 {
        self.axis_color
    }
    fn get_data_color(&self, index: usize) -> Vec4 {
        self.data_colors[index % self.data_colors.len()]
    }
    fn apply_to_egui(&self, ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.extreme_bg_color = egui::Color32::BLACK;
        visuals.widgets.noninteractive.bg_fill = egui::Color32::BLACK;
        visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::WHITE;
        ctx.set_visuals(visuals);
    }
}

/// Custom theme from user configuration
#[derive(Debug, Clone)]
pub struct CustomTheme {
    pub background_color: Vec4,
    pub text_color: Vec4,
    pub accent_color: Vec4,
    pub grid_color: Vec4,
    pub axis_color: Vec4,
    pub data_colors: Vec<Vec4>,
}

impl CustomTheme {
    /// Create a custom theme from configuration
    pub fn from_config(config: &CustomColorConfig) -> Self {
        Self {
            background_color: hex_to_vec4(&config.background_primary)
                .unwrap_or(Vec4::new(0.1, 0.1, 0.1, 1.0)),
            text_color: hex_to_vec4(&config.text_primary).unwrap_or(Vec4::new(1.0, 1.0, 1.0, 1.0)),
            accent_color: hex_to_vec4(&config.accent_primary)
                .unwrap_or(Vec4::new(0.0, 0.8, 0.4, 1.0)),
            grid_color: hex_to_vec4(&config.grid_major).unwrap_or(Vec4::new(0.3, 0.3, 0.3, 0.6)),
            axis_color: hex_to_vec4(&config.axis_color).unwrap_or(Vec4::new(0.7, 0.7, 0.7, 1.0)),
            data_colors: config
                .data_colors
                .iter()
                .filter_map(|hex| hex_to_vec4(hex))
                .collect(),
        }
    }
}

impl PlotTheme for CustomTheme {
    fn get_background_color(&self) -> Vec4 {
        self.background_color
    }
    fn get_text_color(&self) -> Vec4 {
        self.text_color
    }
    fn get_accent_color(&self) -> Vec4 {
        self.accent_color
    }
    fn get_grid_color(&self) -> Vec4 {
        self.grid_color
    }
    fn get_axis_color(&self) -> Vec4 {
        self.axis_color
    }
    fn get_data_color(&self, index: usize) -> Vec4 {
        if self.data_colors.is_empty() {
            Vec4::new(0.5, 0.5, 0.5, 1.0) // Default gray
        } else {
            self.data_colors[index % self.data_colors.len()]
        }
    }
    fn apply_to_egui(&self, ctx: &egui::Context) {
        let mut visuals =
            if self.background_color.x + self.background_color.y + self.background_color.z < 1.5 {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            };

        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(
            (self.background_color.x * 255.0) as u8,
            (self.background_color.y * 255.0) as u8,
            (self.background_color.z * 255.0) as u8,
            255,
        );

        ctx.set_visuals(visuals);
    }
}

/// Convert hex color string to Vec4
fn hex_to_vec4(hex: &str) -> Option<Vec4> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }

    let r = u8::from_str_radix(&hex[0..2], 16).ok()? as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()? as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()? as f32 / 255.0;

    Some(Vec4::new(r, g, b, 1.0))
}

/// Validate theme configuration
pub fn validate_theme_config(config: &PlotThemeConfig) -> Result<(), String> {
    // Validate font sizes
    if config.typography.title_font_size <= 0.0 {
        return Err("Title font size must be positive".to_string());
    }
    if config.typography.axis_label_font_size <= 0.0 {
        return Err("Axis label font size must be positive".to_string());
    }

    // Validate layout values
    if config.layout.plot_padding < 0.0 {
        return Err("Plot padding must be non-negative".to_string());
    }
    if config.layout.data_line_width <= 0.0 {
        return Err("Data line width must be positive".to_string());
    }

    // Validate custom colors if present
    if config.variant == ThemeVariant::Custom {
        if let Some(custom) = &config.custom_colors {
            for color in &custom.data_colors {
                if hex_to_vec4(color).is_none() {
                    return Err(format!("Invalid hex color: {color}"));
                }
            }
        } else {
            return Err("Custom theme variant requires custom_colors configuration".to_string());
        }
    }

    // Validate animation settings
    if config.interaction.animation_duration_ms > 5000 {
        return Err("Animation duration too long (max 5000ms)".to_string());
    }

    // Validate performance settings
    if config.interaction.max_fps == 0 || config.interaction.max_fps > 240 {
        return Err("Max FPS must be between 1 and 240".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = PlotThemeConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_hex_to_vec4_conversion() {
        let color = hex_to_vec4("#ff0000").unwrap();
        assert!((color.x - 1.0).abs() < 0.01);
        assert!(color.y.abs() < 0.01);
        assert!(color.z.abs() < 0.01);
        assert!((color.w - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_invalid_hex_colors() {
        assert!(hex_to_vec4("invalid").is_none());
        assert!(hex_to_vec4("#gg0000").is_none());
        assert!(hex_to_vec4("#ff00").is_none());
    }

    #[test]
    fn test_theme_variants() {
        let config = PlotThemeConfig::default();
        let theme = config.build_theme();

        // Should create a valid theme
        let bg_color = theme.get_background_color();
        assert!(bg_color.w > 0.0); // Alpha should be positive
    }

    #[test]
    fn test_custom_theme_validation() {
        let mut config = PlotThemeConfig {
            variant: ThemeVariant::Custom,
            ..Default::default()
        };

        // Should fail without custom colors
        assert!(config.validate().is_err());

        // Should pass with valid custom colors
        config.custom_colors = Some(CustomColorConfig::default());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_bounds() {
        let mut config = PlotThemeConfig::default();

        // Test negative font size
        config.typography.title_font_size = -1.0;
        assert!(config.validate().is_err());

        // Test excessive animation duration
        config.typography.title_font_size = 18.0; // Reset
        config.interaction.animation_duration_ms = 10000;
        assert!(config.validate().is_err());

        // Test invalid FPS
        config.interaction.animation_duration_ms = 300; // Reset
        config.interaction.max_fps = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_typography_defaults() {
        let typography = TypographyConfig::default();
        assert!(typography.title_font_size > typography.subtitle_font_size);
        assert!(typography.subtitle_font_size > typography.axis_label_font_size);
        assert!(typography.enable_antialiasing);
    }

    #[test]
    fn test_data_color_cycling() {
        let theme = ModernDarkTheme::default();
        let color1 = theme.get_data_color(0);
        let color2 = theme.get_data_color(theme.data_colors.len());

        // Should cycle back to first color
        assert_eq!(color1, color2);
    }
}
