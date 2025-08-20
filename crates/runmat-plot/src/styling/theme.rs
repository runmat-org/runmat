//! Modern dark theme system with professional styling
//!
//! Provides a sleek, modern dark theme with excellent contrast,
//! readability, and visual hierarchy.

use glam::Vec4;

/// Professional color palette for the modern dark theme
#[derive(Debug, Clone)]
pub struct ModernDarkTheme {
    // Background colors
    pub background_primary: Vec4,
    pub background_secondary: Vec4,
    pub background_tertiary: Vec4,

    // Text colors
    pub text_primary: Vec4,
    pub text_secondary: Vec4,
    pub text_accent: Vec4,

    // Accent colors (inspired by the green theme in the image)
    pub accent_primary: Vec4,
    pub accent_secondary: Vec4,
    pub accent_success: Vec4,
    pub accent_warning: Vec4,
    pub accent_error: Vec4,

    // Plot colors
    pub plot_background: Vec4,
    pub grid_major: Vec4,
    pub grid_minor: Vec4,
    pub axis_color: Vec4,

    // Data colors (beautiful palette for multiple series)
    pub data_colors: Vec<Vec4>,
}

impl Default for ModernDarkTheme {
    fn default() -> Self {
        Self {
            // Rich dark backgrounds with subtle gradients
            background_primary: Vec4::new(0.08, 0.09, 0.11, 1.0), // Deep charcoal
            background_secondary: Vec4::new(0.12, 0.14, 0.16, 1.0), // Lighter charcoal
            background_tertiary: Vec4::new(0.16, 0.18, 0.21, 1.0), // Card backgrounds

            // High contrast text for excellent readability
            text_primary: Vec4::new(0.95, 0.96, 0.97, 1.0), // Almost white
            text_secondary: Vec4::new(0.75, 0.78, 0.82, 1.0), // Light grey
            text_accent: Vec4::new(0.40, 0.85, 0.55, 1.0),  // Beautiful green accent

            // Modern accent colors inspired by the provided image
            accent_primary: Vec4::new(0.35, 0.78, 0.48, 1.0), // Primary green
            accent_secondary: Vec4::new(0.28, 0.65, 0.40, 1.0), // Darker green
            accent_success: Vec4::new(0.42, 0.85, 0.55, 1.0), // Success green
            accent_warning: Vec4::new(0.95, 0.75, 0.25, 1.0), // Warm amber
            accent_error: Vec4::new(0.92, 0.35, 0.35, 1.0),   // Soft red

            // Professional plot styling
            plot_background: Vec4::new(0.10, 0.11, 0.13, 1.0), // Slightly lighter than main bg
            grid_major: Vec4::new(0.25, 0.27, 0.30, 0.6),      // Subtle grid lines
            grid_minor: Vec4::new(0.20, 0.22, 0.25, 0.3),      // Very subtle minor grid
            axis_color: Vec4::new(0.65, 0.68, 0.72, 1.0),      // Clear axis lines

            // Beautiful data color palette (carefully chosen for contrast and aesthetics)
            data_colors: vec![
                Vec4::new(0.35, 0.78, 0.48, 1.0), // Primary green
                Vec4::new(0.25, 0.65, 0.85, 1.0), // Beautiful blue
                Vec4::new(0.95, 0.55, 0.25, 1.0), // Warm orange
                Vec4::new(0.75, 0.35, 0.85, 1.0), // Royal purple
                Vec4::new(0.95, 0.75, 0.25, 1.0), // Golden yellow
                Vec4::new(0.85, 0.35, 0.55, 1.0), // Rose pink
                Vec4::new(0.25, 0.85, 0.75, 1.0), // Turquoise
                Vec4::new(0.65, 0.75, 0.35, 1.0), // Lime green
            ],
        }
    }
}

impl ModernDarkTheme {
    /// Get data color by index (cycles through palette)
    pub fn get_data_color(&self, index: usize) -> Vec4 {
        self.data_colors[index % self.data_colors.len()]
    }

    /// Create a lighter variant of a color (for highlights)
    pub fn lighten_color(color: Vec4, factor: f32) -> Vec4 {
        Vec4::new(
            (color.x + factor).min(1.0),
            (color.y + factor).min(1.0),
            (color.z + factor).min(1.0),
            color.w,
        )
    }

    /// Create a darker variant of a color (for shadows)
    pub fn darken_color(color: Vec4, factor: f32) -> Vec4 {
        Vec4::new(
            (color.x - factor).max(0.0),
            (color.y - factor).max(0.0),
            (color.z - factor).max(0.0),
            color.w,
        )
    }

    /// Apply this theme to egui context
    pub fn apply_to_egui(&self, ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();

        // Main UI colors
        visuals.window_fill = egui::Color32::from_rgba_unmultiplied(
            (self.background_secondary.x * 255.0) as u8,
            (self.background_secondary.y * 255.0) as u8,
            (self.background_secondary.z * 255.0) as u8,
            (self.background_secondary.w * 255.0) as u8,
        );

        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(
            (self.background_tertiary.x * 255.0) as u8,
            (self.background_tertiary.y * 255.0) as u8,
            (self.background_tertiary.z * 255.0) as u8,
            (self.background_tertiary.w * 255.0) as u8,
        );

        // Text colors
        visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::from_rgba_unmultiplied(
            (self.text_primary.x * 255.0) as u8,
            (self.text_primary.y * 255.0) as u8,
            (self.text_primary.z * 255.0) as u8,
            (self.text_primary.w * 255.0) as u8,
        );

        // Accent colors
        visuals.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(
            (self.accent_primary.x * 255.0) as u8,
            (self.accent_primary.y * 255.0) as u8,
            (self.accent_primary.z * 255.0) as u8,
            64, // Semi-transparent
        );

        // Apply the theme
        ctx.set_visuals(visuals);
    }
}

/// Typography system for professional text rendering
#[derive(Debug, Clone)]
pub struct Typography {
    pub title_font_size: f32,
    pub subtitle_font_size: f32,
    pub axis_label_font_size: f32,
    pub tick_label_font_size: f32,
    pub legend_font_size: f32,

    pub title_font_family: String,
    pub body_font_family: String,
    pub monospace_font_family: String,
}

impl Default for Typography {
    fn default() -> Self {
        Self {
            title_font_size: 18.0,
            subtitle_font_size: 14.0,
            axis_label_font_size: 12.0,
            tick_label_font_size: 10.0,
            legend_font_size: 11.0,

            title_font_family: "SF Pro Display".to_string(), // Modern, clean
            body_font_family: "SF Pro Text".to_string(),     // Readable
            monospace_font_family: "SF Mono".to_string(),    // For numbers
        }
    }
}

/// Professional spacing and layout constants
#[derive(Debug, Clone)]
pub struct Layout {
    pub plot_padding: f32,
    pub title_margin: f32,
    pub axis_margin: f32,
    pub legend_margin: f32,
    pub grid_line_width: f32,
    pub axis_line_width: f32,
    pub data_line_width: f32,
    pub point_size: f32,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            plot_padding: 20.0,
            title_margin: 15.0,
            axis_margin: 10.0,
            legend_margin: 8.0,
            grid_line_width: 0.5,
            axis_line_width: 1.0,
            data_line_width: 2.0,
            point_size: 4.0,
        }
    }
}
