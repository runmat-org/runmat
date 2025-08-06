//! GUI overlay system for interactive plot controls and annotations
//!
//! This module handles the egui-based UI that sits on top of the WGPU-rendered
//! plot content, providing controls, axis labels, grid lines, and titles.

use crate::core::{plot_utils, PlotRenderer};
use crate::styling::{ModernDarkTheme, PlotThemeConfig};
use egui::{Align2, Color32, Context, FontId, Pos2, Rect, Stroke};

/// GUI overlay manager for plot annotations and controls
pub struct PlotOverlay {
    /// Current theme
    #[allow(dead_code)] // TODO: Use for theme customization
    theme: PlotThemeConfig,

    /// Cached plot area from last frame
    plot_area: Option<Rect>,

    /// Show debug information
    show_debug: bool,

    /// Show Dystr information modal
    show_dystr_modal: bool,
}

/// Configuration for the plot overlay
#[derive(Debug, Clone)]
pub struct OverlayConfig {
    /// Whether to show the sidebar with controls
    pub show_sidebar: bool,

    /// Whether to show grid lines
    pub show_grid: bool,

    /// Whether to show axis labels
    pub show_axes: bool,

    /// Whether to show plot title
    pub show_title: bool,

    /// Custom plot title (if any)
    pub title: Option<String>,

    /// Custom axis labels
    pub x_label: Option<String>,
    pub y_label: Option<String>,

    /// Sidebar width
    pub sidebar_width: f32,

    /// Margins around plot area
    pub plot_margins: PlotMargins,
}

#[derive(Debug, Clone)]
pub struct PlotMargins {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            show_sidebar: true,
            show_grid: true,
            show_axes: true,
            show_title: true,
            title: Some("Plot".to_string()),
            x_label: Some("X".to_string()),
            y_label: Some("Y".to_string()),
            sidebar_width: 280.0,
            plot_margins: PlotMargins {
                left: 60.0,
                right: 20.0,
                top: 40.0,
                bottom: 60.0,
            },
        }
    }
}

/// Information about the current frame's UI state
#[derive(Debug)]
pub struct FrameInfo {
    /// The plot area where WGPU should render
    pub plot_area: Option<Rect>,

    /// Whether the UI consumed any input events
    pub consumed_input: bool,

    /// Performance metrics to display
    pub metrics: OverlayMetrics,
}

#[derive(Debug, Default)]
pub struct OverlayMetrics {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub render_time_ms: f64,
    pub fps: f32,
}

impl PlotOverlay {
    /// Create a new plot overlay
    pub fn new() -> Self {
        Self {
            theme: PlotThemeConfig::default(),
            plot_area: None,
            show_debug: false,
            show_dystr_modal: false,
        }
    }

    /// Apply theme to egui context
    pub fn apply_theme(&self, ctx: &Context) {
        // Apply the modern dark theme (using stored config for future customization)
        let theme = ModernDarkTheme::default();
        theme.apply_to_egui(ctx);

        // Make context transparent so WGPU content shows through
        let mut visuals = ctx.style().visuals.clone();
        visuals.window_fill = Color32::TRANSPARENT;
        visuals.panel_fill = Color32::TRANSPARENT;
        ctx.set_visuals(visuals);
    }

    /// Render the complete overlay UI
    pub fn render(
        &mut self,
        ctx: &Context,
        plot_renderer: &PlotRenderer,
        config: &OverlayConfig,
        metrics: OverlayMetrics,
    ) -> FrameInfo {
        let mut consumed_input = false;
        let mut plot_area = None;

        // Render sidebar if enabled
        if config.show_sidebar {
            consumed_input |= self.render_sidebar(ctx, plot_renderer, config, &metrics);
        }

        // Render main plot area
        let central_response = egui::CentralPanel::default()
            .frame(egui::Frame::none()) // Transparent frame
            .show(ctx, |ui| {
                plot_area = Some(self.render_plot_area(ui, plot_renderer, config));
            });

        consumed_input |= central_response.response.hovered();

        // Render Dystr modal if needed
        if self.show_dystr_modal {
            consumed_input |= self.render_dystr_modal(ctx);
        }

        // Store plot area for next frame
        self.plot_area = plot_area;

        FrameInfo {
            plot_area,
            consumed_input,
            metrics,
        }
    }

    /// Render the sidebar with controls and information
    fn render_sidebar(
        &mut self,
        ctx: &Context,
        plot_renderer: &PlotRenderer,
        config: &OverlayConfig,
        metrics: &OverlayMetrics,
    ) -> bool {
        let mut consumed_input = false;

        let sidebar_response = egui::SidePanel::left("plot_controls")
            .resizable(true)
            .default_width(config.sidebar_width)
            .min_width(200.0)
            .show(ctx, |ui| {
                ui.style_mut().visuals.widgets.noninteractive.bg_fill = Color32::from_gray(25);
                ui.style_mut().visuals.widgets.inactive.bg_fill = Color32::from_gray(35);
                ui.style_mut().visuals.widgets.hovered.bg_fill = Color32::from_gray(45);

                // Header with Dystr branding
                ui.horizontal(|ui| {
                    // Placeholder for Dystr logo (32x32 square)
                    let logo_size = egui::Vec2::splat(32.0);
                    let logo_rect = ui.allocate_exact_size(logo_size, egui::Sense::click()).0;
                    
                    // Draw placeholder logo background
                    ui.painter().rect_filled(
                        logo_rect,
                        4.0, // rounded corners
                        Color32::from_rgb(100, 100, 100)
                    );
                    
                    // Draw "D" placeholder text in the logo area
                    ui.painter().text(
                        logo_rect.center(),
                        Align2::CENTER_CENTER,
                        "D",
                        FontId::proportional(20.0),
                        Color32::WHITE
                    );
                    
                    ui.vertical(|ui| {
                        ui.heading("RustMat");
                        ui.horizontal(|ui| {
                            ui.small("a community project by ");
                            if ui.small_button("dystr.com").clicked() {
                                self.show_dystr_modal = true;
                            }
                        });
                    });
                });
                ui.separator();
                ui.label("GC Stats: [not available]");

                // Camera information
                ui.collapsing("📷 Camera", |ui| {
                    let camera = plot_renderer.camera();
                    ui.label(format!(
                        "Position: {:.2}, {:.2}, {:.2}",
                        camera.position.x, camera.position.y, camera.position.z
                    ));
                    ui.label(format!(
                        "Target: {:.2}, {:.2}, {:.2}",
                        camera.target.x, camera.target.y, camera.target.z
                    ));

                    if let Some(bounds) = plot_renderer.data_bounds() {
                        ui.label(format!("Data X: {:.2} to {:.2}", bounds.0, bounds.1));
                        ui.label(format!("Data Y: {:.2} to {:.2}", bounds.2, bounds.3));
                    }
                });

                // Scene information
                ui.collapsing("🎬 Scene", |ui| {
                    let stats = plot_renderer.scene_statistics();
                    ui.label(format!("Nodes: {}", stats.total_nodes));
                    ui.label(format!("Visible: {}", stats.visible_nodes));
                    ui.label(format!("Vertices: {}", stats.total_vertices));
                    ui.label(format!("Triangles: {}", stats.total_triangles));
                });

                // Performance metrics
                ui.collapsing("⚡ Performance", |ui| {
                    ui.label(format!("FPS: {:.1}", metrics.fps));
                    ui.label(format!("Render: {:.2}ms", metrics.render_time_ms));
                    ui.label(format!("Vertices: {}", metrics.vertex_count));
                    ui.label(format!("Triangles: {}", metrics.triangle_count));
                });

                // Theme selection
                ui.collapsing("🎨 Theme", |ui| {
                    ui.label("Modern Dark (Active)");
                    ui.checkbox(&mut self.show_debug, "Show Debug Info");
                });

                ui.separator();

                // Plot controls
                ui.collapsing("🔧 Controls", |ui| {
                    ui.label("🖱️ Left drag: Rotate");
                    ui.label("🖱️ Right drag: Pan");
                    ui.label("🖱️ Scroll: Zoom");
                    ui.label("📱 Touch: Pinch to zoom");
                });
            });

        consumed_input |= sidebar_response.response.hovered();
        consumed_input
    }

    /// Render the main plot area with grid, axes, and annotations
    fn render_plot_area(
        &mut self,
        ui: &mut egui::Ui,
        plot_renderer: &PlotRenderer,
        config: &OverlayConfig,
    ) -> Rect {
        let available_rect = ui.available_rect_before_wrap();

        // Calculate plot area with margins
        let plot_rect = Rect::from_min_size(
            available_rect.min + egui::Vec2::new(config.plot_margins.left, config.plot_margins.top),
            available_rect.size()
                - egui::Vec2::new(
                    config.plot_margins.left + config.plot_margins.right,
                    config.plot_margins.top + config.plot_margins.bottom,
                ),
        );

        // Ensure square aspect ratio for better plots
        let size = plot_rect.width().min(plot_rect.height());
        let centered_plot_rect =
            Rect::from_center_size(plot_rect.center(), egui::Vec2::splat(size));

        // Draw plot frame
        ui.painter().rect_stroke(
            centered_plot_rect,
            0.0,
            Stroke::new(1.5, Color32::from_gray(180)),
        );

        // Draw grid if enabled
        if config.show_grid {
            self.draw_grid(ui, centered_plot_rect, plot_renderer);
        }

        // Draw axes if enabled
        if config.show_axes {
            self.draw_axes(ui, centered_plot_rect, plot_renderer, config);
        }

        // Draw title if enabled
        if config.show_title {
            if let Some(title) = &config.title {
                self.draw_title(ui, centered_plot_rect, title);
            }
        }

        // Draw axis labels
        if let Some(x_label) = &config.x_label {
            self.draw_x_label(ui, centered_plot_rect, x_label);
        }
        if let Some(y_label) = &config.y_label {
            self.draw_y_label(ui, centered_plot_rect, y_label);
        }

        centered_plot_rect
    }

    /// Draw grid lines based on data bounds
    fn draw_grid(&self, ui: &mut egui::Ui, plot_rect: Rect, plot_renderer: &PlotRenderer) {
        if let Some(data_bounds) = plot_renderer.data_bounds() {
            let grid_color_major = Color32::from_gray(80);
            let _grid_color_minor = Color32::from_gray(60);

            let (x_min, x_max, y_min, y_max) = data_bounds;
            let x_range = x_max - x_min;
            let y_range = y_max - y_min;

            // Calculate tick intervals
            let x_tick_interval = plot_utils::calculate_tick_interval(x_range);
            let y_tick_interval = plot_utils::calculate_tick_interval(y_range);

            // Draw vertical grid lines
            let mut x_val = (x_min / x_tick_interval).ceil() * x_tick_interval;
            while x_val <= x_max {
                let x_screen =
                    plot_rect.min.x + ((x_val - x_min) / x_range) as f32 * plot_rect.width();
                ui.painter().line_segment(
                    [
                        Pos2::new(x_screen, plot_rect.min.y),
                        Pos2::new(x_screen, plot_rect.max.y),
                    ],
                    Stroke::new(0.8, grid_color_major),
                );
                x_val += x_tick_interval;
            }

            // Draw horizontal grid lines
            let mut y_val = (y_min / y_tick_interval).ceil() * y_tick_interval;
            while y_val <= y_max {
                let y_screen =
                    plot_rect.max.y - ((y_val - y_min) / y_range) as f32 * plot_rect.height();
                ui.painter().line_segment(
                    [
                        Pos2::new(plot_rect.min.x, y_screen),
                        Pos2::new(plot_rect.max.x, y_screen),
                    ],
                    Stroke::new(0.8, grid_color_major),
                );
                y_val += y_tick_interval;
            }
        }
    }

    /// Draw axis ticks and numeric labels
    fn draw_axes(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        _config: &OverlayConfig,
    ) {
        if let Some(data_bounds) = plot_renderer.data_bounds() {
            let (x_min, x_max, y_min, y_max) = data_bounds;
            let x_range = x_max - x_min;
            let y_range = y_max - y_min;
            let tick_length = 6.0;
            let label_offset = 15.0;

            // Calculate tick intervals
            let x_tick_interval = plot_utils::calculate_tick_interval(x_range);
            let y_tick_interval = plot_utils::calculate_tick_interval(y_range);

            // Draw X-axis ticks and labels
            let mut x_val = (x_min / x_tick_interval).ceil() * x_tick_interval;
            while x_val <= x_max {
                let x_screen =
                    plot_rect.min.x + ((x_val - x_min) / x_range) as f32 * plot_rect.width();

                // Tick mark
                ui.painter().line_segment(
                    [
                        Pos2::new(x_screen, plot_rect.max.y),
                        Pos2::new(x_screen, plot_rect.max.y + tick_length),
                    ],
                    Stroke::new(1.0, Color32::WHITE),
                );

                // Label
                ui.painter().text(
                    Pos2::new(x_screen, plot_rect.max.y + label_offset),
                    Align2::CENTER_CENTER,
                    plot_utils::format_tick_label(x_val),
                    FontId::proportional(10.0),
                    Color32::from_gray(200),
                );

                x_val += x_tick_interval;
            }

            // Draw Y-axis ticks and labels
            let mut y_val = (y_min / y_tick_interval).ceil() * y_tick_interval;
            while y_val <= y_max {
                let y_screen =
                    plot_rect.max.y - ((y_val - y_min) / y_range) as f32 * plot_rect.height();

                // Tick mark
                ui.painter().line_segment(
                    [
                        Pos2::new(plot_rect.min.x - tick_length, y_screen),
                        Pos2::new(plot_rect.min.x, y_screen),
                    ],
                    Stroke::new(1.0, Color32::WHITE),
                );

                // Label
                ui.painter().text(
                    Pos2::new(plot_rect.min.x - label_offset, y_screen),
                    Align2::CENTER_CENTER,
                    plot_utils::format_tick_label(y_val),
                    FontId::proportional(10.0),
                    Color32::from_gray(200),
                );

                y_val += y_tick_interval;
            }
        }
    }

    /// Draw plot title
    fn draw_title(&self, ui: &mut egui::Ui, plot_rect: Rect, title: &str) {
        ui.painter().text(
            Pos2::new(plot_rect.center().x, plot_rect.min.y - 20.0),
            Align2::CENTER_CENTER,
            title,
            FontId::proportional(16.0),
            Color32::WHITE,
        );
    }

    /// Draw X-axis label
    fn draw_x_label(&self, ui: &mut egui::Ui, plot_rect: Rect, label: &str) {
        ui.painter().text(
            Pos2::new(plot_rect.center().x, plot_rect.max.y + 40.0),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(14.0),
            Color32::WHITE,
        );
    }

    /// Draw Y-axis label
    fn draw_y_label(&self, ui: &mut egui::Ui, plot_rect: Rect, label: &str) {
        ui.painter().text(
            Pos2::new(plot_rect.min.x - 40.0, plot_rect.center().y),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(14.0),
            Color32::WHITE,
        );
    }

    /// Get the plot area from the last frame
    pub fn plot_area(&self) -> Option<Rect> {
        self.plot_area
    }

    /// Render the Dystr information modal
    fn render_dystr_modal(&mut self, ctx: &Context) -> bool {
        let mut consumed_input = false;
        
        egui::Window::new("About Dystr")
            .anchor(Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .collapsible(false)
            .resizable(false)
            .default_width(400.0)
            .show(ctx, |ui| {
                consumed_input = true;
                
                ui.vertical_centered(|ui| {
                    ui.add_space(10.0);
                    
                    // Dystr logo placeholder (larger for modal)
                    let logo_size = egui::Vec2::splat(64.0);
                    let logo_rect = ui.allocate_exact_size(logo_size, egui::Sense::hover()).0;
                    
                    ui.painter().rect_filled(
                        logo_rect,
                        8.0, // rounded corners
                        Color32::from_rgb(60, 130, 200) // Dystr brand color placeholder
                    );
                    
                    ui.painter().text(
                        logo_rect.center(),
                        Align2::CENTER_CENTER,
                        "D",
                        FontId::proportional(40.0),
                        Color32::WHITE
                    );
                    
                    ui.add_space(15.0);
                    
                    ui.heading("Welcome to RustMat");
                    ui.add_space(10.0);
                    
                    ui.label("RustMat is a high-performance MATLAB-compatible");
                    ui.label("numerical computing platform, built as part of");
                    ui.label("the Dystr computation ecosystem.");
                    
                    ui.add_space(15.0);
                    
                    ui.label("🚀 V8-inspired JIT compilation");
                    ui.label("⚡ BLAS/LAPACK acceleration");
                    ui.label("🎯 Full MATLAB compatibility");
                    ui.label("🔬 Advanced plotting & visualization");
                    
                    ui.add_space(20.0);
                    
                    ui.horizontal(|ui| {
                        if ui.button("Visit dystr.com").clicked() {
                            // Open dystr.com in browser
                            if let Err(e) = webbrowser::open("https://dystr.com") {
                                eprintln!("Failed to open browser: {}", e);
                            }
                        }
                        
                        if ui.button("Close").clicked() {
                            self.show_dystr_modal = false;
                        }
                    });
                    
                    ui.add_space(10.0);
                });
            });

        consumed_input
    }
}
