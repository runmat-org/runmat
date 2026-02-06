//! GUI overlay system for interactive plot controls and annotations
//!
//! This module handles the egui-based UI that sits on top of the WGPU-rendered
//! plot content, providing controls, axis labels, grid lines, and titles.

use crate::core::{plot_utils, PlotRenderer};
use crate::styling::{ModernDarkTheme, PlotThemeConfig};
use egui::{Align2, Color32, Context, FontId, Pos2, Rect, Stroke};
use glam::{Vec3, Vec4};

/// GUI overlay manager for plot annotations and controls
pub struct PlotOverlay {
    /// Current theme
    #[allow(dead_code)] // TODO: Use for theme customization
    theme: PlotThemeConfig,

    /// Cached plot area from last frame
    plot_area: Option<Rect>,
    /// Cached toolbar rectangle (egui points)
    toolbar_rect: Option<Rect>,
    /// Cached sidebar rectangle (egui points)
    sidebar_rect: Option<Rect>,

    /// Show debug information
    show_debug: bool,

    /// Show Dystr information modal
    show_dystr_modal: bool,

    // Toolbar state
    want_save_png: bool,
    want_save_svg: bool,
    want_reset_view: bool,
    want_toggle_grid: Option<bool>,
    want_toggle_legend: Option<bool>,
}

/// Configuration for the plot overlay
#[derive(Debug, Clone)]
pub struct OverlayConfig {
    /// Whether to show the sidebar with controls
    pub show_sidebar: bool,

    /// Whether to show the top toolbar (legend/grid/reset/save).
    pub show_toolbar: bool,

    /// Scale factor applied to overlay font sizes (1.0 = default).
    pub font_scale: f32,

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
            show_toolbar: true,
            font_scale: 1.0,
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

impl Default for PlotOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl PlotOverlay {
    /// Create a new plot overlay
    pub fn new() -> Self {
        Self {
            theme: PlotThemeConfig::default(),
            plot_area: None,
            toolbar_rect: None,
            sidebar_rect: None,
            show_debug: false,
            show_dystr_modal: false,
            want_save_png: false,
            want_save_svg: false,
            want_reset_view: false,
            want_toggle_grid: None,
            want_toggle_legend: None,
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
                // Toolbar (top-right)
                if config.show_toolbar {
                    egui::TopBottomPanel::top("plot_toolbar")
                        .frame(egui::Frame::none())
                        .show_inside(ui, |ui| {
                            let padded = ui.max_rect().shrink2(egui::vec2(12.0, 6.0));
                            self.toolbar_rect = Some(padded);
                            ui.allocate_ui_at_rect(padded, |ui| {
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
                                        ui.spacing_mut().button_padding = egui::vec2(8.0, 6.0);
                                        if ui.button("Save PNG").clicked() {
                                            self.want_save_png = true;
                                        }
                                        if ui.button("Save SVG").clicked() {
                                            self.want_save_svg = true;
                                        }
                                        if ui.button("Reset View").clicked() {
                                            self.want_reset_view = true;
                                        }
                                        let mut grid = plot_renderer.overlay_show_grid();
                                        if ui.toggle_value(&mut grid, "Grid").changed() {
                                            self.want_toggle_grid = Some(grid);
                                        }
                                        let mut legend = plot_renderer.overlay_show_legend();
                                        if ui.toggle_value(&mut legend, "Legend").changed() {
                                            self.want_toggle_legend = Some(legend);
                                        }
                                    },
                                );
                            });
                        });
                } else {
                    self.toolbar_rect = None;
                }
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
                        Color32::from_rgb(100, 100, 100),
                    );

                    // Draw "D" placeholder text in the logo area
                    ui.painter().text(
                        logo_rect.center(),
                        Align2::CENTER_CENTER,
                        "D",
                        FontId::proportional(20.0),
                        Color32::WHITE,
                    );

                    ui.vertical(|ui| {
                        ui.heading("RunMat");
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

                    if let Some(vb) = plot_renderer.view_bounds() {
                        ui.label(format!("View X: {:.2} to {:.2}", vb.0, vb.1));
                        ui.label(format!("View Y: {:.2} to {:.2}", vb.2, vb.3));
                    }
                    if let Some(db) = plot_renderer.data_bounds() {
                        ui.label(format!("Data X: {:.2} to {:.2}", db.0, db.1));
                        ui.label(format!("Data Y: {:.2} to {:.2}", db.2, db.3));
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
                    ui.label("🖱️ Orbit: MMB drag (or RMB drag)");
                    ui.label("🖱️ Pan: Shift + MMB drag (or Shift + RMB drag)");
                    ui.label("🖱️ Zoom: Scroll wheel (zooms to cursor)");
                    ui.label("🖱️ Alt + LMB/MMB/RMB: Orbit/Pan/Zoom");
                    ui.label("📱 Touch: Pinch to zoom");
                });
            });

        consumed_input |= sidebar_response.response.hovered();
        self.sidebar_rect = Some(sidebar_response.response.rect);
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

        // Detect whether we're rendering any 3D (perspective) axes. In 3D we don't need 2D-style
        // margins for ticks/labels, and inset margins read like an "outer frame".
        let (rows, cols) = plot_renderer.figure_axes_grid();
        let has_3d_axes = (0..(rows.max(1) * cols.max(1))).any(|axes_index| {
            let cam = plot_renderer
                .axes_camera(axes_index)
                .unwrap_or_else(|| plot_renderer.camera());
            matches!(
                cam.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            )
        });

        // Calculate plot area. Use full available rect for 3D; keep margins for 2D.
        let plot_rect = if has_3d_axes {
            available_rect
        } else {
            Rect::from_min_size(
                available_rect.min
                    + egui::Vec2::new(config.plot_margins.left, config.plot_margins.top),
                available_rect.size()
                    - egui::Vec2::new(
                        config.plot_margins.left + config.plot_margins.right,
                        config.plot_margins.top + config.plot_margins.bottom,
                    ),
            )
        };

        // Use full available rectangular plot area (do not force square);
        // camera fitting and axis_equal settings will control aspect.
        let centered_plot_rect = plot_rect;

        if rows * cols > 1 {
            let rects = self.compute_subplot_rects(centered_plot_rect, rows, cols, 8.0, 8.0);
            for (i, r) in rects.iter().enumerate() {
                let cam = plot_renderer
                    .axes_camera(i)
                    .unwrap_or_else(|| plot_renderer.camera());
                if matches!(
                    cam.projection,
                    crate::core::camera::ProjectionType::Perspective { .. }
                ) {
                    self.draw_3d_orientation_gizmo(ui, *r, plot_renderer, i, config.font_scale);
                    self.draw_3d_origin_axis_ticks(ui, *r, plot_renderer, i, config.font_scale);
                    continue;
                }
                // Frame (2D only; 3D uses the axes cube instead)
                if plot_renderer.overlay_show_box() {
                    ui.painter()
                        .rect_stroke(*r, 0.0, Stroke::new(1.5, Color32::from_gray(180)));
                }

                // Grid (2D)
                if config.show_grid {
                    let b = plot_renderer.view_bounds_for_axes(i);
                    self.draw_grid(ui, *r, plot_renderer, b);
                }
                // Axes (2D)
                if config.show_axes {
                    let b = plot_renderer.view_bounds_for_axes(i);
                    self.draw_axes(ui, *r, plot_renderer, config, b);
                }
            }
        } else {
            let cam = plot_renderer.camera();
            if matches!(
                cam.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            ) {
                self.draw_3d_orientation_gizmo(
                    ui,
                    centered_plot_rect,
                    plot_renderer,
                    0,
                    config.font_scale,
                );
                self.draw_3d_origin_axis_ticks(
                    ui,
                    centered_plot_rect,
                    plot_renderer,
                    0,
                    config.font_scale,
                );
            } else {
                // Draw plot frame (2D only; 3D uses the axes cube instead)
                if plot_renderer.overlay_show_box() {
                    ui.painter().rect_stroke(
                        centered_plot_rect,
                        0.0,
                        Stroke::new(1.5, Color32::from_gray(180)),
                    );
                }
                // Draw grid if enabled
                if config.show_grid {
                    self.draw_grid(ui, centered_plot_rect, plot_renderer, None);
                }

                // Draw axes if enabled
                if config.show_axes {
                    self.draw_axes(ui, centered_plot_rect, plot_renderer, config, None);
                    // Emphasize zero baseline if within data range
                    if let Some((x_min, x_max, y_min, y_max)) = plot_renderer
                        .view_bounds()
                        .or_else(|| plot_renderer.data_bounds())
                    {
                        let zero_stroke = Stroke::new(1.5, Color32::from_gray(200));
                        if y_min < 0.0 && y_max > 0.0 {
                            let y_screen = centered_plot_rect.max.y
                                - ((0.0 - y_min) / (y_max - y_min)) as f32
                                    * centered_plot_rect.height();
                            ui.painter().line_segment(
                                [
                                    Pos2::new(centered_plot_rect.min.x, y_screen),
                                    Pos2::new(centered_plot_rect.max.x, y_screen),
                                ],
                                zero_stroke,
                            );
                        }
                        if x_min < 0.0 && x_max > 0.0 {
                            let x_screen = centered_plot_rect.min.x
                                + ((0.0 - x_min) / (x_max - x_min)) as f32
                                    * centered_plot_rect.width();
                            ui.painter().line_segment(
                                [
                                    Pos2::new(x_screen, centered_plot_rect.min.y),
                                    Pos2::new(x_screen, centered_plot_rect.max.y),
                                ],
                                zero_stroke,
                            );
                        }
                    }
                }
            }
        }

        // Draw title if enabled (single global title)
        if config.show_title {
            if let Some(title) = &config.title {
                self.draw_title(ui, centered_plot_rect, title, config.font_scale);
            }
        }

        // Draw axis labels only for 2D plots. In 3D (perspective) the axes cube already
        // communicates orientation and drawing large "X/Y" labels looks like a 2D frame.
        if !has_3d_axes {
            if let Some(x_label) = &config.x_label {
                self.draw_x_label(ui, centered_plot_rect, x_label, config.font_scale);
            }
            if let Some(y_label) = &config.y_label {
                self.draw_y_label(ui, centered_plot_rect, y_label, config.font_scale);
            }
        }

        // Draw legend if enabled and entries available
        if plot_renderer.overlay_show_legend() {
            let entries = plot_renderer.overlay_legend_entries();
            if !entries.is_empty() {
                let pad = 6.0;
                let mut y = centered_plot_rect.min.y + pad + 4.0;
                let x = centered_plot_rect.max.x - 140.0; // legend box width approx
                                                          // Background
                let legend_rect = Rect::from_min_max(
                    egui::pos2(x - pad, centered_plot_rect.min.y + pad),
                    egui::pos2(
                        centered_plot_rect.max.x - pad,
                        y + entries.len() as f32 * 18.0 + pad,
                    ),
                );
                ui.painter().rect_filled(
                    legend_rect,
                    4.0,
                    Color32::from_rgba_premultiplied(0, 0, 0, 96),
                );
                // Entries
                for e in entries {
                    let c = Color32::from_rgb(
                        (e.color.x * 255.0) as u8,
                        (e.color.y * 255.0) as u8,
                        (e.color.z * 255.0) as u8,
                    );
                    let swatch_rect =
                        Rect::from_min_size(egui::pos2(x, y - 6.0), egui::vec2(16.0, 8.0));
                    // Draw a small symbol representative of plot type
                    match e.plot_type {
                        crate::plots::figure::PlotType::Line
                        | crate::plots::figure::PlotType::Contour => {
                            // Line: horizontal segment
                            let ymid = swatch_rect.center().y;
                            ui.painter().line_segment(
                                [
                                    Pos2::new(swatch_rect.min.x, ymid),
                                    Pos2::new(swatch_rect.max.x, ymid),
                                ],
                                Stroke::new(2.0, c),
                            );
                        }
                        crate::plots::figure::PlotType::Scatter
                        | crate::plots::figure::PlotType::Scatter3 => {
                            // Marker: small filled circle with edge
                            let center = swatch_rect.center();
                            ui.painter().circle_filled(center, 3.5, c);
                            ui.painter().circle_stroke(
                                center,
                                3.5,
                                Stroke::new(1.0, Color32::BLACK),
                            );
                        }
                        crate::plots::figure::PlotType::Bar
                        | crate::plots::figure::PlotType::Area
                        | crate::plots::figure::PlotType::Surface
                        | crate::plots::figure::PlotType::Pie
                        | crate::plots::figure::PlotType::Image
                        | crate::plots::figure::PlotType::ContourFill => {
                            // Filled rect
                            ui.painter().rect_filled(swatch_rect, 2.0, c);
                            ui.painter().rect_stroke(
                                swatch_rect,
                                2.0,
                                Stroke::new(1.0, Color32::BLACK),
                            );
                        }
                        crate::plots::figure::PlotType::ErrorBar
                        | crate::plots::figure::PlotType::Stairs
                        | crate::plots::figure::PlotType::Stem
                        | crate::plots::figure::PlotType::Quiver => {
                            // Generic: line + small tick/arrow
                            let ymid = swatch_rect.center().y;
                            ui.painter().line_segment(
                                [
                                    Pos2::new(swatch_rect.min.x, ymid),
                                    Pos2::new(swatch_rect.max.x - 4.0, ymid),
                                ],
                                Stroke::new(1.5, c),
                            );
                            ui.painter().line_segment(
                                [
                                    Pos2::new(swatch_rect.max.x - 4.0, ymid - 3.0),
                                    Pos2::new(swatch_rect.max.x, ymid),
                                ],
                                Stroke::new(1.0, c),
                            );
                        }
                    }
                    ui.painter().text(
                        egui::pos2(x + 22.0, y),
                        Align2::LEFT_CENTER,
                        &e.label,
                        FontId::proportional(12.0),
                        Color32::WHITE,
                    );
                    y += 18.0;
                }
            }
        }

        // Draw colorbar if enabled
        if plot_renderer.overlay_colorbar_enabled() {
            // Simple vertical colorbar on the right side inside plot
            let bar_width = 12.0;
            let pad = 8.0;
            let bar_rect = Rect::from_min_max(
                egui::pos2(
                    centered_plot_rect.max.x - bar_width - pad,
                    centered_plot_rect.min.y + pad,
                ),
                egui::pos2(
                    centered_plot_rect.max.x - pad,
                    centered_plot_rect.max.y - pad,
                ),
            );
            // Fill with gradient according to colormap
            let steps = 64;
            for i in 0..steps {
                let t0 = i as f32 / steps as f32;
                let t1 = (i + 1) as f32 / steps as f32;
                let y0 = bar_rect.min.y + (1.0 - t0) * bar_rect.height();
                let y1 = bar_rect.min.y + (1.0 - t1) * bar_rect.height();
                let cmap = plot_renderer.overlay_colormap();
                let c = cmap.map_value(t0);
                let col = Color32::from_rgb(
                    (c.x * 255.0) as u8,
                    (c.y * 255.0) as u8,
                    (c.z * 255.0) as u8,
                );
                ui.painter().rect_filled(
                    Rect::from_min_max(
                        egui::pos2(bar_rect.min.x, y1),
                        egui::pos2(bar_rect.max.x, y0),
                    ),
                    0.0,
                    col,
                );
            }
            ui.painter()
                .rect_stroke(bar_rect, 0.0, Stroke::new(1.0, Color32::WHITE));
        }

        centered_plot_rect
    }

    /// Compute subplot rectangles within a given plot area for a rows x cols grid (row-major)
    pub fn compute_subplot_rects(
        &self,
        outer: Rect,
        rows: usize,
        cols: usize,
        hgap: f32,
        vgap: f32,
    ) -> Vec<Rect> {
        let rows = rows.max(1) as f32;
        let cols = cols.max(1) as f32;
        let total_hgap = hgap * (cols - 1.0);
        let total_vgap = vgap * (rows - 1.0);
        let cell_w = ((outer.width()).max(1.0) - total_hgap).max(1.0) / cols;
        let cell_h = ((outer.height()).max(1.0) - total_vgap).max(1.0) / rows;
        let mut rects = Vec::new();
        for r in 0..rows as i32 {
            for c in 0..cols as i32 {
                let x = outer.min.x + c as f32 * (cell_w + hgap);
                let y = outer.min.y + r as f32 * (cell_h + vgap);
                rects.push(Rect::from_min_size(
                    egui::pos2(x, y),
                    egui::vec2(cell_w, cell_h),
                ));
            }
        }
        rects
    }

    /// Draw grid lines based on data bounds
    fn draw_grid(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        view_bounds_override: Option<(f64, f64, f64, f64)>,
    ) {
        if let Some(data_bounds) = view_bounds_override
            .or_else(|| plot_renderer.view_bounds())
            .or_else(|| plot_renderer.data_bounds())
        {
            let grid_color_major = Color32::from_gray(80);
            let _grid_color_minor = Color32::from_gray(60);

            let (x_min, x_max, y_min, y_max) = data_bounds;
            let x_range = x_max - x_min;
            let y_range = y_max - y_min;

            // Calculate tick intervals
            let x_tick_interval = if plot_renderer.overlay_x_log() {
                0.0
            } else {
                plot_utils::calculate_tick_interval(x_range)
            };
            let y_tick_interval = if plot_renderer.overlay_y_log() {
                0.0
            } else {
                plot_utils::calculate_tick_interval(y_range)
            };

            // Draw vertical grid lines (linear vs log)
            if plot_renderer.overlay_x_log() {
                // Decades within [x_min, x_max]
                let start_decade = x_min.log10().floor() as i32;
                let end_decade = x_max.log10().ceil() as i32;
                for d in start_decade..=end_decade {
                    let decade = 10f64.powi(d);
                    for m in [1.0, 2.0, 5.0].iter() {
                        let x_val = decade * m;
                        if x_val < x_min || x_val > x_max {
                            continue;
                        }
                        let x_screen = plot_rect.min.x
                            + ((x_val.log10() - x_min.log10()) / (x_max.log10() - x_min.log10()))
                                as f32
                                * plot_rect.width();
                        ui.painter().line_segment(
                            [
                                Pos2::new(x_screen, plot_rect.min.y),
                                Pos2::new(x_screen, plot_rect.max.y),
                            ],
                            Stroke::new(0.8, grid_color_major),
                        );
                    }
                }
            } else {
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
            }

            // Draw horizontal grid lines (linear vs log)
            if plot_renderer.overlay_y_log() {
                let start_decade = y_min.log10().floor() as i32;
                let end_decade = y_max.log10().ceil() as i32;
                for d in start_decade..=end_decade {
                    let decade = 10f64.powi(d);
                    for m in [1.0, 2.0, 5.0].iter() {
                        let y_val = decade * m;
                        if y_val < y_min || y_val > y_max {
                            continue;
                        }
                        let y_screen = plot_rect.max.y
                            - ((y_val.log10() - y_min.log10()) / (y_max.log10() - y_min.log10()))
                                as f32
                                * plot_rect.height();
                        ui.painter().line_segment(
                            [
                                Pos2::new(plot_rect.min.x, y_screen),
                                Pos2::new(plot_rect.max.x, y_screen),
                            ],
                            Stroke::new(0.8, grid_color_major),
                        );
                    }
                }
            } else {
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
    }

    /// Draw axis ticks and numeric labels
    fn draw_axes(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        config: &OverlayConfig,
        view_bounds_override: Option<(f64, f64, f64, f64)>,
    ) {
        if let Some(data_bounds) = view_bounds_override
            .or_else(|| plot_renderer.view_bounds())
            .or_else(|| plot_renderer.data_bounds())
        {
            let (x_min, x_max, y_min, y_max) = data_bounds;
            let x_range = x_max - x_min;
            let y_range = y_max - y_min;
            let scale = config.font_scale.max(0.75);
            let tick_length = 6.0 * scale;
            let label_offset = 15.0 * scale;
            let tick_font = FontId::proportional(10.0 * scale);

            let x_log = plot_renderer.overlay_x_log();
            let y_log = plot_renderer.overlay_y_log();

            // Categorical axis support
            let (mut cat_x, mut cat_y) = (false, false);
            if let Some((is_x, labels)) = plot_renderer.overlay_categorical_labels() {
                if is_x {
                    cat_x = true;
                } else {
                    cat_y = true;
                }
                if is_x {
                    // Draw X categorical labels at integer positions (1..n)
                    for (idx, label) in labels.iter().enumerate() {
                        let x_val = (idx + 1) as f64;
                        if x_val < x_min || x_val > x_max {
                            continue;
                        }
                        let x_screen = plot_rect.min.x
                            + ((x_val - x_min) / x_range) as f32 * plot_rect.width();
                        // Tick
                        ui.painter().line_segment(
                            [
                                Pos2::new(x_screen, plot_rect.max.y),
                                Pos2::new(x_screen, plot_rect.max.y + tick_length),
                            ],
                            Stroke::new(1.0, Color32::WHITE),
                        );
                        // Label
                        let text = truncate_label(label, 14);
                        ui.painter().text(
                            Pos2::new(x_screen, plot_rect.max.y + label_offset),
                            Align2::CENTER_CENTER,
                            text,
                            tick_font.clone(),
                            Color32::from_gray(200),
                        );
                    }
                } else {
                    // Draw Y categorical labels at integer positions (1..n)
                    for (idx, label) in labels.iter().enumerate() {
                        let y_val = (idx + 1) as f64;
                        if y_val < y_min || y_val > y_max {
                            continue;
                        }
                        let y_screen = plot_rect.max.y
                            - ((y_val - y_min) / y_range) as f32 * plot_rect.height();
                        // Tick
                        ui.painter().line_segment(
                            [
                                Pos2::new(plot_rect.min.x - tick_length, y_screen),
                                Pos2::new(plot_rect.min.x, y_screen),
                            ],
                            Stroke::new(1.0, Color32::WHITE),
                        );
                        // Label
                        let text = truncate_label(label, 14);
                        ui.painter().text(
                            Pos2::new(plot_rect.min.x - label_offset, y_screen),
                            Align2::CENTER_CENTER,
                            text,
                            tick_font.clone(),
                            Color32::from_gray(200),
                        );
                    }
                }
            }

            // Draw X-axis ticks and labels (categorical handled above)
            if x_log {
                let start_decade = x_min.log10().floor() as i32;
                let end_decade = x_max.log10().ceil() as i32;
                for d in start_decade..=end_decade {
                    let decade = 10f64.powi(d);
                    let x_screen = plot_rect.min.x
                        + ((decade.log10() - x_min.log10()) / (x_max.log10() - x_min.log10()))
                            as f32
                            * plot_rect.width();
                    // Tick mark
                    ui.painter().line_segment(
                        [
                            Pos2::new(x_screen, plot_rect.max.y),
                            Pos2::new(x_screen, plot_rect.max.y + tick_length),
                        ],
                        Stroke::new(1.0, Color32::WHITE),
                    );
                    // Label like 10^d
                    ui.painter().text(
                        Pos2::new(x_screen, plot_rect.max.y + label_offset),
                        Align2::CENTER_CENTER,
                        format!("10^{}", d),
                        tick_font.clone(),
                        Color32::from_gray(200),
                    );
                }
            } else if !cat_x {
                let x_tick_interval = plot_utils::calculate_tick_interval(x_range);
                let mut x_val = (x_min / x_tick_interval).ceil() * x_tick_interval;
                while x_val <= x_max {
                    let x_screen =
                        plot_rect.min.x + ((x_val - x_min) / x_range) as f32 * plot_rect.width();
                    ui.painter().line_segment(
                        [
                            Pos2::new(x_screen, plot_rect.max.y),
                            Pos2::new(x_screen, plot_rect.max.y + tick_length),
                        ],
                        Stroke::new(1.0, Color32::WHITE),
                    );
                    ui.painter().text(
                        Pos2::new(x_screen, plot_rect.max.y + label_offset),
                        Align2::CENTER_CENTER,
                        plot_utils::format_tick_label(x_val),
                        tick_font.clone(),
                        Color32::from_gray(200),
                    );
                    x_val += x_tick_interval;
                }
            }

            // Draw Y-axis ticks and labels (categorical handled above)
            if y_log {
                let start_decade = y_min.log10().floor() as i32;
                let end_decade = y_max.log10().ceil() as i32;
                for d in start_decade..=end_decade {
                    let decade = 10f64.powi(d);
                    let y_screen = plot_rect.max.y
                        - ((decade.log10() - y_min.log10()) / (y_max.log10() - y_min.log10()))
                            as f32
                            * plot_rect.height();
                    ui.painter().line_segment(
                        [
                            Pos2::new(plot_rect.min.x - tick_length, y_screen),
                            Pos2::new(plot_rect.min.x, y_screen),
                        ],
                        Stroke::new(1.0, Color32::WHITE),
                    );
                    ui.painter().text(
                        Pos2::new(plot_rect.min.x - label_offset, y_screen),
                        Align2::CENTER_CENTER,
                        format!("10^{}", d),
                        tick_font.clone(),
                        Color32::from_gray(200),
                    );
                }
            } else if !cat_y {
                let y_tick_interval = plot_utils::calculate_tick_interval(y_range);
                let mut y_val = (y_min / y_tick_interval).ceil() * y_tick_interval;
                while y_val <= y_max {
                    let y_screen =
                        plot_rect.max.y - ((y_val - y_min) / y_range) as f32 * plot_rect.height();
                    ui.painter().line_segment(
                        [
                            Pos2::new(plot_rect.min.x - tick_length, y_screen),
                            Pos2::new(plot_rect.min.x, y_screen),
                        ],
                        Stroke::new(1.0, Color32::WHITE),
                    );
                    ui.painter().text(
                        Pos2::new(plot_rect.min.x - label_offset, y_screen),
                        Align2::CENTER_CENTER,
                        plot_utils::format_tick_label(y_val),
                        tick_font.clone(),
                        Color32::from_gray(200),
                    );
                    y_val += y_tick_interval;
                }
            }
        }
    }

    /// Draw a CAD-style XYZ orientation gizmo in the bottom-left corner of the plot rect.
    /// This is drawn in screen-space (overlay) but rotates with the current 3D camera.
    fn draw_3d_orientation_gizmo(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        font_scale: f32,
    ) {
        let cam_ref = plot_renderer
            .axes_camera(axes_index)
            .unwrap_or_else(|| plot_renderer.camera());
        let cam = cam_ref.clone();

        let forward = (cam.target - cam.position).normalize_or_zero();
        if forward.length_squared() < 1e-9 {
            return;
        }
        let world_up = cam.up.normalize_or_zero();
        let right = forward.cross(world_up).normalize_or_zero();
        if right.length_squared() < 1e-9 {
            return;
        }
        let up = right.cross(forward).normalize_or_zero();
        if up.length_squared() < 1e-9 {
            return;
        }

        let scale = font_scale.max(0.75);
        let base = plot_rect.width().min(plot_rect.height()).max(1.0);
        let gizmo_size = (base * 0.16).clamp(44.0, 110.0) * scale;
        let pad = (10.0 * scale).round();
        let origin = Pos2::new(plot_rect.min.x + pad, plot_rect.max.y - pad);

        struct AxisItem {
            label: &'static str,
            dir_world: Vec3,
            color: Color32,
            z_sort: f32,
        }

        let mut axes = [
            AxisItem {
                label: "X",
                dir_world: Vec3::X,
                color: Color32::from_rgb(235, 80, 80),
                z_sort: 0.0,
            },
            AxisItem {
                label: "Y",
                dir_world: Vec3::Y,
                color: Color32::from_rgb(90, 220, 120),
                z_sort: 0.0,
            },
            AxisItem {
                label: "Z",
                dir_world: Vec3::Z,
                color: Color32::from_rgb(90, 160, 255),
                z_sort: 0.0,
            },
        ];

        // Transform world axis directions into camera space and use the camera-space z as a
        // painter's-order hint (draw farther axes first so nearer ones sit on top).
        for a in axes.iter_mut() {
            let x = a.dir_world.dot(right);
            let y = a.dir_world.dot(up);
            let z = a.dir_world.dot(-forward);
            a.z_sort = z;
            a.dir_world = Vec3::new(x, y, z);
        }
        axes.sort_by(|a, b| {
            a.z_sort
                .partial_cmp(&b.z_sort)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let painter = ui.painter();
        // Subtle background to keep gizmo readable over bright surfaces.
        let bg_rect = Rect::from_min_size(
            Pos2::new(origin.x - 4.0 * scale, origin.y - gizmo_size - 8.0 * scale),
            egui::Vec2::new(gizmo_size + 12.0 * scale, gizmo_size + 12.0 * scale),
        );
        painter.rect_filled(
            bg_rect,
            6.0 * scale,
            Color32::from_rgba_premultiplied(0, 0, 0, 70),
        );

        painter.circle_filled(origin, 2.0 * scale, Color32::from_gray(210));

        let axis_len = gizmo_size * 0.65;
        let head_len = (8.0 * scale).min(axis_len * 0.35);
        let head_w = 5.0 * scale;
        let font = FontId::proportional(11.0 * scale);

        for a in axes.iter() {
            let dir2 = egui::Vec2::new(a.dir_world.x, -a.dir_world.y);
            let mag = dir2.length();
            if !mag.is_finite() || mag < 1e-4 {
                continue;
            }
            let d = dir2 / mag;

            let end = origin + d * axis_len;
            let stroke = Stroke::new(2.0 * scale, a.color);
            painter.line_segment([origin, end], stroke);

            // Arrow head
            let base = end - d * head_len;
            let perp = egui::Vec2::new(-d.y, d.x);
            painter.line_segment([end, base + perp * head_w], stroke);
            painter.line_segment([end, base - perp * head_w], stroke);

            // Label near arrow tip
            let label_pos = end + d * (10.0 * scale);
            painter.text(
                label_pos,
                Align2::CENTER_CENTER,
                a.label,
                font.clone(),
                a.color,
            );
        }
    }

    /// Draw dynamic tick labels for the 3D origin triad axes (X/Y/Z).
    /// These labels are screen-space (egui) but are positioned by projecting 3D points.
    fn draw_3d_origin_axis_ticks(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        font_scale: f32,
    ) {
        let cam_ref = plot_renderer
            .axes_camera(axes_index)
            .unwrap_or_else(|| plot_renderer.camera());
        let mut cam = cam_ref.clone();
        let w = plot_rect.width().max(1.0);
        let h = plot_rect.height().max(1.0);
        cam.update_aspect_ratio(w / h);
        let view_proj = cam.view_proj_matrix();

        let project = |p: Vec3| -> Option<Pos2> {
            let clip: Vec4 = view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
            if !clip.w.is_finite() || clip.w.abs() < 1e-6 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            if !(ndc.x.is_finite() && ndc.y.is_finite()) {
                return None;
            }
            let sx = plot_rect.min.x + ((ndc.x + 1.0) * 0.5) * plot_rect.width();
            let sy = plot_rect.min.y + ((1.0 - ndc.y) * 0.5) * plot_rect.height();
            Some(Pos2::new(sx, sy))
        };

        let nice_step = |raw: f64| -> f64 {
            if !raw.is_finite() || raw <= 0.0 {
                return 1.0;
            }
            let pow10 = 10.0_f64.powf(raw.log10().floor());
            let norm = raw / pow10;
            let mult = if norm <= 1.0 {
                1.0
            } else if norm <= 2.0 {
                2.0
            } else if norm <= 5.0 {
                5.0
            } else {
                10.0
            };
            mult * pow10
        };

        // Use the same basic heuristic as the renderer: choose a major tick spacing based on
        // projected pixels per world unit near the origin.
        let origin = Vec3::ZERO;
        let px_per_world = match (project(origin), project(origin + Vec3::X)) {
            (Some(a), Some(b)) => ((b.x - a.x).hypot(b.y - a.y) as f64).max(1e-3),
            _ => 1.0,
        };
        let desired_major_px = 120.0_f64;
        let major_step = nice_step((desired_major_px / px_per_world).max(1e-6));
        if !(major_step.is_finite() && major_step > 0.0) {
            return;
        }
        let axis_len = (major_step as f32 * 5.0).max(0.5);

        let scale = font_scale.max(0.75);
        let font = FontId::proportional(11.0 * scale);
        let painter = ui.painter();
        let col_x = Color32::from_rgb(235, 80, 80);
        let col_y = Color32::from_rgb(90, 220, 120);
        let col_z = Color32::from_rgb(90, 160, 255);

        let draw_axis = |axis: Vec3, color: Color32| {
            for i in 1..=6 {
                let t = (i as f32) * (major_step as f32);
                if t >= axis_len * 0.999 {
                    break;
                }
                let p = origin + axis * t;
                let Some(pos) = project(p) else { continue };
                // Offset labels slightly away from the axis in screen-space based on camera right/up.
                let offset = egui::Vec2::new(6.0 * scale, -6.0 * scale);
                painter.text(
                    pos + offset,
                    Align2::LEFT_CENTER,
                    plot_utils::format_tick_label((i as f64) * major_step),
                    font.clone(),
                    color,
                );
            }
        };
        draw_axis(Vec3::X, col_x);
        draw_axis(Vec3::Y, col_y);
        draw_axis(Vec3::Z, col_z);
    }

    /// Draw plot title
    fn draw_title(&self, ui: &mut egui::Ui, plot_rect: Rect, title: &str, scale: f32) {
        let scale = scale.max(0.75);
        ui.painter().text(
            Pos2::new(plot_rect.center().x, plot_rect.min.y - 20.0 * scale),
            Align2::CENTER_CENTER,
            title,
            FontId::proportional(16.0 * scale),
            Color32::WHITE,
        );
    }

    /// Draw X-axis label
    fn draw_x_label(&self, ui: &mut egui::Ui, plot_rect: Rect, label: &str, scale: f32) {
        let scale = scale.max(0.75);
        ui.painter().text(
            Pos2::new(plot_rect.center().x, plot_rect.max.y + 40.0 * scale),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(14.0 * scale),
            Color32::WHITE,
        );
    }

    /// Draw Y-axis label
    fn draw_y_label(&self, ui: &mut egui::Ui, plot_rect: Rect, label: &str, scale: f32) {
        let scale = scale.max(0.75);
        ui.painter().text(
            Pos2::new(plot_rect.min.x - 40.0 * scale, plot_rect.center().y),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(14.0 * scale),
            Color32::WHITE,
        );
    }

    /// Get the plot area from the last frame
    pub fn plot_area(&self) -> Option<Rect> {
        self.plot_area
    }

    /// Get toolbar rectangle from last frame
    pub fn toolbar_rect(&self) -> Option<Rect> {
        self.toolbar_rect
    }

    /// Get sidebar rectangle from last frame
    pub fn sidebar_rect(&self) -> Option<Rect> {
        self.sidebar_rect
    }

    pub fn take_toolbar_actions(&mut self) -> (bool, bool, bool, Option<bool>, Option<bool>) {
        let out = (
            self.want_save_png,
            self.want_save_svg,
            self.want_reset_view,
            self.want_toggle_grid.take(),
            self.want_toggle_legend.take(),
        );
        self.want_save_png = false;
        self.want_save_svg = false;
        self.want_reset_view = false;
        out
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
                        8.0,                             // rounded corners
                        Color32::from_rgb(60, 130, 200), // Dystr brand color placeholder
                    );

                    ui.painter().text(
                        logo_rect.center(),
                        Align2::CENTER_CENTER,
                        "D",
                        FontId::proportional(40.0),
                        Color32::WHITE,
                    );

                    ui.add_space(15.0);

                    ui.heading("Welcome to RunMat");
                    ui.add_space(10.0);

                    ui.label("RunMat is a high-performance MATLAB-compatible");
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
                                eprintln!("Failed to open browser: {e}");
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

fn truncate_label(label: &str, max_len: usize) -> String {
    if label.chars().count() <= max_len {
        return label.to_string();
    }
    let mut out = String::new();
    for (i, ch) in label.chars().enumerate() {
        if i >= max_len - 1 {
            break;
        }
        out.push(ch);
    }
    out.push('…');
    out
}
