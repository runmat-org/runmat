//! GUI overlay system for interactive plot controls and annotations
//!
//! This module handles the egui-based UI that sits on top of the WGPU-rendered
//! plot content, providing controls, axis labels, grid lines, and titles.

use crate::core::{plot_utils, PlotRenderer};
use crate::plots::TextStyle;
use crate::styling::{ModernDarkTheme, PlotThemeConfig, ThemeVariant};
use egui::{Align2, Color32, Context, FontId, Pos2, Rect, Stroke};
use glam::{Vec3, Vec4};

/// GUI overlay manager for plot annotations and controls
pub struct PlotOverlay {
    /// Current theme
    theme: PlotThemeConfig,

    /// Cached plot area from last frame
    plot_area: Option<Rect>,
    /// Cached per-axes content rectangles from last frame (egui points, snapped)
    axes_plot_rects: Vec<Rect>,
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

#[derive(Debug, Clone, Copy)]
struct PanelLayout {
    plot_rect: Rect,
    frame_rect: Rect,
    title_rect: Rect,
    x_label_rect: Rect,
    y_label_rect: Rect,
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
    const SUBPLOT_GAP_POINTS: f32 = 6.0;

    fn style_color(style: &TextStyle, fallback: Color32) -> Color32 {
        style
            .color
            .map(|c| {
                Color32::from_rgb(
                    (c.x.clamp(0.0, 1.0) * 255.0) as u8,
                    (c.y.clamp(0.0, 1.0) * 255.0) as u8,
                    (c.z.clamp(0.0, 1.0) * 255.0) as u8,
                )
            })
            .unwrap_or(fallback)
    }

    fn style_font_size(style: &TextStyle, default_size: f32, scale: f32) -> f32 {
        style.font_size.unwrap_or(default_size) * scale.max(0.75)
    }

    fn style_is_bold(style: &TextStyle) -> bool {
        style
            .font_weight
            .as_deref()
            .map(|weight| weight.eq_ignore_ascii_case("bold"))
            .unwrap_or(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn paint_styled_text(
        painter: &egui::Painter,
        pos: Pos2,
        align: Align2,
        text: &str,
        font_size: f32,
        color: Color32,
        bold: bool,
        shadow_alpha: u8,
    ) {
        let font = FontId::proportional(font_size);
        painter.text(
            pos + egui::vec2(1.0, 1.0),
            align,
            text,
            font.clone(),
            Color32::from_rgba_premultiplied(0, 0, 0, shadow_alpha),
        );
        painter.text(pos, align, text, font.clone(), color);
        if bold {
            painter.text(pos + egui::vec2(0.6, 0.0), align, text, font.clone(), color);
            painter.text(pos + egui::vec2(0.0, 0.6), align, text, font.clone(), color);
            painter.text(pos + egui::vec2(0.6, 0.6), align, text, font, color);
        }
    }

    fn label_stride(labels: &[String], axis_span_px: f32, font_size_px: f32) -> usize {
        if labels.len() <= 1 || axis_span_px <= 1.0 {
            return 1;
        }
        let max_chars = labels
            .iter()
            .map(|label| truncate_label(label, 14).chars().count())
            .max()
            .unwrap_or(0) as f32;
        let estimated_label_width = (max_chars * font_size_px * 0.55).max(font_size_px * 2.0);
        let slot_width = (axis_span_px / labels.len() as f32).max(1.0);
        ((estimated_label_width / slot_width).ceil().max(1.0)) as usize
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_histogram_axis_ticks(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        ppp: f32,
        axis_color: Color32,
        label_color: Color32,
        tick_length: f32,
        label_offset: f32,
        tick_font: FontId,
        border_bottom: f32,
        x_min: f64,
        x_max: f64,
        edges: &[f64],
    ) {
        if edges.len() < 2 || (x_max - x_min).abs() <= f64::EPSILON {
            return;
        }
        let labels: Vec<String> = edges
            .iter()
            .map(|value| plot_utils::format_tick_label(*value))
            .collect();
        let stride = Self::label_stride(&labels, plot_rect.width(), tick_font.size);
        let denom = (edges.len() - 1) as f64;
        for (idx, label) in labels.iter().enumerate() {
            if idx != 0 && idx != labels.len() - 1 && idx % stride != 0 {
                continue;
            }
            let frac = idx as f64 / denom;
            let x_val = x_min + frac * (x_max - x_min);
            let x_screen =
                plot_rect.min.x + ((x_val - x_min) / (x_max - x_min)) as f32 * plot_rect.width();
            let x_screen = Self::snap_coord(x_screen, ppp);
            ui.painter().line_segment(
                [
                    Pos2::new(x_screen, border_bottom),
                    Pos2::new(x_screen, border_bottom + tick_length),
                ],
                Stroke::new(1.0, axis_color),
            );
            ui.painter().text(
                Pos2::new(x_screen, border_bottom + label_offset),
                Align2::CENTER_CENTER,
                label,
                tick_font.clone(),
                label_color,
            );
        }
    }

    /// Create a new plot overlay
    pub fn new() -> Self {
        Self {
            theme: PlotThemeConfig::default(),
            plot_area: None,
            axes_plot_rects: Vec::new(),
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

    pub fn set_theme_config(&mut self, theme: PlotThemeConfig) {
        self.theme = theme;
    }

    fn has_visible_text(text: Option<&str>) -> bool {
        text.map(|s| !s.trim().is_empty()).unwrap_or(false)
    }

    fn approx_text_width_points(text: &str, font_size: f32) -> f32 {
        (text.chars().count() as f32) * font_size * 0.56
    }

    fn estimate_y_axis_band_width(
        &self,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        has_y_label: bool,
        scale: f32,
    ) -> f32 {
        let tick_font_size = 10.0 * scale;
        let label_offset = 15.0 * scale;

        let y_log = plot_renderer.overlay_y_log_for_axes(axes_index);
        let categorical = plot_renderer
            .overlay_categorical_labels_for_axes(axes_index)
            .filter(|(is_x, _)| !*is_x)
            .map(|(_, labels)| labels)
            .or_else(|| {
                plot_renderer
                    .overlay_categorical_labels()
                    .and_then(|(is_x, labels)| if !is_x { Some(labels.clone()) } else { None })
            });

        let max_label_width = if let Some(labels) = categorical {
            labels
                .iter()
                .map(|label| {
                    Self::approx_text_width_points(&truncate_label(label, 14), tick_font_size)
                })
                .fold(0.0_f32, f32::max)
        } else if let Some((_x_min, _x_max, y_min, y_max)) =
            plot_renderer.overlay_display_bounds_for_axes(axes_index)
        {
            if y_log && y_min > 0.0 && y_max > 0.0 {
                let start_decade = y_min.log10().floor() as i32;
                let end_decade = y_max.log10().ceil() as i32;
                (start_decade..=end_decade)
                    .map(|d| Self::approx_text_width_points(&format!("10^{d}"), tick_font_size))
                    .fold(0.0_f32, f32::max)
            } else {
                plot_utils::generate_major_ticks(y_min, y_max)
                    .into_iter()
                    .map(plot_utils::format_tick_label)
                    .map(|label| Self::approx_text_width_points(&label, tick_font_size))
                    .fold(0.0_f32, f32::max)
            }
        } else {
            Self::approx_text_width_points("-1.00", tick_font_size)
        };

        let y_tick_zone = label_offset + max_label_width * 0.5 + 4.0 * scale;
        let y_label_zone = if has_y_label {
            11.0 * scale
        } else {
            4.0 * scale
        };
        (y_tick_zone + y_label_zone).max(24.0 * scale)
    }

    #[allow(clippy::too_many_arguments)]
    fn layout_2d_panel(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        title: Option<&str>,
        x_label: Option<&str>,
        y_label: Option<&str>,
        scale: f32,
    ) -> PanelLayout {
        let scale = scale.max(0.75);
        let has_title = Self::has_visible_text(title);
        let has_x_label = Self::has_visible_text(x_label);
        let has_y_label = Self::has_visible_text(y_label);
        let outer_w = outer.width().max(1.0);
        let outer_h = outer.height().max(1.0);
        let title_gap = if has_title { 4.0 * scale } else { 1.5 * scale };
        let x_gap = 4.0 * scale;
        let right_pad = 3.0 * scale;

        let mut title_h = if has_title {
            (28.0 * scale).min(outer_h * 0.16)
        } else {
            0.0
        };
        let mut x_h = ((24.0 + if has_x_label { 14.0 } else { 0.0 }) * scale).min(outer_h * 0.28);
        let y_band_estimate =
            self.estimate_y_axis_band_width(plot_renderer, axes_index, has_y_label, scale);
        let mut y_w = y_band_estimate.min(outer_w * 0.30);

        let min_plot_w = (outer_w * 0.56).max(44.0 * scale).min(outer_w);
        let min_plot_h = (outer_h * 0.54).max(44.0 * scale).min(outer_h);

        if outer_w - y_w < min_plot_w {
            y_w = (outer_w - min_plot_w).max(0.0);
        }

        let available_h = outer_h - title_h - title_gap - x_h - x_gap;
        if available_h < min_plot_h {
            let deficit = min_plot_h - available_h;
            let reducible = title_h + x_h;
            if reducible > 0.0 {
                let keep = ((reducible - deficit).max(0.0)) / reducible;
                title_h *= keep;
                x_h *= keep;
            }
        }

        let plot_rect = Rect::from_min_max(
            egui::pos2(outer.min.x + y_w, outer.min.y + title_h + title_gap),
            egui::pos2(
                (outer.max.x - right_pad).max(outer.min.x + y_w + 1.0),
                outer.max.y - x_h - x_gap,
            ),
        );
        let frame_rect = plot_rect;
        let title_rect = Rect::from_min_max(
            egui::pos2(outer.min.x, outer.min.y),
            egui::pos2(outer.max.x, plot_rect.min.y),
        );
        let x_label_rect = Rect::from_min_max(
            egui::pos2(outer.min.x, plot_rect.max.y),
            egui::pos2(outer.max.x, outer.max.y),
        );
        let y_label_rect = Rect::from_min_max(
            egui::pos2(outer.min.x, plot_rect.min.y),
            egui::pos2(plot_rect.min.x, plot_rect.max.y),
        );
        PanelLayout {
            plot_rect,
            frame_rect,
            title_rect,
            x_label_rect,
            y_label_rect,
        }
    }

    fn layout_3d_panel(&self, outer: Rect, title: Option<&str>, scale: f32) -> PanelLayout {
        let scale = scale.max(0.75);
        let title_h = if Self::has_visible_text(title) {
            (28.0 * scale).min(outer.height().max(1.0) * 0.16)
        } else {
            0.0
        };
        let plot_rect =
            Rect::from_min_max(egui::pos2(outer.min.x, outer.min.y + title_h), outer.max);
        let title_rect = Rect::from_min_max(outer.min, egui::pos2(outer.max.x, plot_rect.min.y));
        PanelLayout {
            plot_rect,
            frame_rect: plot_rect,
            title_rect,
            x_label_rect: plot_rect,
            y_label_rect: plot_rect,
        }
    }

    fn axes_is_3d(plot_renderer: &PlotRenderer, axes_index: usize) -> bool {
        let cam = plot_renderer
            .axes_camera(axes_index)
            .unwrap_or_else(|| plot_renderer.camera());
        matches!(
            cam.projection,
            crate::core::camera::ProjectionType::Perspective { .. }
        )
    }

    fn panel_layout_for_axes(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) -> PanelLayout {
        if Self::axes_is_3d(plot_renderer, axes_index) {
            self.layout_3d_panel(
                outer,
                plot_renderer
                    .overlay_title_for_axes(axes_index)
                    .map(|s| s.as_str()),
                scale,
            )
        } else {
            self.layout_2d_panel(
                outer,
                plot_renderer,
                axes_index,
                plot_renderer
                    .overlay_title_for_axes(axes_index)
                    .map(|s| s.as_str()),
                plot_renderer
                    .overlay_x_label_for_axes(axes_index)
                    .map(|s| s.as_str()),
                plot_renderer
                    .overlay_y_label_for_axes(axes_index)
                    .map(|s| s.as_str()),
                scale,
            )
        }
    }

    pub fn compute_subplot_plot_rects(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        font_scale: f32,
    ) -> Vec<Rect> {
        let plot_area = Self::outer_plot_area_for_axes(outer, plot_renderer);
        let (rows, cols) = plot_renderer.figure_axes_grid();
        if rows * cols <= 1 {
            vec![
                self.panel_layout_for_axes(plot_area, plot_renderer, 0, font_scale)
                    .plot_rect,
            ]
        } else {
            let rects = self.compute_subplot_rects(
                plot_area,
                rows,
                cols,
                Self::SUBPLOT_GAP_POINTS,
                Self::SUBPLOT_GAP_POINTS,
            );
            rects
                .into_iter()
                .enumerate()
                .map(|(axes_index, rect)| {
                    self.panel_layout_for_axes(rect, plot_renderer, axes_index, font_scale)
                        .plot_rect
                })
                .collect()
        }
    }

    pub fn snap_rect_to_pixels(rect: Rect, pixels_per_point: f32) -> Rect {
        let ppp = pixels_per_point.max(0.5);
        let min_x = (rect.min.x * ppp).round() / ppp;
        let min_y = (rect.min.y * ppp).round() / ppp;
        let width = (rect.width() * ppp).round().max(1.0) / ppp;
        let height = (rect.height() * ppp).round().max(1.0) / ppp;
        Rect::from_min_size(egui::pos2(min_x, min_y), egui::vec2(width, height))
    }

    fn snap_coord(value: f32, pixels_per_point: f32) -> f32 {
        let ppp = pixels_per_point.max(0.5);
        (value * ppp).round() / ppp
    }

    fn border_centerline_edges(
        plot_rect: Rect,
        pixels_per_point: f32,
        stroke_width: f32,
    ) -> (f32, f32, f32, f32) {
        let offset = stroke_width * 0.5;
        let left = Self::snap_coord(plot_rect.min.x - offset, pixels_per_point);
        let right = Self::snap_coord(plot_rect.max.x + offset, pixels_per_point);
        let top = Self::snap_coord(plot_rect.min.y - offset, pixels_per_point);
        let bottom = Self::snap_coord(plot_rect.max.y + offset, pixels_per_point);
        (left, right, top, bottom)
    }

    fn draw_2d_border(&self, ui: &mut egui::Ui, plot_rect: Rect) {
        let stroke = Stroke::new(1.5, self.theme_axis_color());
        let ppp = ui.ctx().pixels_per_point();
        let (left, right, top, bottom) =
            Self::border_centerline_edges(plot_rect, ppp, stroke.width);
        ui.painter()
            .line_segment([Pos2::new(left, top), Pos2::new(right, top)], stroke);
        ui.painter()
            .line_segment([Pos2::new(left, bottom), Pos2::new(right, bottom)], stroke);
        ui.painter()
            .line_segment([Pos2::new(left, top), Pos2::new(left, bottom)], stroke);
        ui.painter()
            .line_segment([Pos2::new(right, top), Pos2::new(right, bottom)], stroke);
    }

    fn draw_plot_box_mask(&self, ui: &mut egui::Ui, plot_rect: Rect) {
        let mask = 2.0;
        let bg = self.theme_background_color();
        let top = Rect::from_min_max(
            Pos2::new(plot_rect.min.x - mask, plot_rect.min.y - mask),
            Pos2::new(plot_rect.max.x + mask, plot_rect.min.y),
        );
        let bottom = Rect::from_min_max(
            Pos2::new(plot_rect.min.x - mask, plot_rect.max.y),
            Pos2::new(plot_rect.max.x + mask, plot_rect.max.y + mask),
        );
        let left = Rect::from_min_max(
            Pos2::new(plot_rect.min.x - mask, plot_rect.min.y - mask),
            Pos2::new(plot_rect.min.x, plot_rect.max.y + mask),
        );
        let right = Rect::from_min_max(
            Pos2::new(plot_rect.max.x, plot_rect.min.y - mask),
            Pos2::new(plot_rect.max.x + mask, plot_rect.max.y + mask),
        );
        ui.painter().rect_filled(top, 0.0, bg);
        ui.painter().rect_filled(bottom, 0.0, bg);
        ui.painter().rect_filled(left, 0.0, bg);
        ui.painter().rect_filled(right, 0.0, bg);
    }

    pub fn compute_subplot_plot_rects_snapped(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        font_scale: f32,
        pixels_per_point: f32,
    ) -> Vec<Rect> {
        self.compute_subplot_plot_rects(outer, plot_renderer, font_scale)
            .into_iter()
            .map(|rect| Self::snap_rect_to_pixels(rect, pixels_per_point))
            .collect()
    }

    pub fn outer_plot_area_for_axes(available_rect: Rect, plot_renderer: &PlotRenderer) -> Rect {
        let (rows, cols) = plot_renderer.figure_axes_grid();
        let single_axes_is_3d = rows * cols <= 1 && Self::axes_is_3d(plot_renderer, 0);
        if single_axes_is_3d || rows * cols > 1 {
            available_rect
        } else {
            available_rect.shrink2(egui::vec2(8.0, 8.0))
        }
    }

    fn theme_text_color(&self) -> Color32 {
        let text = self.theme.build_theme().get_text_color();
        Color32::from_rgba_premultiplied(
            (text.x.clamp(0.0, 1.0) * 255.0) as u8,
            (text.y.clamp(0.0, 1.0) * 255.0) as u8,
            (text.z.clamp(0.0, 1.0) * 255.0) as u8,
            (text.w.clamp(0.0, 1.0) * 255.0) as u8,
        )
    }

    fn theme_axis_color(&self) -> Color32 {
        let axis = self.theme.build_theme().get_axis_color();
        Color32::from_rgba_premultiplied(
            (axis.x.clamp(0.0, 1.0) * 255.0) as u8,
            (axis.y.clamp(0.0, 1.0) * 255.0) as u8,
            (axis.z.clamp(0.0, 1.0) * 255.0) as u8,
            (axis.w.clamp(0.0, 1.0) * 255.0) as u8,
        )
    }

    fn theme_background_color(&self) -> Color32 {
        let bg = self.theme.build_theme().get_background_color();
        Color32::from_rgba_premultiplied(
            (bg.x.clamp(0.0, 1.0) * 255.0) as u8,
            (bg.y.clamp(0.0, 1.0) * 255.0) as u8,
            (bg.z.clamp(0.0, 1.0) * 255.0) as u8,
            (bg.w.clamp(0.0, 1.0) * 255.0) as u8,
        )
    }

    fn themed_grid_colors(&self) -> (Color32, Color32) {
        let grid = self.theme.build_theme().get_grid_color();
        let major = Color32::from_rgba_premultiplied(
            (grid.x.clamp(0.0, 1.0) * 255.0) as u8,
            (grid.y.clamp(0.0, 1.0) * 255.0) as u8,
            (grid.z.clamp(0.0, 1.0) * 255.0) as u8,
            ((grid.w.clamp(0.15, 0.55)) * 255.0) as u8,
        );
        let minor = Color32::from_rgba_premultiplied(
            (grid.x.clamp(0.0, 1.0) * 255.0) as u8,
            (grid.y.clamp(0.0, 1.0) * 255.0) as u8,
            (grid.z.clamp(0.0, 1.0) * 255.0) as u8,
            ((grid.w * 0.6).clamp(0.10, 0.34) * 255.0) as u8,
        );
        (major, minor)
    }

    /// Apply theme to egui context
    pub fn apply_theme(&self, ctx: &Context) {
        match self.theme.variant {
            ThemeVariant::ModernDark => {
                ModernDarkTheme::default().apply_to_egui(ctx);
            }
            ThemeVariant::ClassicLight => {
                ctx.set_visuals(egui::Visuals::light());
            }
            ThemeVariant::HighContrast => {
                let mut visuals = egui::Visuals::dark();
                visuals.extreme_bg_color = egui::Color32::BLACK;
                visuals.widgets.noninteractive.bg_fill = egui::Color32::BLACK;
                visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::WHITE;
                ctx.set_visuals(visuals);
            }
            ThemeVariant::Custom => {
                let mut visuals = egui::Visuals::light();
                let bg = self.theme.build_theme().get_background_color();
                if bg.x + bg.y + bg.z < 1.5 {
                    visuals = egui::Visuals::dark();
                }
                ctx.set_visuals(visuals);
            }
        }

        // Make context transparent so WGPU content shows through
        let mut visuals = ctx.style().visuals.clone();
        visuals.window_fill = Color32::TRANSPARENT;
        visuals.panel_fill = Color32::TRANSPARENT;
        visuals.extreme_bg_color = Color32::TRANSPARENT;
        visuals.faint_bg_color = Color32::TRANSPARENT;
        visuals.widgets.noninteractive.bg_fill = Color32::TRANSPARENT;
        visuals.widgets.inactive.bg_fill = Color32::TRANSPARENT;
        visuals.widgets.hovered.bg_fill = Color32::TRANSPARENT;
        visuals.widgets.active.bg_fill = Color32::TRANSPARENT;
        visuals.widgets.open.bg_fill = Color32::TRANSPARENT;
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
                    let label = match self.theme.variant {
                        ThemeVariant::ModernDark => "Modern Dark",
                        ThemeVariant::ClassicLight => "Classic Light",
                        ThemeVariant::HighContrast => "High Contrast",
                        ThemeVariant::Custom => "Custom",
                    };
                    ui.label(format!("{label} (Active)"));
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
        let mut rendered_axes_rects: Vec<Rect> = Vec::new();

        let (rows, cols) = plot_renderer.figure_axes_grid();
        let plot_rect = Self::outer_plot_area_for_axes(available_rect, plot_renderer);

        // Use full available rectangular plot area (do not force square);
        // camera fitting and axis_equal settings will control aspect.
        let plot_area_rect = plot_rect;

        if rows * cols > 1 {
            let rects = self.compute_subplot_rects(
                plot_area_rect,
                rows,
                cols,
                Self::SUBPLOT_GAP_POINTS,
                Self::SUBPLOT_GAP_POINTS,
            );
            for (i, cell_rect) in rects.iter().enumerate() {
                let cam = plot_renderer
                    .axes_camera(i)
                    .unwrap_or_else(|| plot_renderer.camera());
                let panel_layout =
                    self.panel_layout_for_axes(*cell_rect, plot_renderer, i, config.font_scale);
                let r =
                    Self::snap_rect_to_pixels(panel_layout.plot_rect, ui.ctx().pixels_per_point());
                let frame_rect =
                    Self::snap_rect_to_pixels(panel_layout.frame_rect, ui.ctx().pixels_per_point());
                rendered_axes_rects.push(r);
                log::debug!(
                    target: "runmat_plot.axes_layout",
                    "computed axes panel layout axes_index={} rows={} cols={} is_3d={} cell=({}, {})..({}, {}) frame=({}, {})..({}, {}) content=({}, {})..({}, {})",
                    i,
                    rows,
                    cols,
                    Self::axes_is_3d(plot_renderer, i),
                    cell_rect.min.x,
                    cell_rect.min.y,
                    cell_rect.max.x,
                    cell_rect.max.y,
                    frame_rect.min.x,
                    frame_rect.min.y,
                    frame_rect.max.x,
                    frame_rect.max.y,
                    r.min.x,
                    r.min.y,
                    r.max.x,
                    r.max.y
                );
                if matches!(
                    cam.projection,
                    crate::core::camera::ProjectionType::Perspective { .. }
                ) {
                    if config.show_title {
                        if let Some(title) = plot_renderer.overlay_title_for_axes(i) {
                            self.draw_title_in_rect(
                                ui,
                                panel_layout.title_rect,
                                title,
                                config.font_scale,
                            );
                        }
                    }
                    self.draw_3d_orientation_gizmo(ui, r, plot_renderer, i, config.font_scale);
                    self.draw_3d_origin_axis_ticks(ui, r, plot_renderer, i, config.font_scale);
                    self.draw_projected_world_texts(ui, r, plot_renderer, i, config.font_scale);
                    for (label, pos) in plot_renderer.pie_labels_for_axes(i) {
                        self.draw_pie_label(ui, r, &label, pos, config.font_scale);
                    }
                    if plot_renderer.overlay_show_legend_for_axes(i) {
                        let entries = plot_renderer.overlay_legend_entries_for_axes(i);
                        self.draw_legend(ui, r, &entries, config.font_scale);
                    }
                    continue;
                }
                // Frame (2D only; 3D uses the axes cube instead)
                if plot_renderer.overlay_show_box_for_axes(i) {
                    self.draw_plot_box_mask(ui, r);
                    self.draw_2d_border(ui, frame_rect);
                }

                // Grid (2D)
                if config.show_grid && plot_renderer.overlay_show_grid_for_axes(i) {
                    let b = plot_renderer.view_bounds_for_axes(i);
                    self.draw_grid(ui, r, plot_renderer, b, Some(i));
                }
                // Axes (2D)
                if config.show_axes {
                    let b = plot_renderer.view_bounds_for_axes(i);
                    self.draw_axes(ui, r, plot_renderer, config, b, Some(i));
                }

                if config.show_title {
                    if let Some(title) = plot_renderer.overlay_title_for_axes(i) {
                        self.draw_title_in_rect(
                            ui,
                            panel_layout.title_rect,
                            title,
                            config.font_scale,
                        );
                    }
                }
                if !matches!(
                    cam.projection,
                    crate::core::camera::ProjectionType::Perspective { .. }
                ) {
                    if let Some(x_label) = plot_renderer.overlay_x_label_for_axes(i) {
                        self.draw_x_label_in_rect(
                            ui,
                            panel_layout.x_label_rect,
                            x_label,
                            config.font_scale,
                        );
                    }
                }
                if !matches!(
                    cam.projection,
                    crate::core::camera::ProjectionType::Perspective { .. }
                ) {
                    if let Some(y_label) = plot_renderer.overlay_y_label_for_axes(i) {
                        self.draw_y_label_in_rect(
                            ui,
                            panel_layout.y_label_rect,
                            y_label,
                            config.font_scale,
                        );
                    }
                }
                self.draw_projected_world_texts(ui, r, plot_renderer, i, config.font_scale);
                for (label, pos) in plot_renderer.pie_labels_for_axes(i) {
                    self.draw_pie_label(ui, r, &label, pos, config.font_scale);
                }
                if plot_renderer.overlay_show_legend_for_axes(i) {
                    let entries = plot_renderer.overlay_legend_entries_for_axes(i);
                    self.draw_legend(ui, r, &entries, config.font_scale);
                }
            }
        } else {
            let cam = plot_renderer.camera();
            let panel_layout =
                self.panel_layout_for_axes(plot_area_rect, plot_renderer, 0, config.font_scale);
            let centered_plot_rect =
                Self::snap_rect_to_pixels(panel_layout.plot_rect, ui.ctx().pixels_per_point());
            let centered_frame_rect =
                Self::snap_rect_to_pixels(panel_layout.frame_rect, ui.ctx().pixels_per_point());
            rendered_axes_rects.push(centered_plot_rect);
            log::debug!(
                target: "runmat_plot.axes_layout",
                "computed axes panel layout axes_index=0 rows={} cols={} is_3d={} cell=({}, {})..({}, {}) frame=({}, {})..({}, {}) content=({}, {})..({}, {})",
                rows,
                cols,
                Self::axes_is_3d(plot_renderer, 0),
                plot_area_rect.min.x,
                plot_area_rect.min.y,
                plot_area_rect.max.x,
                plot_area_rect.max.y,
                centered_frame_rect.min.x,
                centered_frame_rect.min.y,
                centered_frame_rect.max.x,
                centered_frame_rect.max.y,
                centered_plot_rect.min.x,
                centered_plot_rect.min.y,
                centered_plot_rect.max.x,
                centered_plot_rect.max.y
            );
            if config.show_title {
                if let Some(title) = plot_renderer
                    .overlay_title_for_axes(0)
                    .or(config.title.as_ref())
                {
                    self.draw_title_in_rect(ui, panel_layout.title_rect, title, config.font_scale);
                }
            }
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
                self.draw_projected_world_texts(
                    ui,
                    centered_plot_rect,
                    plot_renderer,
                    0,
                    config.font_scale,
                );
            } else {
                // Draw plot frame (2D only; 3D uses the axes cube instead)
                if plot_renderer.overlay_show_box() {
                    self.draw_plot_box_mask(ui, centered_plot_rect);
                    self.draw_2d_border(ui, centered_frame_rect);
                }
                // Draw grid if enabled
                if config.show_grid {
                    self.draw_grid(ui, centered_plot_rect, plot_renderer, None, None);
                }

                // Draw axes if enabled
                if config.show_axes {
                    self.draw_axes(ui, centered_plot_rect, plot_renderer, config, None, None);
                    // Emphasize zero baseline if within data range
                    if let Some((x_min, x_max, y_min, y_max)) = plot_renderer
                        .view_bounds()
                        .or_else(|| plot_renderer.data_bounds())
                    {
                        let axis_color = self.theme_axis_color();
                        let zero_stroke = Stroke::new(1.5, axis_color);
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
                if let Some(x_label) = plot_renderer
                    .overlay_x_label_for_axes(0)
                    .or(config.x_label.as_ref())
                {
                    self.draw_x_label_in_rect(
                        ui,
                        panel_layout.x_label_rect,
                        x_label,
                        config.font_scale,
                    );
                }
                if let Some(y_label) = plot_renderer
                    .overlay_y_label_for_axes(0)
                    .or(config.y_label.as_ref())
                {
                    self.draw_y_label_in_rect(
                        ui,
                        panel_layout.y_label_rect,
                        y_label,
                        config.font_scale,
                    );
                }
                self.draw_projected_world_texts(
                    ui,
                    centered_plot_rect,
                    plot_renderer,
                    0,
                    config.font_scale,
                );
            }
        }
        let centered_plot_rect = if rows * cols <= 1 {
            self.panel_layout_for_axes(plot_area_rect, plot_renderer, 0, config.font_scale)
                .plot_rect
        } else {
            plot_area_rect
        };
        for (label, pos) in if rows * cols <= 1 {
            plot_renderer.active_axes_pie_labels()
        } else {
            Vec::new()
        } {
            self.draw_pie_label(ui, centered_plot_rect, &label, pos, config.font_scale);
        }

        // Draw legend if enabled and entries available
        if rows * cols <= 1 && plot_renderer.overlay_show_legend() {
            self.draw_legend(
                ui,
                centered_plot_rect,
                plot_renderer.overlay_legend_entries(),
                config.font_scale,
            );
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
            let bg = plot_renderer.theme.build_theme().get_background_color();
            let bg_luma = 0.2126 * bg.x + 0.7152 * bg.y + 0.0722 * bg.z;
            let border = if bg_luma > 0.62 {
                Color32::from_gray(60)
            } else {
                Color32::WHITE
            };
            ui.painter()
                .rect_stroke(bar_rect, 0.0, Stroke::new(1.0, border));
        }

        self.axes_plot_rects = rendered_axes_rects;
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
        axes_index: Option<usize>,
    ) {
        let ppp = ui.ctx().pixels_per_point();
        let edge_eps = 0.51 / ppp.max(0.5);
        if let Some(data_bounds) = view_bounds_override
            .or_else(|| plot_renderer.view_bounds())
            .or_else(|| plot_renderer.data_bounds())
        {
            let (grid_color_major, _grid_color_minor) = self.themed_grid_colors();

            let (x_min, x_max, y_min, y_max) = data_bounds;
            let x_range = x_max - x_min;
            let y_range = y_max - y_min;

            // Calculate tick intervals
            let x_log = axes_index
                .map(|idx| plot_renderer.overlay_x_log_for_axes(idx))
                .unwrap_or_else(|| plot_renderer.overlay_x_log());
            let y_log = axes_index
                .map(|idx| plot_renderer.overlay_y_log_for_axes(idx))
                .unwrap_or_else(|| plot_renderer.overlay_y_log());

            let x_ticks = if x_log {
                Vec::new()
            } else {
                plot_utils::generate_major_ticks(x_min, x_max)
            };
            let y_ticks = if y_log {
                Vec::new()
            } else {
                plot_utils::generate_major_ticks(y_min, y_max)
            };

            // Draw vertical grid lines (linear vs log)
            if x_log {
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
                        let x_screen = Self::snap_coord(x_screen, ppp);
                        if (x_screen - plot_rect.min.x).abs() <= edge_eps
                            || (x_screen - plot_rect.max.x).abs() <= edge_eps
                        {
                            continue;
                        }
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
                for x_val in x_ticks {
                    let x_screen =
                        plot_rect.min.x + ((x_val - x_min) / x_range) as f32 * plot_rect.width();
                    let x_screen = Self::snap_coord(x_screen, ppp);
                    if (x_screen - plot_rect.min.x).abs() <= edge_eps
                        || (x_screen - plot_rect.max.x).abs() <= edge_eps
                    {
                        continue;
                    }
                    ui.painter().line_segment(
                        [
                            Pos2::new(x_screen, plot_rect.min.y),
                            Pos2::new(x_screen, plot_rect.max.y),
                        ],
                        Stroke::new(0.8, grid_color_major),
                    );
                }
            }

            // Draw horizontal grid lines (linear vs log)
            if y_log {
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
                        let y_screen = Self::snap_coord(y_screen, ppp);
                        if (y_screen - plot_rect.min.y).abs() <= edge_eps
                            || (y_screen - plot_rect.max.y).abs() <= edge_eps
                        {
                            continue;
                        }
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
                for y_val in y_ticks {
                    let y_screen =
                        plot_rect.max.y - ((y_val - y_min) / y_range) as f32 * plot_rect.height();
                    let y_screen = Self::snap_coord(y_screen, ppp);
                    if (y_screen - plot_rect.min.y).abs() <= edge_eps
                        || (y_screen - plot_rect.max.y).abs() <= edge_eps
                    {
                        continue;
                    }
                    ui.painter().line_segment(
                        [
                            Pos2::new(plot_rect.min.x, y_screen),
                            Pos2::new(plot_rect.max.x, y_screen),
                        ],
                        Stroke::new(0.8, grid_color_major),
                    );
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
        axes_index: Option<usize>,
    ) {
        let ppp = ui.ctx().pixels_per_point();
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
            let axis_color = self.theme_axis_color();
            let label_color = self.theme_text_color();
            let border_left = plot_rect.min.x;
            let border_bottom = plot_rect.max.y;

            let x_log = axes_index
                .map(|idx| plot_renderer.overlay_x_log_for_axes(idx))
                .unwrap_or_else(|| plot_renderer.overlay_x_log());
            let y_log = axes_index
                .map(|idx| plot_renderer.overlay_y_log_for_axes(idx))
                .unwrap_or_else(|| plot_renderer.overlay_y_log());

            // Histogram numeric tick support and categorical axis support
            let (mut cat_x, mut cat_y) = (false, false);
            let mut custom_hist_x = false;
            if let Some((true, edges)) =
                axes_index.and_then(|idx| plot_renderer.overlay_histogram_edges_for_axes(idx))
            {
                custom_hist_x = true;
                self.draw_histogram_axis_ticks(
                    ui,
                    plot_rect,
                    ppp,
                    axis_color,
                    label_color,
                    tick_length,
                    label_offset,
                    tick_font.clone(),
                    border_bottom,
                    x_min,
                    x_max,
                    &edges,
                );
            }
            if let Some((is_x, labels)) = axes_index
                .and_then(|idx| plot_renderer.overlay_categorical_labels_for_axes(idx))
                .or_else(|| {
                    plot_renderer
                        .overlay_categorical_labels()
                        .map(|(is_x, labels)| (is_x, labels.clone()))
                })
            {
                if is_x {
                    cat_x = true;
                } else {
                    cat_y = true;
                }
                if is_x {
                    let stride = Self::label_stride(&labels, plot_rect.width(), tick_font.size);
                    // Draw X categorical labels at integer positions (1..n)
                    for (label_idx, label) in labels.iter().enumerate() {
                        if label_idx != 0
                            && label_idx != labels.len() - 1
                            && label_idx % stride != 0
                        {
                            continue;
                        }
                        let x_val = (label_idx + 1) as f64;
                        if x_val < x_min || x_val > x_max {
                            continue;
                        }
                        let x_screen = plot_rect.min.x
                            + ((x_val - x_min) / x_range) as f32 * plot_rect.width();
                        let x_screen = Self::snap_coord(x_screen, ppp);
                        // Tick
                        ui.painter().line_segment(
                            [
                                Pos2::new(x_screen, border_bottom),
                                Pos2::new(x_screen, border_bottom + tick_length),
                            ],
                            Stroke::new(1.0, axis_color),
                        );
                        // Label
                        let text = truncate_label(label, 14);
                        ui.painter().text(
                            Pos2::new(x_screen, border_bottom + label_offset),
                            Align2::CENTER_CENTER,
                            text,
                            tick_font.clone(),
                            label_color,
                        );
                    }
                } else {
                    let stride = Self::label_stride(&labels, plot_rect.height(), tick_font.size);
                    // Draw Y categorical labels at integer positions (1..n)
                    for (label_idx, label) in labels.iter().enumerate() {
                        if label_idx != 0
                            && label_idx != labels.len() - 1
                            && label_idx % stride != 0
                        {
                            continue;
                        }
                        let y_val = (label_idx + 1) as f64;
                        if y_val < y_min || y_val > y_max {
                            continue;
                        }
                        let y_screen = plot_rect.max.y
                            - ((y_val - y_min) / y_range) as f32 * plot_rect.height();
                        let y_screen = Self::snap_coord(y_screen, ppp);
                        // Tick
                        ui.painter().line_segment(
                            [
                                Pos2::new(border_left - tick_length, y_screen),
                                Pos2::new(border_left, y_screen),
                            ],
                            Stroke::new(1.0, axis_color),
                        );
                        // Label
                        let text = truncate_label(label, 14);
                        ui.painter().text(
                            Pos2::new(border_left - label_offset, y_screen),
                            Align2::CENTER_CENTER,
                            text,
                            tick_font.clone(),
                            label_color,
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
                    let x_screen = Self::snap_coord(x_screen, ppp);
                    // Tick mark
                    ui.painter().line_segment(
                        [
                            Pos2::new(x_screen, border_bottom),
                            Pos2::new(x_screen, border_bottom + tick_length),
                        ],
                        Stroke::new(1.0, axis_color),
                    );
                    // Label like 10^d
                    ui.painter().text(
                        Pos2::new(x_screen, border_bottom + label_offset),
                        Align2::CENTER_CENTER,
                        format!("10^{}", d),
                        tick_font.clone(),
                        label_color,
                    );
                }
            } else if !cat_x && !custom_hist_x {
                for x_val in plot_utils::generate_major_ticks(x_min, x_max) {
                    let x_screen =
                        plot_rect.min.x + ((x_val - x_min) / x_range) as f32 * plot_rect.width();
                    let x_screen = Self::snap_coord(x_screen, ppp);
                    ui.painter().line_segment(
                        [
                            Pos2::new(x_screen, border_bottom),
                            Pos2::new(x_screen, border_bottom + tick_length),
                        ],
                        Stroke::new(1.0, axis_color),
                    );
                    ui.painter().text(
                        Pos2::new(x_screen, border_bottom + label_offset),
                        Align2::CENTER_CENTER,
                        plot_utils::format_tick_label(x_val),
                        tick_font.clone(),
                        label_color,
                    );
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
                    let y_screen = Self::snap_coord(y_screen, ppp);
                    ui.painter().line_segment(
                        [
                            Pos2::new(border_left - tick_length, y_screen),
                            Pos2::new(border_left, y_screen),
                        ],
                        Stroke::new(1.0, axis_color),
                    );
                    ui.painter().text(
                        Pos2::new(border_left - label_offset, y_screen),
                        Align2::CENTER_CENTER,
                        format!("10^{}", d),
                        tick_font.clone(),
                        label_color,
                    );
                }
            } else if !cat_y {
                for y_val in plot_utils::generate_major_ticks(y_min, y_max) {
                    let y_screen =
                        plot_rect.max.y - ((y_val - y_min) / y_range) as f32 * plot_rect.height();
                    let y_screen = Self::snap_coord(y_screen, ppp);
                    ui.painter().line_segment(
                        [
                            Pos2::new(border_left - tick_length, y_screen),
                            Pos2::new(border_left, y_screen),
                        ],
                        Stroke::new(1.0, axis_color),
                    );
                    ui.painter().text(
                        Pos2::new(border_left - label_offset, y_screen),
                        Align2::CENTER_CENTER,
                        plot_utils::format_tick_label(y_val),
                        tick_font.clone(),
                        label_color,
                    );
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
        let pad = (30.0 * scale).round();
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
        let font = FontId::proportional(10.0 * scale);
        let painter = ui.painter();
        let col_x = Color32::from_rgb(235, 80, 80);
        let col_y = Color32::from_rgb(90, 220, 120);
        let col_z = Color32::from_rgb(90, 160, 255);
        let panel_center = plot_rect.center();

        let outward_offset = |pos: Pos2, base: f32| {
            let dir = pos - panel_center;
            let len = dir.length().max(1.0);
            (dir / len) * base
        };

        if let Some(pos) = project(Vec3::X * axis_len * 1.10) {
            if let Some(label) = plot_renderer.overlay_x_label_for_axes(axes_index) {
                let style = plot_renderer
                    .overlay_x_label_style_for_axes(axes_index)
                    .cloned()
                    .unwrap_or_default();
                let offset = outward_offset(pos, 12.0 * scale) + egui::vec2(4.0 * scale, 0.0);
                Self::paint_styled_text(
                    painter,
                    pos + offset,
                    Align2::LEFT_CENTER,
                    label,
                    Self::style_font_size(&style, 12.0, scale),
                    Self::style_color(&style, col_x),
                    Self::style_is_bold(&style),
                    110,
                );
            }
        }
        if let Some(pos) = project(Vec3::Y * axis_len * 1.10) {
            if let Some(label) = plot_renderer.overlay_y_label_for_axes(axes_index) {
                let style = plot_renderer
                    .overlay_y_label_style_for_axes(axes_index)
                    .cloned()
                    .unwrap_or_default();
                let offset =
                    outward_offset(pos, 12.0 * scale) + egui::vec2(2.0 * scale, -2.0 * scale);
                Self::paint_styled_text(
                    painter,
                    pos + offset,
                    Align2::LEFT_CENTER,
                    label,
                    Self::style_font_size(&style, 12.0, scale),
                    Self::style_color(&style, col_y),
                    Self::style_is_bold(&style),
                    110,
                );
            }
        }
        if let Some(pos) = project(Vec3::Z * axis_len * 1.10) {
            if let Some(label) = plot_renderer.overlay_z_label_for_axes(axes_index) {
                let style = plot_renderer
                    .overlay_z_label_style_for_axes(axes_index)
                    .cloned()
                    .unwrap_or_default();
                let offset = outward_offset(pos, 12.0 * scale) + egui::vec2(0.0, -4.0 * scale);
                Self::paint_styled_text(
                    painter,
                    pos + offset,
                    Align2::LEFT_BOTTOM,
                    label,
                    Self::style_font_size(&style, 12.0, scale),
                    Self::style_color(&style, col_z),
                    Self::style_is_bold(&style),
                    110,
                );
            }
        }

        let draw_axis = |axis: Vec3, color: Color32| {
            for i in 1..=6 {
                let t = (i as f32) * (major_step as f32);
                if t >= axis_len * 0.999 {
                    break;
                }
                let p = origin + axis * t;
                let Some(pos) = project(p) else { continue };
                // Offset labels slightly away from the axis in screen-space based on camera right/up.
                let offset =
                    outward_offset(pos, 7.0 * scale) + egui::Vec2::new(3.0 * scale, -3.0 * scale);
                painter.text(
                    pos + offset + egui::vec2(1.0, 1.0),
                    Align2::LEFT_CENTER,
                    plot_utils::format_tick_label((i as f64) * major_step),
                    font.clone(),
                    Color32::from_rgba_premultiplied(0, 0, 0, 90),
                );
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

    fn draw_title_in_rect(&self, ui: &mut egui::Ui, rect: Rect, title: &str, scale: f32) {
        let scale = scale.max(0.75);
        let text_color = self.theme_text_color();
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            title,
            FontId::proportional(16.0 * scale),
            text_color,
        );
    }

    fn draw_legend(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        entries: &[crate::plots::figure::LegendEntry],
        scale: f32,
    ) {
        if entries.is_empty() {
            return;
        }
        let scale = scale.max(0.75);
        let theme = self.theme.build_theme();
        let bg = theme.get_background_color();
        let text = theme.get_text_color();
        let legend_text = Color32::from_rgb(
            (text.x.clamp(0.0, 1.0) * 255.0) as u8,
            (text.y.clamp(0.0, 1.0) * 255.0) as u8,
            (text.z.clamp(0.0, 1.0) * 255.0) as u8,
        );
        let bg_luma = 0.2126 * bg.x + 0.7152 * bg.y + 0.0722 * bg.z;
        let legend_bg = if bg_luma > 0.62 {
            Color32::from_rgba_premultiplied(255, 255, 255, 170)
        } else {
            Color32::from_rgba_premultiplied(0, 0, 0, 128)
        };
        let legend_stroke = if bg_luma > 0.62 {
            Color32::from_rgb(55, 55, 55)
        } else {
            Color32::BLACK
        };
        let pad = 8.0 * scale;
        let row_h = (16.0 * scale).clamp(13.0, 18.0);
        let swatch_w = 14.0 * scale;
        let text_x_gap = 18.0 * scale;
        let legend_w = (plot_rect.width() * 0.30).clamp(92.0, 132.0);
        let x = plot_rect.max.x - legend_w - pad;
        let mut y = plot_rect.min.y + pad + 4.0 * scale;
        let legend_rect = Rect::from_min_max(
            egui::pos2(x - pad, plot_rect.min.y + pad),
            egui::pos2(x + legend_w, y + entries.len() as f32 * row_h + pad),
        );
        ui.painter().rect_filled(legend_rect, 4.0, legend_bg);
        y += 10.0 * scale;
        for e in entries {
            let c = Color32::from_rgb(
                (e.color.x * 255.0) as u8,
                (e.color.y * 255.0) as u8,
                (e.color.z * 255.0) as u8,
            );
            let swatch_rect = Rect::from_min_size(
                egui::pos2(x, y - 5.0 * scale),
                egui::vec2(swatch_w, 7.0 * scale),
            );
            match e.plot_type {
                crate::plots::figure::PlotType::Line
                | crate::plots::figure::PlotType::Line3
                | crate::plots::figure::PlotType::Contour => {
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
                    let center = swatch_rect.center();
                    ui.painter().circle_filled(center, 3.5, c);
                    ui.painter()
                        .circle_stroke(center, 3.5, Stroke::new(1.0, legend_stroke));
                }
                crate::plots::figure::PlotType::Bar
                | crate::plots::figure::PlotType::Area
                | crate::plots::figure::PlotType::Surface
                | crate::plots::figure::PlotType::Pie
                | crate::plots::figure::PlotType::ContourFill => {
                    ui.painter().rect_filled(swatch_rect, 2.0, c);
                    ui.painter()
                        .rect_stroke(swatch_rect, 2.0, Stroke::new(1.0, legend_stroke));
                }
                crate::plots::figure::PlotType::ErrorBar
                | crate::plots::figure::PlotType::Stairs
                | crate::plots::figure::PlotType::Stem
                | crate::plots::figure::PlotType::Quiver => {
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
                egui::pos2(x + text_x_gap, y),
                Align2::LEFT_CENTER,
                &e.label,
                FontId::proportional(11.0 * scale),
                legend_text,
            );
            y += row_h;
        }
    }

    fn draw_x_label_in_rect(&self, ui: &mut egui::Ui, rect: Rect, label: &str, scale: f32) {
        let scale = scale.max(0.75);
        let text_color = self.theme_text_color();
        ui.painter().text(
            Pos2::new(rect.center().x, rect.max.y - rect.height() * 0.24),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(14.0 * scale),
            text_color,
        );
    }

    fn draw_y_label_in_rect(&self, ui: &mut egui::Ui, rect: Rect, label: &str, scale: f32) {
        let scale = scale.max(0.75);
        let text_color = self.theme_text_color();
        let galley = ui.fonts(|fonts| {
            fonts.layout_no_wrap(
                label.to_owned(),
                FontId::proportional(13.0 * scale),
                text_color,
            )
        });
        let size = galley.size();
        let center = Pos2::new(rect.min.x + rect.width() * 0.32, rect.center().y);
        let pos = Pos2::new(center.x - size.y * 0.5, center.y + size.x * 0.5);
        let mut shape = egui::epaint::TextShape::new(pos, galley, text_color);
        shape.angle = -std::f32::consts::FRAC_PI_2;
        shape.override_text_color = Some(text_color);
        ui.painter().add(shape);
    }

    fn project_world_to_screen(
        &self,
        plot_rect: Rect,
        camera: &crate::core::Camera,
        point: glam::Vec3,
    ) -> Option<Pos2> {
        let mut cam = camera.clone();
        let clip = cam.view_proj_matrix() * point.extend(1.0);
        if !clip.x.is_finite() || !clip.y.is_finite() || !clip.z.is_finite() || !clip.w.is_finite()
        {
            return None;
        }
        if clip.w.abs() < 1.0e-6 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.z < -1.1 || ndc.z > 1.1 {
            return None;
        }
        if clip.w <= 0.0
            && matches!(
                camera.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            )
        {
            return None;
        }
        let x = plot_rect.min.x + (ndc.x + 1.0) * 0.5 * plot_rect.width();
        let y = plot_rect.min.y + (1.0 - (ndc.y + 1.0) * 0.5) * plot_rect.height();
        Some(Pos2::new(x, y))
    }

    fn draw_projected_world_texts(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) {
        let Some(camera) = plot_renderer
            .axes_camera(axes_index)
            .or_else(|| Some(plot_renderer.camera()))
        else {
            return;
        };
        let annotations = plot_renderer.world_text_annotations_for_axes(axes_index);
        let is_3d = Self::axes_is_3d(plot_renderer, axes_index);
        for (position, text, style) in annotations {
            if !style.visible || text.trim().is_empty() {
                continue;
            }
            let Some(screen) = self.project_world_to_screen(plot_rect, camera, position) else {
                continue;
            };
            let color = Self::style_color(&style, self.theme_text_color());
            let font_size = Self::style_font_size(&style, 14.0, scale);
            let offset = if is_3d {
                egui::vec2(0.0, -8.0 * scale.max(0.75))
            } else {
                egui::vec2(0.0, 6.0 * scale.max(0.75))
            };
            Self::paint_styled_text(
                ui.painter(),
                screen + offset,
                Align2::CENTER_CENTER,
                &text,
                font_size,
                color,
                Self::style_is_bold(&style),
                if is_3d { 120 } else { 90 },
            );
        }
    }

    fn draw_pie_label(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        label: &str,
        pos: glam::Vec2,
        scale: f32,
    ) {
        let center = plot_rect.center();
        let radius = plot_rect.width().min(plot_rect.height()) * 0.4;
        let screen = Pos2::new(center.x + pos.x * radius, center.y - pos.y * radius);
        ui.painter().text(
            screen,
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(12.0 * scale.max(0.75)),
            self.theme_text_color(),
        );
    }

    /// Get the plot area from the last frame
    pub fn plot_area(&self) -> Option<Rect> {
        self.plot_area
    }

    /// Get per-axes snapped content rectangles from the last frame.
    pub fn axes_plot_rects(&self) -> &[Rect] {
        &self.axes_plot_rects
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
