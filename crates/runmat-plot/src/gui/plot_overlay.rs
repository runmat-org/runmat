//! GUI overlay system for interactive plot controls and annotations
//!
//! This module handles the egui-based UI that sits on top of the WGPU-rendered
//! plot content, providing controls, axis labels, grid lines, and titles.

use crate::core::{plot_utils, PlotRenderer};
use crate::geometry_scene::GeometrySceneScalarLegend;
use crate::legend_text::{with_unit, LegendPrecision};
use crate::overlay::cad_overlay::{CadOverlayActions, CadOverlayRenderInput, CadOverlayState};
use crate::plots::TextStyle;
use crate::styling::{ModernDarkTheme, PlotThemeConfig, ThemeVariant};
use egui::{Align2, Color32, Context, FontId, Pos2, Rect, Stroke};
use glam::{Vec3, Vec4};

fn transparent_frame() -> egui::Frame {
    egui::Frame::NONE
}

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

    // Toolbar state
    want_save_png: bool,
    want_save_svg: bool,
    want_reset_view: bool,
    want_toggle_grid: Option<bool>,
    want_toggle_legend: Option<bool>,

    cad_overlay: CadOverlayState,
    cad_actions: CadOverlayActions,
}

/// Configuration for the plot overlay
#[derive(Debug, Clone)]
pub struct OverlayConfig {
    /// Whether to show the top toolbar (legend/grid/reset/save).
    pub show_toolbar: bool,

    /// Whether to render the CAD/geometry overlay UI inside the plot surface.
    ///
    /// Hosted web/Tauri surfaces reserve React UI outside the native surface
    /// instead, because native-layer surfaces cannot be occluded by DOM.
    pub show_cad_overlay: bool,

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
            show_toolbar: true,
            show_cad_overlay: true,
            font_scale: 1.0,
            show_grid: true,
            show_axes: true,
            show_title: true,
            // No placeholder text: an axes without a title or labels shows none, as in MATLAB.
            title: None,
            x_label: None,
            y_label: None,
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

    fn axes_tick_font_size(
        plot_renderer: &PlotRenderer,
        axes_index: Option<usize>,
        scale: f32,
    ) -> f32 {
        let font_size = axes_index
            .and_then(|idx| plot_renderer.overlay_axes_style_for_axes(idx))
            .and_then(|style| style.font_size)
            .unwrap_or(10.0);
        font_size * scale.max(0.75)
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
    ) {
        // One fill per glyph: a drop shadow doubles text on a light page, so plot
        // text draws without one.
        let font = FontId::proportional(font_size);
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
        // Ticks at nice data values, as MATLAB's histogram axes have; each is placed through
        // the bin edges onto the bars, which sit at evenly spaced positions.
        let denom = (edges.len() - 1) as f64;
        let (first, last) = (edges[0], edges[edges.len() - 1]);
        let place = |value: f64| -> Option<f64> {
            let i = edges
                .windows(2)
                .position(|w| value >= w[0] && value <= w[1])?;
            let width = edges[i + 1] - edges[i];
            let within = if width > 0.0 {
                (value - edges[i]) / width
            } else {
                0.0
            };
            Some((i as f64 + within) / denom)
        };
        for value in plot_utils::generate_major_ticks(first.min(last), first.max(last)) {
            let Some(frac) = place(value) else {
                continue;
            };
            let label = plot_utils::format_tick_label(value);
            let x_val = x_min + frac * (x_max - x_min);
            let x_screen =
                plot_rect.min.x + ((x_val - x_min) / (x_max - x_min)) as f32 * plot_rect.width();
            let x_screen = Self::snap_coord(x_screen, ppp);
            ui.painter().line_segment(
                [
                    Pos2::new(x_screen, border_bottom),
                    Pos2::new(x_screen, border_bottom + tick_length),
                ],
                Stroke::new(1.0_f32, axis_color),
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
            want_save_png: false,
            want_save_svg: false,
            want_reset_view: false,
            want_toggle_grid: None,
            want_toggle_legend: None,
            cad_overlay: CadOverlayState::default(),
            cad_actions: CadOverlayActions::default(),
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

    fn rotated_text_extent(width: f32, height: f32, angle_degrees: f64) -> (f32, f32) {
        if angle_degrees.abs() <= f64::EPSILON {
            return (width, height);
        }
        let radians = (angle_degrees as f32).to_radians();
        let sin = radians.sin().abs();
        let cos = radians.cos().abs();
        (width * cos + height * sin, width * sin + height * cos)
    }

    fn estimate_x_axis_max_tick_label_width(
        &self,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) -> f32 {
        let tick_font_size = Self::axes_tick_font_size(plot_renderer, Some(axes_index), scale);
        let labels = self.estimate_x_axis_tick_labels(plot_renderer, axes_index);
        if labels.is_empty() {
            Self::approx_text_width_points("-1.00", tick_font_size)
        } else {
            labels
                .iter()
                .map(|label| Self::approx_text_width_points(label, tick_font_size))
                .fold(0.0_f32, f32::max)
        }
    }

    fn estimate_x_axis_tick_labels(
        &self,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
    ) -> Vec<String> {
        let x_log = plot_renderer.overlay_x_log_for_axes(axes_index);
        let explicit_tick_labels = plot_renderer
            .overlay_x_tick_labels_for_axes(axes_index)
            .filter(|labels| !labels.is_empty());
        let x_tick_format = plot_renderer.overlay_x_tick_format_for_axes(axes_index);
        let categorical_tick_labels = plot_renderer
            .overlay_categorical_labels_for_axes(axes_index)
            .and_then(|(is_x, labels)| if is_x { Some(labels) } else { None });

        if let Some(labels) = explicit_tick_labels {
            labels
                .iter()
                .map(|label| truncate_label(label, 14))
                .collect()
        } else if let Some(labels) = categorical_tick_labels {
            labels
                .iter()
                .map(|label| truncate_label(label, 14))
                .collect()
        } else if let Some((x_min, x_max, _y_min, _y_max)) =
            plot_renderer.overlay_display_bounds_for_axes(axes_index)
        {
            if x_log && x_min > 0.0 && x_max > 0.0 {
                let start_decade = x_min.log10().floor() as i32;
                let end_decade = x_max.log10().ceil() as i32;
                (start_decade..=end_decade)
                    .filter(|d| Self::tick_within_log_limits(10f64.powi(*d), x_min, x_max))
                    .map(Self::power_of_ten_label)
                    .collect()
            } else {
                let formatter = plot_utils::TickLabelFormatter::new(x_tick_format.as_deref());
                plot_utils::generate_major_ticks(x_min, x_max)
                    .into_iter()
                    .map(|value| formatter.format(value))
                    .collect()
            }
        } else {
            vec!["-1.00".to_string(), "1.00".to_string()]
        }
    }

    const Y_LABEL_FONT_POINTS: f32 = 13.0;

    fn estimate_y_axis_band_width(
        &self,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        has_y_label: bool,
        scale: f32,
    ) -> f32 {
        let tick_font_size = Self::axes_tick_font_size(plot_renderer, Some(axes_index), scale);
        let label_offset = 15.0 * scale;

        let y_log = plot_renderer.overlay_y_log_for_axes(axes_index);
        let categorical = plot_renderer
            .overlay_categorical_labels_for_axes(axes_index)
            .filter(|(is_x, _)| !*is_x)
            .map(|(_, labels)| labels);

        // Explicit tick labels (yticklabels) or tick values (yticks) are what will be drawn, so
        // measure those before falling back to estimated automatic ticks.
        let explicit_labels = plot_renderer
            .overlay_y_tick_labels_for_axes(axes_index)
            .filter(|labels| !labels.is_empty())
            .or_else(|| {
                plot_renderer
                    .overlay_y_ticks_for_axes(axes_index)
                    .map(|ticks| {
                        ticks
                            .into_iter()
                            .map(plot_utils::format_tick_label)
                            .collect::<Vec<_>>()
                    })
            });
        let max_label_width = if let Some(labels) = categorical.or(explicit_labels) {
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
                    .filter(|d| Self::tick_within_log_limits(10f64.powi(*d), y_min, y_max))
                    .map(|d| {
                        Self::approx_text_width_points(&Self::power_of_ten_label(d), tick_font_size)
                    })
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

        let y_angle = plot_renderer.overlay_y_tick_label_rotation_for_axes(axes_index);
        let (label_width, _) = Self::rotated_text_extent(max_label_width, tick_font_size, y_angle);
        let y_tick_zone = label_offset + label_width * 0.5 + 4.0 * scale;
        // The rotated title needs one line of its own (13pt text) plus a gap from the tick labels.
        let y_label_zone = if has_y_label {
            Self::Y_LABEL_FONT_POINTS * scale * 1.25 + 6.0 * scale
        } else {
            4.0 * scale
        };
        (y_tick_zone + y_label_zone).max(24.0 * scale)
    }

    fn estimate_x_axis_edge_padding(
        &self,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) -> f32 {
        let tick_font_size = Self::axes_tick_font_size(plot_renderer, Some(axes_index), scale);
        let labels = self.estimate_x_axis_tick_labels(plot_renderer, axes_index);

        let max_edge_label_width = if labels.is_empty() {
            Self::approx_text_width_points("-1.00", tick_font_size)
        } else if labels.len() == 1 {
            Self::approx_text_width_points(&labels[0], tick_font_size)
        } else {
            let left = Self::approx_text_width_points(&labels[0], tick_font_size);
            let right = Self::approx_text_width_points(&labels[labels.len() - 1], tick_font_size);
            left.max(right)
        };

        let x_angle = plot_renderer.overlay_x_tick_label_rotation_for_axes(axes_index);
        let (label_width, _) =
            Self::rotated_text_extent(max_edge_label_width, tick_font_size, x_angle);
        (label_width * 0.5 + 3.0 * scale).max(6.0 * scale)
    }

    #[allow(clippy::too_many_arguments)]
    fn layout_2d_panel(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        title: Option<&str>,
        subtitle: Option<&str>,
        x_label: Option<&str>,
        y_label: Option<&str>,
        scale: f32,
    ) -> PanelLayout {
        let scale = scale.max(0.75);
        let has_title = Self::has_visible_text(title);
        let has_subtitle = Self::has_visible_text(subtitle);
        let has_x_label = Self::has_visible_text(x_label);
        let has_y_label = Self::has_visible_text(y_label);
        let outer_w = outer.width().max(1.0);
        let outer_h = outer.height().max(1.0);
        let has_title_band = has_title || has_subtitle;
        let title_gap = if has_title_band {
            4.0 * scale
        } else {
            1.5 * scale
        };
        let x_gap = 4.0 * scale;
        let x_edge_pad = self.estimate_x_axis_edge_padding(plot_renderer, axes_index, scale);
        let mut right_pad = (3.0 * scale).max(x_edge_pad + 1.0 * scale);

        let mut title_h = if has_title_band {
            let base = if has_title && has_subtitle {
                44.0
            } else {
                28.0
            };
            // A multi-line title (sprintf with \n, or a cell of lines) needs a line per row.
            let extra = Self::extra_title_lines(title) as f32 * 19.0;
            ((base + extra) * scale).min(outer_h * 0.22)
        } else {
            0.0
        };
        let x_angle = plot_renderer.overlay_x_tick_label_rotation_for_axes(axes_index);
        let x_tick_label_width =
            self.estimate_x_axis_max_tick_label_width(plot_renderer, axes_index, scale);
        let (_, x_tick_label_height) =
            Self::rotated_text_extent(x_tick_label_width, 13.0 * scale, x_angle);
        let x_tick_band = (15.0 * scale + x_tick_label_height + 4.0 * scale).max(24.0 * scale);
        let mut x_h =
            (x_tick_band + if has_x_label { 14.0 * scale } else { 0.0 }).min(outer_h * 0.28);
        let y_band_estimate =
            self.estimate_y_axis_band_width(plot_renderer, axes_index, has_y_label, scale);
        let mut y_w = y_band_estimate.min(outer_w * 0.30);

        let min_plot_w = (outer_w * 0.56).max(44.0 * scale).min(outer_w);
        let min_plot_h = (outer_h * 0.54).max(44.0 * scale).min(outer_h);

        if outer_w - y_w - right_pad < min_plot_w {
            y_w = (outer_w - right_pad - min_plot_w).max(0.0);
        }
        if outer_w - y_w - right_pad < min_plot_w {
            right_pad = (outer_w - y_w - min_plot_w).max(0.0);
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

    fn layout_3d_panel(
        &self,
        outer: Rect,
        title: Option<&str>,
        subtitle: Option<&str>,
        scale: f32,
    ) -> PanelLayout {
        let scale = scale.max(0.75);
        let has_title = Self::has_visible_text(title);
        let has_subtitle = Self::has_visible_text(subtitle);
        let title_h = if has_title || has_subtitle {
            let base = if has_title && has_subtitle {
                44.0
            } else {
                28.0
            };
            let extra = Self::extra_title_lines(title) as f32 * 19.0;
            ((base + extra) * scale).min(outer.height().max(1.0) * 0.22)
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

    fn positioned_axes_outer_rect(
        outer: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
    ) -> Rect {
        let Some((position, units)) = plot_renderer.overlay_position_for_axes(axes_index) else {
            return outer;
        };
        Self::positioned_axes_rect(outer, position, &units)
    }

    fn positioned_axes_rect(outer: Rect, position: [f64; 4], units: &str) -> Rect {
        // A newly created browser surface can be smaller than the overlay's
        // margins. Keep the placement rectangle nonempty before clamping.
        let outer = Rect::from_min_size(
            outer.min,
            egui::vec2(outer.width().max(1.0), outer.height().max(1.0)),
        );
        let (left, bottom, width, height) = if units.eq_ignore_ascii_case("normalized") {
            (
                position[0] as f32 * outer.width(),
                position[1] as f32 * outer.height(),
                position[2] as f32 * outer.width(),
                position[3] as f32 * outer.height(),
            )
        } else {
            (
                position[0] as f32,
                position[1] as f32,
                position[2] as f32,
                position[3] as f32,
            )
        };
        let width = width.max(1.0).min(outer.width().max(1.0));
        let height = height.max(1.0).min(outer.height().max(1.0));
        let x = (outer.min.x + left).clamp(outer.min.x, outer.max.x - 1.0);
        let y = (outer.max.y - bottom - height).clamp(outer.min.y, outer.max.y - 1.0);
        Rect::from_min_size(
            egui::pos2(x, y),
            egui::vec2(width.min(outer.max.x - x), height.min(outer.max.y - y)),
        )
    }

    /// The vertical span of a 3-D axes' colorbar. A 3-D plot area reaches the panel edges, so
    /// the bar keeps a tenth of the height clear above and below, as MATLAB's 3-D axes position
    /// does, and its end tick labels stay inside the figure.
    fn colorbar_span_3d(plot_rect: Rect) -> Rect {
        let inset = plot_rect.height() * 0.1;
        Rect::from_min_max(
            egui::pos2(plot_rect.min.x, plot_rect.min.y + inset),
            egui::pos2(plot_rect.max.x, plot_rect.max.y - inset),
        )
    }

    /// Width reserved to the right of an axes for its colorbar and tick labels.
    fn colorbar_band_width(scale: f32) -> f32 {
        68.0 * scale.max(0.75)
    }

    /// Panel layouts for a subplot grid with the plot boxes aligned as MATLAB's subplot aligns
    /// them: every 2-D axes in a grid column shares its left and right edges, and every one in
    /// a grid row its top and bottom, so a missing label or title does not shift one box
    /// against its neighbours. Axes with an explicit Position and 3-D axes keep their own.
    fn grid_panel_layouts(
        &self,
        cells: &[Rect],
        cols: usize,
        plot_renderer: &PlotRenderer,
        scale: f32,
    ) -> Vec<PanelLayout> {
        let outers: Vec<Rect> = cells
            .iter()
            .enumerate()
            .map(|(i, cell)| Self::positioned_axes_outer_rect(*cell, plot_renderer, i))
            .collect();
        let mut layouts: Vec<PanelLayout> = outers
            .iter()
            .enumerate()
            .map(|(i, outer)| self.panel_layout_for_axes(*outer, plot_renderer, i, scale))
            .collect();
        if cols == 0 || cells.len() < 2 {
            return layouts;
        }
        let aligned = |i: usize| {
            plot_renderer.overlay_position_for_axes(i).is_none()
                && !Self::axes_is_3d(plot_renderer, i)
        };
        let rows = cells.len().div_ceil(cols);
        let (mut left, mut right) = (vec![0.0_f32; cols], vec![0.0_f32; cols]);
        let (mut top, mut bottom) = (vec![0.0_f32; rows], vec![0.0_f32; rows]);
        for (i, layout) in layouts.iter().enumerate() {
            if !aligned(i) {
                continue;
            }
            let (r, c) = (i / cols, i % cols);
            let (cell, plot) = (outers[i], layout.plot_rect);
            left[c] = left[c].max(plot.min.x - cell.min.x);
            right[c] = right[c].max(cell.max.x - plot.max.x);
            top[r] = top[r].max(plot.min.y - cell.min.y);
            bottom[r] = bottom[r].max(cell.max.y - plot.max.y);
        }
        for (i, layout) in layouts.iter_mut().enumerate() {
            if !aligned(i) {
                continue;
            }
            let (r, c) = (i / cols, i % cols);
            let cell = outers[i];
            let min_x = cell.min.x + left[c];
            let min_y = cell.min.y + top[r];
            let plot_rect = Rect::from_min_max(
                egui::pos2(min_x, min_y),
                egui::pos2(
                    (cell.max.x - right[c]).max(min_x + 1.0),
                    (cell.max.y - bottom[r]).max(min_y + 1.0),
                ),
            );
            *layout = PanelLayout {
                plot_rect,
                frame_rect: plot_rect,
                title_rect: Rect::from_min_max(cell.min, egui::pos2(cell.max.x, plot_rect.min.y)),
                x_label_rect: Rect::from_min_max(egui::pos2(cell.min.x, plot_rect.max.y), cell.max),
                y_label_rect: Rect::from_min_max(
                    egui::pos2(cell.min.x, plot_rect.min.y),
                    egui::pos2(plot_rect.min.x, plot_rect.max.y),
                ),
            };
        }
        layouts
    }

    fn panel_layout_for_axes(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) -> PanelLayout {
        // A colorbar sits outside the axes on the right ('eastoutside'), as in MATLAB.
        let outer = if plot_renderer.overlay_colorbar_enabled_for_axes(axes_index) {
            let band = Self::colorbar_band_width(scale).min(outer.width() * 0.3);
            Rect::from_min_max(outer.min, egui::pos2(outer.max.x - band, outer.max.y))
        } else {
            outer
        };
        if Self::axes_is_3d(plot_renderer, axes_index) {
            self.layout_3d_panel(
                outer,
                plot_renderer
                    .overlay_title_for_axes(axes_index)
                    .map(|s| s.as_str()),
                plot_renderer
                    .overlay_subtitle_for_axes(axes_index)
                    .map(|s| s.as_str()),
                scale,
            )
        } else {
            let layout = self.layout_2d_panel_for_axes(outer, plot_renderer, axes_index, scale);
            // With a fixed data aspect (`axis equal`), shrink the plot box to that aspect,
            // centered in the space it had, and lay the labels out around the smaller box.
            let Some(aspect) = plot_renderer.overlay_plot_box_aspect_for_axes(axes_index) else {
                return layout;
            };
            let plot = layout.plot_rect;
            if plot.width() <= 1.0 || plot.height() <= 1.0 {
                return layout;
            }
            let current = (plot.width() / plot.height()) as f64;
            let (dx, dy) = if current > aspect {
                (plot.width() - (plot.height() as f64 * aspect) as f32, 0.0)
            } else {
                (0.0, plot.height() - (plot.width() as f64 / aspect) as f32)
            };
            if dx < 0.5 && dy < 0.5 {
                return layout;
            }
            let shrunk = Rect::from_min_max(
                egui::pos2(outer.min.x + dx * 0.5, outer.min.y + dy * 0.5),
                egui::pos2(outer.max.x - dx * 0.5, outer.max.y - dy * 0.5),
            );
            self.layout_2d_panel_for_axes(shrunk, plot_renderer, axes_index, scale)
        }
    }

    fn layout_2d_panel_for_axes(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) -> PanelLayout {
        self.layout_2d_panel(
            outer,
            plot_renderer,
            axes_index,
            plot_renderer
                .overlay_title_for_axes(axes_index)
                .map(|s| s.as_str()),
            plot_renderer
                .overlay_subtitle_for_axes(axes_index)
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

    fn reserve_sg_title_band(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        scale: f32,
    ) -> (Rect, Option<Rect>) {
        let Some(title) = plot_renderer.overlay_sg_title().map(String::as_str) else {
            return (outer, None);
        };
        if !Self::has_visible_text(Some(title)) {
            return (outer, None);
        }

        let scale = scale.max(0.75);
        let band_h = (30.0 * scale).min(outer.height().max(1.0) * 0.14);
        let gap = 4.0 * scale;
        let title_max_y = (outer.min.y + band_h).min(outer.max.y);
        let content_min_y = (title_max_y + gap).min(outer.max.y);
        (
            Rect::from_min_max(egui::pos2(outer.min.x, content_min_y), outer.max),
            Some(Rect::from_min_max(
                outer.min,
                egui::pos2(outer.max.x, title_max_y),
            )),
        )
    }

    fn compute_subplot_plot_rects_impl(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        font_scale: f32,
        show_title: bool,
    ) -> Vec<Rect> {
        let plot_area = Self::outer_plot_area_for_axes(outer, plot_renderer);
        let (plot_area, _) = if show_title {
            self.reserve_sg_title_band(plot_area, plot_renderer, font_scale)
        } else {
            (plot_area, None)
        };
        let (rows, cols) = plot_renderer.figure_axes_grid();
        if rows * cols <= 1 {
            (0..plot_renderer.figure_axes_count().max(1))
                .map(|axes_index| {
                    let axes_outer =
                        Self::positioned_axes_outer_rect(plot_area, plot_renderer, axes_index);
                    self.panel_layout_for_axes(axes_outer, plot_renderer, axes_index, font_scale)
                        .plot_rect
                })
                .collect()
        } else {
            let rects = self.compute_subplot_rects(
                plot_area,
                rows,
                cols,
                Self::SUBPLOT_GAP_POINTS,
                Self::SUBPLOT_GAP_POINTS,
            );
            self.grid_panel_layouts(&rects, cols, plot_renderer, font_scale)
                .into_iter()
                .map(|layout| layout.plot_rect)
                .collect()
        }
    }

    pub fn compute_subplot_plot_rects(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        font_scale: f32,
    ) -> Vec<Rect> {
        self.compute_subplot_plot_rects_impl(outer, plot_renderer, font_scale, true)
    }

    /// Like [`compute_subplot_plot_rects`] but lets the caller control whether the
    /// super-title band is reserved. Pass `show_title: false` when the overlay is
    /// rendered without titles so that GPU viewports and hit-rects stay aligned.
    pub fn compute_subplot_plot_rects_explicit(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        font_scale: f32,
        show_title: bool,
    ) -> Vec<Rect> {
        self.compute_subplot_plot_rects_impl(outer, plot_renderer, font_scale, show_title)
    }

    pub fn snap_rect_to_pixels(rect: Rect, pixels_per_point: f32) -> Rect {
        let ppp = pixels_per_point.max(0.5);
        let min_x = (rect.min.x * ppp).round() / ppp;
        let min_y = (rect.min.y * ppp).round() / ppp;
        let width = (rect.width() * ppp).round().max(1.0) / ppp;
        let height = (rect.height() * ppp).round().max(1.0) / ppp;
        Rect::from_min_size(egui::pos2(min_x, min_y), egui::vec2(width, height))
    }

    /// Whether an explicit tick lies on the axis span. Limits can come from f32 data bounds, so a
    /// tick exactly at a limit (e.g. `xticks` at the first and last sample) may differ from it in
    /// the last bits; a small relative tolerance keeps those end ticks.
    fn tick_within_limits(value: f64, min: f64, max: f64) -> bool {
        let tolerance = ((max - min).abs().max(min.abs()).max(max.abs())) * 1.0e-6;
        value >= min - tolerance && value <= max + tolerance
    }

    /// Like `tick_within_limits`, with the tolerance taken in decades for a log axis.
    fn tick_within_log_limits(value: f64, min: f64, max: f64) -> bool {
        if !(value > 0.0 && min > 0.0 && max > 0.0) {
            return false;
        }
        let (v, lo, hi) = (value.log10(), min.log10(), max.log10());
        let tolerance = (hi - lo).abs().max(1.0) * 1.0e-6;
        v >= lo - tolerance && v <= hi + tolerance
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
        let stroke = Stroke::new(1.5_f32, self.theme_axis_color());
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
        self.compute_subplot_plot_rects_impl(outer, plot_renderer, font_scale, true)
            .into_iter()
            .map(|rect| Self::snap_rect_to_pixels(rect, pixels_per_point))
            .collect()
    }

    /// Like [`compute_subplot_plot_rects_snapped`] but lets the caller control whether
    /// the super-title band is reserved. Pass `show_title: false` when the overlay is
    /// rendered without titles so that GPU viewports stay aligned with the drawn area.
    pub fn compute_subplot_plot_rects_snapped_explicit(
        &self,
        outer: Rect,
        plot_renderer: &PlotRenderer,
        font_scale: f32,
        pixels_per_point: f32,
        show_title: bool,
    ) -> Vec<Rect> {
        self.compute_subplot_plot_rects_impl(outer, plot_renderer, font_scale, show_title)
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

    /// Render plot actions without reserving space beside the plot.
    fn render_toolbar(&mut self, ui: &mut egui::Ui, show_grid: bool, show_legend: bool) {
        egui::TopBottomPanel::top("plot_toolbar")
            .frame(transparent_frame())
            .show_inside(ui, |ui| {
                let padded = ui.max_rect().shrink2(egui::vec2(12.0, 6.0));
                self.toolbar_rect = Some(padded);
                let add_toolbar = |ui: &mut egui::Ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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
                        let mut grid = show_grid;
                        if ui.toggle_value(&mut grid, "Grid").changed() {
                            self.want_toggle_grid = Some(grid);
                        }
                        let mut legend = show_legend;
                        if ui.toggle_value(&mut legend, "Legend").changed() {
                            self.want_toggle_legend = Some(legend);
                        }
                    });
                };
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(padded), add_toolbar);
            });
    }

    /// Render the complete overlay UI
    pub fn render(
        &mut self,
        ctx: &Context,
        plot_renderer: &mut PlotRenderer,
        config: &OverlayConfig,
        metrics: OverlayMetrics,
    ) -> FrameInfo {
        let mut consumed_input = false;
        let mut plot_area = None;

        // Render main plot area
        let central_response = egui::CentralPanel::default()
            .frame(transparent_frame()) // Transparent frame
            .show(ctx, |ui| {
                // Toolbar (top-right)
                if config.show_toolbar {
                    self.render_toolbar(
                        ui,
                        plot_renderer.overlay_show_grid(),
                        plot_renderer.overlay_show_legend(),
                    );
                } else {
                    self.toolbar_rect = None;
                }
                plot_area = Some(self.render_plot_area(ui, plot_renderer, config));
            });

        if config.show_toolbar {
            consumed_input |= central_response.response.hovered();
        }

        // Store plot area for next frame
        self.plot_area = plot_area;
        crate::collaboration::render_presence(
            ctx,
            &plot_renderer.presence().revision,
            plot_renderer.presence(),
            &self.presence_projections(plot_renderer),
        );

        FrameInfo {
            plot_area,
            consumed_input,
            metrics,
        }
    }

    pub fn presence_projections(
        &self,
        renderer: &PlotRenderer,
    ) -> Vec<crate::collaboration::PresenceProjection> {
        self.axes_plot_rects
            .iter()
            .enumerate()
            .filter_map(|(index, rect)| {
                let mut camera = renderer.axes_camera(index)?.clone();
                // The rendered viewport uses this axes rectangle, not the outer
                // figure's aspect ratio. Rebuild the projection for the same
                // viewport before translating collaborators' data coordinates.
                camera.update_aspect_ratio(rect.width() / rect.height());
                let is_3d = renderer.has_geometry_scene() || renderer.axes_has_3d_content(index);
                Some(crate::collaboration::PresenceProjection {
                    axes: format!("axes:{index}"),
                    rect: [rect.min.x, rect.min.y, rect.width(), rect.height()],
                    view_projection: camera.view_proj_matrix(),
                    data_bounds: if is_3d {
                        None
                    } else {
                        renderer
                            .view_bounds_for_axes(index)
                            .map(|(left, right, bottom, top)| [left, right, bottom, top])
                    },
                    is_3d,
                    x_log: renderer.overlay_x_log_for_axes(index),
                    y_log: renderer.overlay_y_log_for_axes(index),
                })
            })
            .collect()
    }

    /// Render the main plot area with grid, axes, and annotations
    fn render_plot_area(
        &mut self,
        ui: &mut egui::Ui,
        plot_renderer: &mut PlotRenderer,
        config: &OverlayConfig,
    ) -> Rect {
        let available_rect = ui.available_rect_before_wrap();
        let mut rendered_axes_rects: Vec<Rect> = Vec::new();

        let (rows, cols) = plot_renderer.figure_axes_grid();
        let plot_rect = Self::outer_plot_area_for_axes(available_rect, plot_renderer);
        let (plot_rect, sg_title_rect) = if config.show_title {
            self.reserve_sg_title_band(plot_rect, plot_renderer, config.font_scale)
        } else {
            (plot_rect, None)
        };

        // Use full available rectangular plot area (do not force square);
        // camera fitting and axis_equal settings will control aspect.
        let plot_area_rect = plot_rect;

        if let Some(title_rect) = sg_title_rect {
            if let Some(title) = plot_renderer.overlay_sg_title() {
                self.draw_title_in_rect(
                    ui,
                    title_rect,
                    title,
                    Some(plot_renderer.overlay_sg_title_style()),
                    config.font_scale,
                );
            }
        }

        let axes_count = plot_renderer.figure_axes_count().max(1);
        if rows * cols > 1 || axes_count > 1 {
            let rects = if rows * cols > 1 {
                self.compute_subplot_rects(
                    plot_area_rect,
                    rows,
                    cols,
                    Self::SUBPLOT_GAP_POINTS,
                    Self::SUBPLOT_GAP_POINTS,
                )
            } else {
                vec![plot_area_rect; axes_count]
            };
            let grid_cols = if rows * cols > 1 { cols } else { 0 };
            let mut panel_layouts = self
                .grid_panel_layouts(&rects, grid_cols, plot_renderer, config.font_scale)
                .into_iter();
            for (i, cell_rect) in rects.iter().enumerate() {
                let cell_rect = Self::positioned_axes_outer_rect(*cell_rect, plot_renderer, i);
                let panel_layout = panel_layouts
                    .next()
                    .expect("one panel layout per subplot cell");
                let r =
                    Self::snap_rect_to_pixels(panel_layout.plot_rect, ui.ctx().pixels_per_point());
                plot_renderer.fit_pending_figure_camera_for_viewport(
                    i,
                    (
                        (r.width() * ui.ctx().pixels_per_point()).round().max(1.0) as u32,
                        (r.height() * ui.ctx().pixels_per_point()).round().max(1.0) as u32,
                    ),
                );
                let cam = plot_renderer
                    .axes_camera(i)
                    .unwrap_or_else(|| plot_renderer.camera());
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
                        self.draw_title_block_in_rect(
                            ui,
                            panel_layout.title_rect,
                            plot_renderer.overlay_title_for_axes(i).map(String::as_str),
                            plot_renderer.overlay_title_style_for_axes(i),
                            plot_renderer
                                .overlay_subtitle_for_axes(i)
                                .map(String::as_str),
                            plot_renderer.overlay_subtitle_style_for_axes(i),
                            config.font_scale,
                        );
                    }
                    // `axis off` hides the orientation triad, axis ticks and their labels.
                    let axes_visible = Self::axes_visible(plot_renderer, i);
                    if axes_visible {
                        self.draw_3d_orientation_gizmo(ui, r, plot_renderer, i, config.font_scale);
                    }
                    let show_world_grid_overlay = plot_renderer.geometry_overlay().is_none()
                        || plot_renderer.overlay_show_grid_for_axes(i);
                    if show_world_grid_overlay {
                        if axes_visible {
                            self.draw_3d_origin_axis_ticks(
                                ui,
                                r,
                                plot_renderer,
                                i,
                                config.font_scale,
                            );
                        }
                        self.draw_projected_world_texts(ui, r, plot_renderer, i, config.font_scale);
                    }
                    for (label, pos) in plot_renderer.pie_labels_for_axes(i) {
                        self.draw_pie_label(
                            ui,
                            r,
                            plot_renderer,
                            i,
                            &label,
                            pos,
                            config.font_scale,
                        );
                    }
                    if plot_renderer.overlay_show_legend_for_axes(i) {
                        self.draw_axes_legend(ui, r, plot_renderer, i, config.font_scale);
                    }
                    self.draw_colorbar(
                        ui,
                        Self::colorbar_span_3d(r),
                        plot_renderer,
                        i,
                        config.font_scale,
                    );
                    continue;
                }
                // `axis off` hides the box, grid, ticks and axis labels; the title stays.
                let axes_visible = Self::axes_visible(plot_renderer, i);
                // Frame (2D only; 3D uses the axes cube instead)
                if axes_visible && plot_renderer.overlay_show_box_for_axes(i) {
                    self.draw_plot_box_mask(ui, r);
                    self.draw_2d_border(ui, frame_rect);
                }

                // Grid (2D)
                if axes_visible
                    && config.show_grid
                    && (plot_renderer.overlay_show_grid_for_axes(i)
                        || plot_renderer.overlay_show_minor_grid_for_axes(i))
                {
                    let b = plot_renderer.view_bounds_for_axes(i);
                    self.draw_grid(ui, r, plot_renderer, b, Some(i));
                }
                // Axes (2D)
                if config.show_axes && axes_visible {
                    let b = plot_renderer.view_bounds_for_axes(i);
                    self.draw_axes(ui, r, plot_renderer, config, b, Some(i));
                    for overlay_axes in 0..plot_renderer.figure_axes_count() {
                        if plot_renderer.overlay_parent_for_axes(overlay_axes) == Some(i) {
                            self.draw_axes(
                                ui,
                                r,
                                plot_renderer,
                                config,
                                plot_renderer.view_bounds_for_axes(overlay_axes),
                                Some(overlay_axes),
                            );
                        }
                    }
                }

                if config.show_title {
                    self.draw_title_block_in_rect(
                        ui,
                        panel_layout.title_rect,
                        plot_renderer.overlay_title_for_axes(i).map(String::as_str),
                        plot_renderer.overlay_title_style_for_axes(i),
                        plot_renderer
                            .overlay_subtitle_for_axes(i)
                            .map(String::as_str),
                        plot_renderer.overlay_subtitle_style_for_axes(i),
                        config.font_scale,
                    );
                }
                if !matches!(
                    cam.projection,
                    crate::core::camera::ProjectionType::Perspective { .. }
                ) {
                    if let Some(x_label) = plot_renderer
                        .overlay_x_label_for_axes(i)
                        .filter(|_| axes_visible)
                    {
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
                    if let Some(y_label) = plot_renderer
                        .overlay_y_label_for_axes(i)
                        .filter(|_| axes_visible)
                    {
                        self.draw_y_label_in_rect(
                            ui,
                            panel_layout.y_label_rect,
                            y_label,
                            config.font_scale,
                        );
                    }
                }
                self.draw_cell_labels(ui, r, plot_renderer, i, config.font_scale);
                self.draw_reference_line_labels(ui, r, plot_renderer, i, config.font_scale);
                self.draw_colorbar(ui, frame_rect, plot_renderer, i, config.font_scale);
                self.draw_projected_world_texts(ui, r, plot_renderer, i, config.font_scale);
                for (label, pos) in plot_renderer.pie_labels_for_axes(i) {
                    self.draw_pie_label(ui, r, plot_renderer, i, &label, pos, config.font_scale);
                }
                if plot_renderer.overlay_show_legend_for_axes(i) {
                    self.draw_axes_legend(ui, r, plot_renderer, i, config.font_scale);
                }
            }
        } else {
            let panel_layout =
                self.panel_layout_for_axes(plot_area_rect, plot_renderer, 0, config.font_scale);
            let centered_plot_rect =
                Self::snap_rect_to_pixels(panel_layout.plot_rect, ui.ctx().pixels_per_point());
            plot_renderer.fit_pending_figure_camera_for_viewport(
                0,
                (
                    (centered_plot_rect.width() * ui.ctx().pixels_per_point())
                        .round()
                        .max(1.0) as u32,
                    (centered_plot_rect.height() * ui.ctx().pixels_per_point())
                        .round()
                        .max(1.0) as u32,
                ),
            );
            let cam = plot_renderer.camera();
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
                self.draw_title_block_in_rect(
                    ui,
                    panel_layout.title_rect,
                    plot_renderer
                        .overlay_title_for_axes(0)
                        .map(String::as_str)
                        .or(config.title.as_deref()),
                    plot_renderer.overlay_title_style_for_axes(0),
                    plot_renderer
                        .overlay_subtitle_for_axes(0)
                        .map(String::as_str),
                    plot_renderer.overlay_subtitle_style_for_axes(0),
                    config.font_scale,
                );
            }
            if matches!(
                cam.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            ) {
                // `axis off` hides the orientation triad, axis ticks and their labels.
                let axes_visible = Self::axes_visible(plot_renderer, 0);
                if axes_visible {
                    self.draw_3d_orientation_gizmo(
                        ui,
                        centered_plot_rect,
                        plot_renderer,
                        0,
                        config.font_scale,
                    );
                }
                let show_world_grid_overlay = plot_renderer.geometry_overlay().is_none()
                    || plot_renderer.overlay_show_grid_for_axes(0);
                if show_world_grid_overlay {
                    if axes_visible {
                        self.draw_3d_origin_axis_ticks(
                            ui,
                            centered_plot_rect,
                            plot_renderer,
                            0,
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
                self.draw_colorbar(
                    ui,
                    Self::colorbar_span_3d(centered_plot_rect),
                    plot_renderer,
                    0,
                    config.font_scale,
                );
            } else {
                // `axis off` hides the box, grid, ticks and axis labels; the title stays.
                let axes_visible = Self::axes_visible(plot_renderer, 0);
                // Draw plot frame (2D only; 3D uses the axes cube instead)
                if axes_visible && plot_renderer.overlay_show_box() {
                    self.draw_plot_box_mask(ui, centered_plot_rect);
                    self.draw_2d_border(ui, centered_frame_rect);
                }
                // Draw grid if enabled
                if axes_visible
                    && config.show_grid
                    && (plot_renderer.overlay_show_grid()
                        || plot_renderer.overlay_show_minor_grid())
                {
                    self.draw_grid(ui, centered_plot_rect, plot_renderer, None, None);
                }

                // Draw axes if enabled
                if config.show_axes && axes_visible {
                    self.draw_axes(ui, centered_plot_rect, plot_renderer, config, None, Some(0));
                    for overlay_axes in 1..plot_renderer.figure_axes_count() {
                        if plot_renderer.overlay_parent_for_axes(overlay_axes) == Some(0) {
                            self.draw_axes(
                                ui,
                                centered_plot_rect,
                                plot_renderer,
                                config,
                                plot_renderer.view_bounds_for_axes(overlay_axes),
                                Some(overlay_axes),
                            );
                        }
                    }
                }
                if let Some(x_label) = plot_renderer
                    .overlay_x_label_for_axes(0)
                    .or(config.x_label.as_ref())
                    .filter(|_| axes_visible)
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
                    .filter(|_| axes_visible)
                {
                    self.draw_y_label_in_rect(
                        ui,
                        panel_layout.y_label_rect,
                        y_label,
                        config.font_scale,
                    );
                }
                self.draw_cell_labels(ui, centered_plot_rect, plot_renderer, 0, config.font_scale);
                self.draw_reference_line_labels(
                    ui,
                    centered_plot_rect,
                    plot_renderer,
                    0,
                    config.font_scale,
                );
                self.draw_colorbar(ui, centered_frame_rect, plot_renderer, 0, config.font_scale);
                self.draw_projected_world_texts(
                    ui,
                    centered_plot_rect,
                    plot_renderer,
                    0,
                    config.font_scale,
                );
            }
        }
        let centered_plot_rect = if rows * cols <= 1 && axes_count <= 1 {
            self.panel_layout_for_axes(plot_area_rect, plot_renderer, 0, config.font_scale)
                .plot_rect
        } else {
            plot_area_rect
        };
        for (label, pos) in if rows * cols <= 1 && axes_count <= 1 {
            plot_renderer.active_axes_pie_labels()
        } else {
            Vec::new()
        } {
            self.draw_pie_label(
                ui,
                centered_plot_rect,
                plot_renderer,
                0,
                &label,
                pos,
                config.font_scale,
            );
        }

        // Draw legend if enabled and entries available
        if rows * cols <= 1 && axes_count <= 1 && plot_renderer.overlay_show_legend_for_axes(0) {
            // Per-axes entries include each pie slice.
            self.draw_axes_legend(ui, centered_plot_rect, plot_renderer, 0, config.font_scale);
        }

        if let Some(legend) = plot_renderer
            .geometry_overlay()
            .and_then(|overlay| overlay.scalar_legend.as_ref())
        {
            self.draw_geometry_scalar_legend(ui, centered_plot_rect, legend, config.font_scale);
        }

        // Physical annotations belong to the model, even when the host supplies
        // its own CAD toolbar (browser and embedded native surfaces).
        let annotations = plot_renderer
            .geometry_screen_annotations([centered_plot_rect.width(), centered_plot_rect.height()]);
        self.cad_overlay.render_annotation_labels(
            ui.ctx(),
            centered_plot_rect,
            &annotations,
            config.font_scale,
        );

        // Result scenes have their own toolbar and mesh-info popover. The CAD
        // strip duplicates those controls and overlaps the scalar legend.
        if let Some(overlay) = plot_renderer
            .geometry_overlay()
            .filter(|overlay| config.show_cad_overlay && overlay.scalar_legend.is_none())
        {
            let actions = self.cad_overlay.render(CadOverlayRenderInput {
                ctx: ui.ctx(),
                plot_rect: centered_plot_rect,
                overlay,
                grid_enabled: plot_renderer.overlay_show_grid(),
                xray_enabled: plot_renderer.geometry_xray_enabled(),
                owner_visible: &|owner_id| plot_renderer.geometry_owner_visible(owner_id),
                font_scale: config.font_scale,
            });
            self.cad_actions.merge(actions);
        } else {
            self.cad_overlay.clear_interaction();
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
            let (grid_color_major, grid_color_minor) = self.themed_grid_colors();

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
            let show_major_grid = axes_index
                .map(|idx| plot_renderer.overlay_show_grid_for_axes(idx))
                .unwrap_or_else(|| plot_renderer.overlay_show_grid());
            let show_minor_grid = axes_index
                .map(|idx| plot_renderer.overlay_show_minor_grid_for_axes(idx))
                .unwrap_or_else(|| plot_renderer.overlay_show_minor_grid());
            let explicit_x_ticks =
                axes_index.and_then(|idx| plot_renderer.overlay_x_ticks_for_axes(idx));
            let explicit_y_ticks =
                axes_index.and_then(|idx| plot_renderer.overlay_y_ticks_for_axes(idx));

            let x_ticks = if let Some(ticks) = explicit_x_ticks.as_ref() {
                ticks.clone()
            } else if x_log {
                Vec::new()
            } else {
                plot_utils::generate_major_ticks(x_min, x_max)
            };
            let y_ticks = if let Some(ticks) = explicit_y_ticks.as_ref() {
                ticks.clone()
            } else if y_log {
                Vec::new()
            } else {
                plot_utils::generate_major_ticks(y_min, y_max)
            };

            // Draw vertical grid lines (linear vs log)
            if x_log && explicit_x_ticks.is_none() {
                // Decades within [x_min, x_max]
                let start_decade = x_min.log10().floor() as i32;
                let end_decade = x_max.log10().ceil() as i32;
                for d in start_decade..=end_decade {
                    let decade = 10f64.powi(d);
                    for mantissa in 1..=9 {
                        let x_val = decade * mantissa as f64;
                        if x_val < x_min || x_val > x_max {
                            continue;
                        }
                        let is_major = mantissa == 1;
                        if (is_major && !show_major_grid) || (!is_major && !show_minor_grid) {
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
                            Stroke::new(
                                if is_major { 0.8_f32 } else { 0.6_f32 },
                                if is_major {
                                    grid_color_major
                                } else {
                                    grid_color_minor
                                },
                            ),
                        );
                    }
                }
            } else {
                if show_minor_grid && !x_log {
                    for pair in x_ticks.windows(2) {
                        let step = (pair[1] - pair[0]) / 5.0;
                        if !step.is_finite() || step <= 0.0 {
                            continue;
                        }
                        for n in 1..5 {
                            let x_val = pair[0] + step * n as f64;
                            if x_val < x_min || x_val > x_max {
                                continue;
                            }
                            let x_screen = plot_rect.min.x
                                + ((x_val - x_min) / x_range) as f32 * plot_rect.width();
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
                                Stroke::new(0.6_f32, grid_color_minor),
                            );
                        }
                    }
                }
                if show_major_grid {
                    for x_val in x_ticks {
                        if x_val < x_min || x_val > x_max {
                            continue;
                        }
                        let x_screen = if x_log {
                            if x_val <= 0.0 || x_min <= 0.0 || x_max <= 0.0 {
                                continue;
                            }
                            plot_rect.min.x
                                + ((x_val.log10() - x_min.log10())
                                    / (x_max.log10() - x_min.log10()))
                                    as f32
                                    * plot_rect.width()
                        } else {
                            plot_rect.min.x + ((x_val - x_min) / x_range) as f32 * plot_rect.width()
                        };
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
                            Stroke::new(0.8_f32, grid_color_major),
                        );
                    }
                }
            }

            // Draw horizontal grid lines (linear vs log)
            if y_log && explicit_y_ticks.is_none() {
                let start_decade = y_min.log10().floor() as i32;
                let end_decade = y_max.log10().ceil() as i32;
                for d in start_decade..=end_decade {
                    let decade = 10f64.powi(d);
                    for mantissa in 1..=9 {
                        let y_val = decade * mantissa as f64;
                        if y_val < y_min || y_val > y_max {
                            continue;
                        }
                        let is_major = mantissa == 1;
                        if (is_major && !show_major_grid) || (!is_major && !show_minor_grid) {
                            continue;
                        }
                        let y_screen = plot_rect.max.y
                            - ((y_val.log10() - y_min.log10()) / (y_max.log10() - y_min.log10()))
                                as f32
                                * plot_rect.height();
                        let y_screen = plot_renderer
                            .overlay_y_direction_for_axes(axes_index.unwrap_or(0))
                            .orient(y_screen, plot_rect.min.y, plot_rect.max.y);
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
                            Stroke::new(
                                if is_major { 0.8_f32 } else { 0.6_f32 },
                                if is_major {
                                    grid_color_major
                                } else {
                                    grid_color_minor
                                },
                            ),
                        );
                    }
                }
            } else {
                if show_minor_grid && !y_log {
                    for pair in y_ticks.windows(2) {
                        let step = (pair[1] - pair[0]) / 5.0;
                        if !step.is_finite() || step <= 0.0 {
                            continue;
                        }
                        for n in 1..5 {
                            let y_val = pair[0] + step * n as f64;
                            if y_val < y_min || y_val > y_max {
                                continue;
                            }
                            let y_screen = plot_rect.max.y
                                - ((y_val - y_min) / y_range) as f32 * plot_rect.height();
                            let y_screen = plot_renderer
                                .overlay_y_direction_for_axes(axes_index.unwrap_or(0))
                                .orient(y_screen, plot_rect.min.y, plot_rect.max.y);
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
                                Stroke::new(0.6_f32, grid_color_minor),
                            );
                        }
                    }
                }
                if show_major_grid {
                    for y_val in y_ticks {
                        if y_val < y_min || y_val > y_max {
                            continue;
                        }
                        let y_screen = if y_log {
                            if y_val <= 0.0 || y_min <= 0.0 || y_max <= 0.0 {
                                continue;
                            }
                            plot_rect.max.y
                                - ((y_val.log10() - y_min.log10())
                                    / (y_max.log10() - y_min.log10()))
                                    as f32
                                    * plot_rect.height()
                        } else {
                            plot_rect.max.y
                                - ((y_val - y_min) / y_range) as f32 * plot_rect.height()
                        };
                        let y_screen = plot_renderer
                            .overlay_y_direction_for_axes(axes_index.unwrap_or(0))
                            .orient(y_screen, plot_rect.min.y, plot_rect.max.y);
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
                            Stroke::new(0.8_f32, grid_color_major),
                        );
                    }
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
            let tick_font_size = Self::axes_tick_font_size(plot_renderer, axes_index, scale);
            let tick_font = FontId::proportional(tick_font_size);
            let axis_color = self.theme_axis_color();
            let label_color = self.theme_text_color();
            let border_left = plot_rect.min.x;
            let border_bottom = plot_rect.max.y;
            let y_axis_on_right = axes_index
                .map(|idx| plot_renderer.overlay_y_axis_location_for_axes(idx) == "right")
                .unwrap_or(false);
            let draw_x_axis = !y_axis_on_right;
            let y_axis_x = if y_axis_on_right {
                plot_rect.max.x
            } else {
                border_left
            };
            let y_tick_outer_x = if y_axis_on_right {
                y_axis_x + tick_length
            } else {
                y_axis_x - tick_length
            };
            let y_label_x = if y_axis_on_right {
                y_axis_x + label_offset
            } else {
                y_axis_x - label_offset
            };

            let x_log = axes_index
                .map(|idx| plot_renderer.overlay_x_log_for_axes(idx))
                .unwrap_or_else(|| plot_renderer.overlay_x_log());
            let y_log = axes_index
                .map(|idx| plot_renderer.overlay_y_log_for_axes(idx))
                .unwrap_or_else(|| plot_renderer.overlay_y_log());
            let explicit_x_ticks =
                axes_index.and_then(|idx| plot_renderer.overlay_x_ticks_for_axes(idx));
            let explicit_y_ticks =
                axes_index.and_then(|idx| plot_renderer.overlay_y_ticks_for_axes(idx));
            let explicit_x_labels =
                axes_index.and_then(|idx| plot_renderer.overlay_x_tick_labels_for_axes(idx));
            let explicit_y_labels =
                axes_index.and_then(|idx| plot_renderer.overlay_y_tick_labels_for_axes(idx));
            let explicit_x_format =
                axes_index.and_then(|idx| plot_renderer.overlay_x_tick_format_for_axes(idx));
            let explicit_y_format =
                axes_index.and_then(|idx| plot_renderer.overlay_y_tick_format_for_axes(idx));
            let x_tick_formatter =
                plot_utils::TickLabelFormatter::new(explicit_x_format.as_deref());
            let y_tick_formatter =
                plot_utils::TickLabelFormatter::new(explicit_y_format.as_deref());
            let x_tick_angle = axes_index
                .map(|idx| plot_renderer.overlay_x_tick_label_rotation_for_axes(idx))
                .unwrap_or(0.0);
            let y_tick_angle = axes_index
                .map(|idx| plot_renderer.overlay_y_tick_label_rotation_for_axes(idx))
                .unwrap_or(0.0);

            // Histogram numeric tick support and categorical axis support
            let (mut cat_x, mut cat_y) = (false, false);
            let mut custom_hist_x = false;
            if draw_x_axis && explicit_x_ticks.is_none() {
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
            }
            if draw_x_axis {
                if let Some(ticks) = explicit_x_ticks.as_ref() {
                    cat_x = true;
                    for (tick_idx, x_val) in ticks.iter().enumerate() {
                        if !Self::tick_within_limits(*x_val, x_min, x_max) {
                            continue;
                        }
                        let x_screen = if x_log {
                            if *x_val <= 0.0 || x_min <= 0.0 || x_max <= 0.0 {
                                continue;
                            }
                            plot_rect.min.x
                                + (((*x_val).log10() - x_min.log10())
                                    / (x_max.log10() - x_min.log10()))
                                    as f32
                                    * plot_rect.width()
                        } else {
                            plot_rect.min.x
                                + ((*x_val - x_min) / x_range) as f32 * plot_rect.width()
                        };
                        let x_screen = Self::snap_coord(x_screen, ppp);
                        ui.painter().line_segment(
                            [
                                Pos2::new(x_screen, border_bottom),
                                Pos2::new(x_screen, border_bottom + tick_length),
                            ],
                            Stroke::new(1.0_f32, axis_color),
                        );
                        let label = explicit_x_labels
                            .as_ref()
                            .and_then(|labels| labels.get(tick_idx).cloned())
                            .unwrap_or_else(|| x_tick_formatter.format(*x_val));
                        Self::draw_tick_label(
                            ui,
                            Pos2::new(x_screen, border_bottom + label_offset),
                            Align2::CENTER_CENTER,
                            truncate_label(&label, 14),
                            tick_font.clone(),
                            label_color,
                            x_tick_angle,
                        );
                    }
                }
            }
            if draw_x_axis && explicit_x_ticks.is_none() {
                if let Some(labels) = explicit_x_labels
                    .as_ref()
                    .filter(|labels| !labels.is_empty())
                {
                    cat_x = true;
                    let stride = Self::label_stride(labels, plot_rect.width(), tick_font.size);
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
                        ui.painter().line_segment(
                            [
                                Pos2::new(x_screen, border_bottom),
                                Pos2::new(x_screen, border_bottom + tick_length),
                            ],
                            Stroke::new(1.0_f32, axis_color),
                        );
                        let text = truncate_label(label, 14);
                        Self::draw_tick_label(
                            ui,
                            Pos2::new(x_screen, border_bottom + label_offset),
                            Align2::CENTER_CENTER,
                            text,
                            tick_font.clone(),
                            label_color,
                            x_tick_angle,
                        );
                    }
                }
            }
            if let Some(ticks) = explicit_y_ticks.as_ref() {
                cat_y = true;
                for (tick_idx, y_val) in ticks.iter().enumerate() {
                    if !Self::tick_within_limits(*y_val, y_min, y_max) {
                        continue;
                    }
                    let y_screen = if y_log {
                        if *y_val <= 0.0 || y_min <= 0.0 || y_max <= 0.0 {
                            continue;
                        }
                        plot_rect.max.y
                            - (((*y_val).log10() - y_min.log10()) / (y_max.log10() - y_min.log10()))
                                as f32
                                * plot_rect.height()
                    } else {
                        plot_rect.max.y - ((*y_val - y_min) / y_range) as f32 * plot_rect.height()
                    };
                    let y_screen = plot_renderer
                        .overlay_y_direction_for_axes(axes_index.unwrap_or(0))
                        .orient(y_screen, plot_rect.min.y, plot_rect.max.y);
                    let y_screen = Self::snap_coord(y_screen, ppp);
                    ui.painter().line_segment(
                        [
                            Pos2::new(y_tick_outer_x, y_screen),
                            Pos2::new(y_axis_x, y_screen),
                        ],
                        Stroke::new(1.0_f32, axis_color),
                    );
                    let label = explicit_y_labels
                        .as_ref()
                        .and_then(|labels| labels.get(tick_idx).cloned())
                        .unwrap_or_else(|| y_tick_formatter.format(*y_val));
                    Self::draw_tick_label(
                        ui,
                        Pos2::new(y_label_x, y_screen),
                        Align2::CENTER_CENTER,
                        truncate_label(&label, 14),
                        tick_font.clone(),
                        label_color,
                        y_tick_angle,
                    );
                }
            }
            if explicit_y_ticks.is_none() {
                if let Some(labels) = explicit_y_labels
                    .as_ref()
                    .filter(|labels| !labels.is_empty())
                {
                    cat_y = true;
                    let stride = Self::label_stride(labels, plot_rect.height(), tick_font.size);
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
                        let y_screen = plot_renderer
                            .overlay_y_direction_for_axes(axes_index.unwrap_or(0))
                            .orient(y_screen, plot_rect.min.y, plot_rect.max.y);
                        let y_screen = Self::snap_coord(y_screen, ppp);
                        ui.painter().line_segment(
                            [
                                Pos2::new(y_tick_outer_x, y_screen),
                                Pos2::new(y_axis_x, y_screen),
                            ],
                            Stroke::new(1.0_f32, axis_color),
                        );
                        let text = truncate_label(label, 14);
                        Self::draw_tick_label(
                            ui,
                            Pos2::new(y_label_x, y_screen),
                            Align2::CENTER_CENTER,
                            text,
                            tick_font.clone(),
                            label_color,
                            y_tick_angle,
                        );
                    }
                }
            }
            // An axes uses only its own category labels; the figure-wide labels apply only
            // when no axes is named, so a bar chart's categories do not leak into a subplot.
            let categorical = match axes_index {
                Some(idx) => plot_renderer.overlay_categorical_labels_for_axes(idx),
                None => plot_renderer
                    .overlay_categorical_labels()
                    .map(|(is_x, labels)| (is_x, labels.clone())),
            };
            if let Some((is_x, labels)) = categorical {
                if (is_x && cat_x) || (!is_x && cat_y) {
                    // Explicit axes tick labels take precedence over inferred bar labels.
                } else if is_x && draw_x_axis {
                    cat_x = true;
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
                            Stroke::new(1.0_f32, axis_color),
                        );
                        // Label
                        let text = truncate_label(label, 14);
                        Self::draw_tick_label(
                            ui,
                            Pos2::new(x_screen, border_bottom + label_offset),
                            Align2::CENTER_CENTER,
                            text,
                            tick_font.clone(),
                            label_color,
                            x_tick_angle,
                        );
                    }
                } else {
                    cat_y = true;
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
                        let y_screen = plot_renderer
                            .overlay_y_direction_for_axes(axes_index.unwrap_or(0))
                            .orient(y_screen, plot_rect.min.y, plot_rect.max.y);
                        let y_screen = Self::snap_coord(y_screen, ppp);
                        // Tick
                        ui.painter().line_segment(
                            [
                                Pos2::new(y_tick_outer_x, y_screen),
                                Pos2::new(y_axis_x, y_screen),
                            ],
                            Stroke::new(1.0_f32, axis_color),
                        );
                        // Label
                        let text = truncate_label(label, 14);
                        Self::draw_tick_label(
                            ui,
                            Pos2::new(y_label_x, y_screen),
                            Align2::CENTER_CENTER,
                            text,
                            tick_font.clone(),
                            label_color,
                            y_tick_angle,
                        );
                    }
                }
            }

            // Draw X-axis ticks and labels (categorical handled above)
            if draw_x_axis && x_log && !cat_x {
                let start_decade = x_min.log10().floor() as i32;
                let end_decade = x_max.log10().ceil() as i32;
                for d in start_decade..=end_decade {
                    let decade = 10f64.powi(d);
                    // Decades past the limits have no place on the axis.
                    if !Self::tick_within_log_limits(decade, x_min, x_max) {
                        continue;
                    }
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
                        Stroke::new(1.0_f32, axis_color),
                    );
                    // Label like 10^d
                    Self::draw_tick_label(
                        ui,
                        Pos2::new(x_screen, border_bottom + label_offset),
                        Align2::CENTER_CENTER,
                        Self::power_of_ten_label(d),
                        tick_font.clone(),
                        label_color,
                        x_tick_angle,
                    );
                }
            } else if draw_x_axis && !cat_x && !custom_hist_x {
                for x_val in plot_utils::generate_major_ticks(x_min, x_max) {
                    let x_screen =
                        plot_rect.min.x + ((x_val - x_min) / x_range) as f32 * plot_rect.width();
                    let x_screen = Self::snap_coord(x_screen, ppp);
                    ui.painter().line_segment(
                        [
                            Pos2::new(x_screen, border_bottom),
                            Pos2::new(x_screen, border_bottom + tick_length),
                        ],
                        Stroke::new(1.0_f32, axis_color),
                    );
                    Self::draw_tick_label(
                        ui,
                        Pos2::new(x_screen, border_bottom + label_offset),
                        Align2::CENTER_CENTER,
                        x_tick_formatter.format(x_val),
                        tick_font.clone(),
                        label_color,
                        x_tick_angle,
                    );
                }
            }

            // Draw Y-axis ticks and labels (categorical handled above)
            if y_log && !cat_y {
                let start_decade = y_min.log10().floor() as i32;
                let end_decade = y_max.log10().ceil() as i32;
                for d in start_decade..=end_decade {
                    let decade = 10f64.powi(d);
                    if !Self::tick_within_log_limits(decade, y_min, y_max) {
                        continue;
                    }
                    let y_screen = plot_rect.max.y
                        - ((decade.log10() - y_min.log10()) / (y_max.log10() - y_min.log10()))
                            as f32
                            * plot_rect.height();
                    let y_screen = plot_renderer
                        .overlay_y_direction_for_axes(axes_index.unwrap_or(0))
                        .orient(y_screen, plot_rect.min.y, plot_rect.max.y);
                    let y_screen = Self::snap_coord(y_screen, ppp);
                    ui.painter().line_segment(
                        [
                            Pos2::new(y_tick_outer_x, y_screen),
                            Pos2::new(y_axis_x, y_screen),
                        ],
                        Stroke::new(1.0_f32, axis_color),
                    );
                    Self::draw_tick_label(
                        ui,
                        Pos2::new(y_label_x, y_screen),
                        Align2::CENTER_CENTER,
                        Self::power_of_ten_label(d),
                        tick_font.clone(),
                        label_color,
                        y_tick_angle,
                    );
                }
            } else if !cat_y {
                for y_val in plot_utils::generate_major_ticks(y_min, y_max) {
                    let y_screen =
                        plot_rect.max.y - ((y_val - y_min) / y_range) as f32 * plot_rect.height();
                    let y_screen = plot_renderer
                        .overlay_y_direction_for_axes(axes_index.unwrap_or(0))
                        .orient(y_screen, plot_rect.min.y, plot_rect.max.y);
                    let y_screen = Self::snap_coord(y_screen, ppp);
                    ui.painter().line_segment(
                        [
                            Pos2::new(y_tick_outer_x, y_screen),
                            Pos2::new(y_axis_x, y_screen),
                        ],
                        Stroke::new(1.0_f32, axis_color),
                    );
                    Self::draw_tick_label(
                        ui,
                        Pos2::new(y_label_x, y_screen),
                        Align2::CENTER_CENTER,
                        y_tick_formatter.format(y_val),
                        tick_font.clone(),
                        label_color,
                        y_tick_angle,
                    );
                }
            }
        }
    }

    fn draw_tick_label(
        ui: &egui::Ui,
        pos: Pos2,
        anchor: Align2,
        text: impl ToString,
        font_id: FontId,
        text_color: Color32,
        angle_degrees: f64,
    ) {
        let text = text.to_string();
        if angle_degrees.abs() <= f64::EPSILON {
            if let Some(exponent) = text
                .strip_prefix("10^")
                .filter(|e| e.parse::<i32>().is_ok())
            {
                // `10` with the exponent raised and smaller.
                let painter = ui.painter();
                let base = painter.layout_no_wrap("10".to_string(), font_id.clone(), text_color);
                let small = FontId::new(font_id.size * 0.72, font_id.family.clone());
                let power = painter.layout_no_wrap(exponent.to_string(), small, text_color);
                let size = egui::vec2(
                    base.size().x + power.size().x,
                    base.size().y + power.size().y * 0.35,
                );
                let rect = anchor.anchor_size(pos, size);
                let base_pos = Pos2::new(rect.min.x, rect.max.y - base.size().y);
                painter.galley(base_pos, base.clone(), text_color);
                painter.galley(
                    Pos2::new(base_pos.x + base.size().x, rect.min.y),
                    power,
                    text_color,
                );
                return;
            }
            ui.painter().text(pos, anchor, text, font_id, text_color);
            return;
        }

        let galley = ui
            .painter()
            .layout_no_wrap(text.to_string(), font_id, text_color);
        let size = galley.size();
        let rect = anchor.anchor_size(pos, size);
        let local_anchor = pos - rect.min;
        let radians = (angle_degrees as f32).to_radians();
        let sin = radians.sin();
        let cos = radians.cos();
        let rotated_anchor = egui::vec2(
            local_anchor.x * cos - local_anchor.y * sin,
            local_anchor.x * sin + local_anchor.y * cos,
        );
        let mut shape = egui::epaint::TextShape::new(pos - rotated_anchor, galley, text_color);
        shape.angle = radians;
        shape.override_text_color = Some(text_color);
        ui.painter().add(shape);
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
        if !plot_renderer.shows_orientation_triad() {
            return;
        }
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
                dir_world: Vec3::Y
                    * plot_renderer
                        .overlay_y_direction_for_axes(axes_index)
                        .orient(1., -1., 1.),
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

        let axis_len = gizmo_size * 0.65;
        let head_len = (8.0 * scale).min(axis_len * 0.35);
        let head_w = 5.0 * scale;
        let font = FontId::proportional(11.0 * scale);
        // Inset the origin so every arm and its label stays whole, with a margin: the label
        // sits 10 px past the tip, is about 11 px tall, and keeps 8 px clear of the edge.
        let dirs: Vec<[f32; 2]> = axes
            .iter()
            .filter_map(|a| {
                let d = egui::Vec2::new(a.dir_world.x, -a.dir_world.y);
                let m = d.length();
                (m.is_finite() && m >= 1e-4).then(|| [d.x / m, d.y / m])
            })
            .collect();
        let [ox, oy] = crate::core::triad::triad_origin(
            [plot_rect.min.x, plot_rect.min.y],
            [plot_rect.max.x, plot_rect.max.y],
            (30.0 * scale).round(),
            axis_len + 24.0 * scale,
            &dirs,
        );
        let origin = Pos2::new(ox, oy);

        painter.circle_filled(origin, 2.0 * scale, Color32::from_gray(210));

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
        if !plot_renderer.shows_world_triad(axes_index) {
            return;
        }
        let cam_ref = plot_renderer
            .axes_camera(axes_index)
            .unwrap_or_else(|| plot_renderer.camera());
        let mut cam = cam_ref.clone();
        let w = plot_rect.width().max(1.0);
        let h = plot_rect.height().max(1.0);
        cam.update_aspect_ratio(w / h);
        let view_proj = cam.view_proj_matrix() * plot_renderer.axes_model_matrix(axes_index);

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

        // Label the frame the renderer drew its axis lines and tick marks from. Without finite
        // bounds, fall back to an origin triad sized from the on-screen scale.
        let frame = plot_renderer.axis_frame_3d(axes_index).unwrap_or_else(|| {
            let origin = Vec3::ZERO;
            let px_per = |axis: Vec3| match (project(origin), project(origin + axis)) {
                (Some(a), Some(b)) => ((b.x - a.x).hypot(b.y - a.y) as f64).max(1e-6),
                _ => 1.0,
            };
            let steps = [Vec3::X, Vec3::Y, Vec3::Z]
                .map(|axis| crate::core::plot_renderer::nice_tick_step(96.0 / px_per(axis)));
            crate::core::plot_renderer::AxisFrame3d {
                origin,
                world_origin: true,
                z_base: origin,
                lens: steps.map(|step| step as f32 * 5.0),
                steps,
            }
        });
        let origin = frame.origin;
        let lens = frame.lens;

        let scale = font_scale.max(0.75);
        let font = FontId::proportional(Self::axes_tick_font_size(
            plot_renderer,
            Some(axes_index),
            scale,
        ));
        // Clip to this axes' rectangle: projected 3D labels must not spill into
        // neighboring subplots when the camera puts them outside the panel.
        let painter = &ui.painter().with_clip_rect(plot_rect);
        let col_x = Color32::from_rgb(235, 80, 80);
        let col_y = Color32::from_rgb(90, 220, 120);
        let col_z = Color32::from_rgb(90, 160, 255);
        let panel_center = plot_rect.center();

        let outward_offset = |pos: Pos2, base: f32| {
            let dir = pos - panel_center;
            let len = dir.length().max(1.0);
            (dir / len) * base
        };

        if let Some(pos) = project(origin + Vec3::X * lens[0] * 1.10) {
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
                );
            }
        }
        if let Some(pos) = project(origin + Vec3::Y * lens[1] * 1.10) {
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
                );
            }
        }
        if let Some(pos) = project(frame.z_base + Vec3::Z * lens[2] * 1.10) {
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
                );
            }
        }

        // Geometry views keep the world axes and grid, but the plot tick values
        // obscure the model and its result legend.
        if plot_renderer.has_geometry_scene() {
            return;
        }

        let draw_axis = |axis_index: usize, axis: Vec3, color: Color32| {
            for (offset_along, value) in frame.ticks(axis_index) {
                let p = frame.axis_start(axis_index) + axis * offset_along;
                let Some(pos) = project(p) else { continue };
                // Offset labels slightly away from the axis in screen-space based on camera right/up.
                let offset =
                    outward_offset(pos, 7.0 * scale) + egui::Vec2::new(3.0 * scale, -3.0 * scale);
                let text = plot_utils::format_tick_label(value);
                painter.text(pos + offset, Align2::LEFT_CENTER, text, font.clone(), color);
            }
        };
        draw_axis(0, Vec3::X, col_x);
        draw_axis(1, Vec3::Y, col_y);
        draw_axis(2, Vec3::Z, col_z);
    }

    fn draw_title_in_rect(
        &self,
        ui: &mut egui::Ui,
        rect: Rect,
        title: &str,
        style: Option<&TextStyle>,
        scale: f32,
    ) {
        let scale = scale.max(0.75);
        let style = style.cloned().unwrap_or_default();
        let font_size = Self::style_font_size(&style, 16.0, scale);
        // Each line of a multi-line title is centered on its own, as in MATLAB.
        let lines: Vec<&str> = title.split('\n').collect();
        let line_h = font_size * 1.2;
        let first_y = rect.center().y - (lines.len() as f32 - 1.0) * line_h * 0.5;
        for (row, line) in lines.iter().enumerate() {
            Self::paint_styled_text(
                ui.painter(),
                egui::pos2(rect.center().x, first_y + row as f32 * line_h),
                Align2::CENTER_CENTER,
                line,
                font_size,
                Self::style_color(&style, self.theme_text_color()),
                Self::style_is_bold(&style),
            );
        }
    }

    /// Lines past the first in a title.
    fn extra_title_lines(title: Option<&str>) -> usize {
        title.map_or(0, |t| t.trim_end_matches('\n').matches('\n').count())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_title_block_in_rect(
        &self,
        ui: &mut egui::Ui,
        rect: Rect,
        title: Option<&str>,
        title_style: Option<&TextStyle>,
        subtitle: Option<&str>,
        subtitle_style: Option<&TextStyle>,
        scale: f32,
    ) {
        let title = title.filter(|text| Self::has_visible_text(Some(text)));
        let subtitle = subtitle.filter(|text| Self::has_visible_text(Some(text)));
        let Some((first_text, first_style, second)) = title
            .map(|title| (title, title_style, subtitle.map(|s| (s, subtitle_style))))
            .or_else(|| subtitle.map(|subtitle| (subtitle, subtitle_style, None)))
        else {
            return;
        };

        let scale = scale.max(0.75);
        let center = rect.center();
        if let Some((second_text, second_style)) = second {
            let first_style = first_style.cloned().unwrap_or_default();
            let second_style = second_style.cloned().unwrap_or_default();
            Self::paint_styled_text(
                ui.painter(),
                egui::pos2(center.x, center.y - 8.0 * scale),
                Align2::CENTER_CENTER,
                first_text,
                Self::style_font_size(&first_style, 16.0, scale),
                Self::style_color(&first_style, self.theme_text_color()),
                Self::style_is_bold(&first_style),
            );
            Self::paint_styled_text(
                ui.painter(),
                egui::pos2(center.x, center.y + 9.0 * scale),
                Align2::CENTER_CENTER,
                second_text,
                Self::style_font_size(&second_style, 13.0, scale),
                Self::style_color(&second_style, self.theme_text_color()),
                Self::style_is_bold(&second_style),
            );
        } else {
            self.draw_title_in_rect(ui, rect, first_text, first_style, scale);
        }
    }

    /// An axes' drawn data on screen in `plot_rect`, for a legend placed `best`: lines as their
    /// segments and filled marks as their triangles. Log axes keep data units in the scene and
    /// map by their logarithm, as the ticks do.
    /// Maps an axes' data coordinates to screen points in `plot_rect`. Log axes keep data units
    /// in the scene and map by their logarithm, as the ticks do; a value a log axis can't show
    /// (zero or negative) maps to nothing.
    fn data_to_screen(
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        plot_rect: Rect,
    ) -> Option<impl Fn([f32; 2]) -> Option<Pos2>> {
        let (l, r, b, t) = plot_renderer.view_bounds_for_axes(axes_index)?;
        let along = |log: bool, lo: f64, hi: f64| -> Option<Box<dyn Fn(f32) -> Option<f32>>> {
            if log {
                if lo <= 0.0 || hi <= 0.0 {
                    return None;
                }
                let (a, z) = (lo.log10(), hi.log10());
                ((z - a).abs() > f64::EPSILON).then(|| {
                    Box::new(move |v: f32| {
                        (v > 0.0).then(|| ((f64::from(v).log10() - a) / (z - a)) as f32)
                    }) as Box<dyn Fn(f32) -> Option<f32>>
                })
            } else {
                ((hi - lo).abs() > f64::EPSILON).then(|| {
                    Box::new(move |v: f32| Some(((f64::from(v) - lo) / (hi - lo)) as f32))
                        as Box<dyn Fn(f32) -> Option<f32>>
                })
            }
        };
        let fx = along(plot_renderer.overlay_x_log_for_axes(axes_index), l, r)?;
        let fy = along(plot_renderer.overlay_y_log_for_axes(axes_index), b, t)?;
        Some(move |[x, y]: [f32; 2]| -> Option<Pos2> {
            let (u, v) = (fx(x)?, fy(y)?);
            Some(Pos2::new(
                plot_rect.min.x + u * plot_rect.width(),
                plot_rect.max.y - v * plot_rect.height(),
            ))
        })
    }

    fn legend_obstacles(
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        plot_rect: Rect,
    ) -> super::legend_placement::Obstacles {
        use crate::core::plot_renderer::DataPrimitive;
        let mut obstacles = super::legend_placement::Obstacles::new(plot_rect);
        let (Some((l, r, b, t)), Some(screen)) = (
            plot_renderer.view_bounds_for_axes(axes_index),
            Self::data_to_screen(plot_renderer, axes_index, plot_rect),
        ) else {
            return obstacles;
        };
        for primitive in plot_renderer.overlay_data_primitives_for_axes(axes_index, 200_000) {
            match primitive {
                DataPrimitive::Point(p) => {
                    if let Some(p) = screen(p) {
                        obstacles.point(p);
                    }
                }
                DataPrimitive::Segment([a, b]) => {
                    if let (Some(a), Some(b)) = (screen(a), screen(b)) {
                        obstacles.segment(a, b);
                    }
                }
                DataPrimitive::Triangle([a, b, c]) => {
                    if let (Some(a), Some(b), Some(c)) = (screen(a), screen(b), screen(c)) {
                        obstacles.triangle(a, b, c);
                    }
                }
            }
        }
        // Reference lines (spec limits, a mean) are data too: each spans the axes.
        for (orientation, value) in plot_renderer.overlay_reference_lines_for_axes(axes_index) {
            let v = value as f32;
            let (p, q) = match orientation {
                crate::plots::ReferenceLineOrientation::Vertical => ([v, b as f32], [v, t as f32]),
                crate::plots::ReferenceLineOrientation::Horizontal => {
                    ([l as f32, v], [r as f32, v])
                }
            };
            if let (Some(p), Some(q)) = (screen(p), screen(q)) {
                obstacles.segment(p, q);
            }
        }
        obstacles
    }

    fn draw_axes_legend(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) {
        let entries = plot_renderer.overlay_legend_entries_for_axes(axes_index);
        let location = plot_renderer
            .overlay_legend_style_for_axes(axes_index)
            .and_then(|s| s.location.clone());
        let obstacles = if location
            .as_deref()
            .is_some_and(|l| l.trim().to_ascii_lowercase().starts_with("best"))
        {
            Self::legend_obstacles(plot_renderer, axes_index, plot_rect)
        } else {
            super::legend_placement::Obstacles::new(plot_rect)
        };
        self.draw_legend(
            ui,
            plot_rect,
            &entries,
            scale,
            location.as_deref(),
            &obstacles,
        );
    }

    fn draw_legend(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        entries: &[crate::plots::figure::LegendEntry],
        scale: f32,
        location: Option<&str>,
        obstacles: &super::legend_placement::Obstacles,
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
        // Size the box to its longest label (MATLAB's legend fits its text), within the axes.
        let label_font = FontId::proportional(11.0 * scale);
        let widest_label = entries
            .iter()
            .map(|e| {
                ui.painter()
                    .layout_no_wrap(e.label.clone(), label_font.clone(), legend_text)
                    .size()
                    .x
            })
            .fold(0.0_f32, f32::max);
        let legend_w = (text_x_gap + widest_label + pad)
            .max(92.0)
            .min((plot_rect.width() - 2.0 * pad).max(92.0));
        let size = egui::vec2(
            legend_w + pad,
            4.0 * scale + entries.len() as f32 * row_h + pad,
        );
        let origin =
            super::legend_placement::legend_origin(plot_rect, size, pad, location, obstacles);
        let x = origin.x + pad;
        let mut y = origin.y + 4.0 * scale;
        let legend_rect = Rect::from_min_size(origin, size);
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
                | crate::plots::figure::PlotType::Contour
                | crate::plots::figure::PlotType::ReferenceLine => {
                    let ymid = swatch_rect.center().y;
                    if e.line_visible {
                        // The swatch repeats the series' dash pattern, so a dashed spec limit
                        // reads dashed in the legend too.
                        for (a, b) in crate::plots::legend_swatch::legend_dashes(
                            e.line_style,
                            swatch_rect.min.x,
                            swatch_rect.max.x,
                            scale,
                        ) {
                            ui.painter().line_segment(
                                [Pos2::new(a, ymid), Pos2::new(b, ymid)],
                                Stroke::new(2.0_f32, c),
                            );
                        }
                    }
                    // The series' marker in the middle of the swatch, as MATLAB draws it.
                    if let Some(marker) = &e.marker {
                        Self::draw_legend_marker(ui, swatch_rect.center(), marker, 3.5 * scale);
                    }
                }
                crate::plots::figure::PlotType::Scatter
                | crate::plots::figure::PlotType::Scatter3 => {
                    let center = swatch_rect.center();
                    ui.painter().circle_filled(center, 3.5, c);
                    ui.painter()
                        .circle_stroke(center, 3.5, Stroke::new(1.0_f32, legend_stroke));
                }
                crate::plots::figure::PlotType::Bar
                | crate::plots::figure::PlotType::Area
                | crate::plots::figure::PlotType::Surface
                | crate::plots::figure::PlotType::Mesh
                | crate::plots::figure::PlotType::Patch
                | crate::plots::figure::PlotType::Pie
                | crate::plots::figure::PlotType::ContourFill => {
                    ui.painter().rect_filled(swatch_rect, 2.0, c);
                    ui.painter().rect_stroke(
                        swatch_rect,
                        2.0,
                        Stroke::new(1.0_f32, legend_stroke),
                        super::OUTLINE_STROKE_KIND,
                    );
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
                        Stroke::new(1.5_f32, c),
                    );
                    ui.painter().line_segment(
                        [
                            Pos2::new(swatch_rect.max.x - 4.0, ymid - 3.0),
                            Pos2::new(swatch_rect.max.x, ymid),
                        ],
                        Stroke::new(1.0_f32, c),
                    );
                }
            }
            ui.painter().text(
                egui::pos2(x + text_x_gap, y),
                Align2::LEFT_CENTER,
                &e.label,
                label_font.clone(),
                legend_text,
            );
            y += row_h;
        }
    }

    fn draw_legend_marker(
        ui: &mut egui::Ui,
        center: Pos2,
        marker: &crate::plots::figure::LegendMarker,
        radius: f32,
    ) {
        use crate::plots::scatter::MarkerStyle;
        let rgb = |v: glam::Vec4| {
            Color32::from_rgb(
                (v.x.clamp(0.0, 1.0) * 255.0) as u8,
                (v.y.clamp(0.0, 1.0) * 255.0) as u8,
                (v.z.clamp(0.0, 1.0) * 255.0) as u8,
            )
        };
        let edge = Stroke::new(1.2_f32, rgb(marker.edge_color));
        let fill = marker.filled.then(|| rgb(marker.face_color));
        let painter = ui.painter();
        match marker.kind {
            MarkerStyle::Square => {
                let rect = Rect::from_center_size(center, egui::vec2(2.0 * radius, 2.0 * radius));
                if let Some(fill) = fill {
                    painter.rect_filled(rect, 0.0, fill);
                }
                painter.rect_stroke(rect, 0.0, edge, super::OUTLINE_STROKE_KIND);
            }
            MarkerStyle::Plus | MarkerStyle::Cross | MarkerStyle::Star => {
                let d = radius;
                painter.line_segment(
                    [center - egui::vec2(d, 0.0), center + egui::vec2(d, 0.0)],
                    edge,
                );
                painter.line_segment(
                    [center - egui::vec2(0.0, d), center + egui::vec2(0.0, d)],
                    edge,
                );
            }
            _ => {
                if let Some(fill) = fill {
                    painter.circle_filled(center, radius, fill);
                }
                painter.circle_stroke(center, radius, edge);
            }
        }
    }

    fn draw_geometry_scalar_legend(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        legend: &GeometrySceneScalarLegend,
        scale: f32,
    ) {
        let scale = scale.max(0.75);
        let width = (plot_rect.width() * 0.34).clamp(190.0, 300.0);
        let pad = 12.0 * scale;
        let bar_height = 7.0 * scale;
        let note_height = if legend.note.is_some() {
            16.0 * scale
        } else {
            0.0
        };
        let rect = Rect::from_min_size(
            egui::pos2(plot_rect.max.x - width - pad, plot_rect.min.y + pad),
            egui::vec2(width, 56.0 * scale + note_height),
        );
        // Keep the figure-owned scale legible over scene annotations and mesh
        // imagery; the mesh counts belong in the viewport information control.
        ui.painter().rect_filled(
            rect.expand(6.0 * scale),
            4.0 * scale,
            self.theme_background_color(),
        );
        let theme = self.theme.build_theme();
        let text = theme.get_text_color();
        let text_color = Color32::from_rgb(
            (text.x.clamp(0.0, 1.0) * 255.0) as u8,
            (text.y.clamp(0.0, 1.0) * 255.0) as u8,
            (text.z.clamp(0.0, 1.0) * 255.0) as u8,
        );
        let muted = text_color.gamma_multiply(0.85);
        let label_font = FontId::proportional(11.0 * scale);
        // The range values are the data the legend exists for: as large as its label, so they
        // stay readable when a poster is shown small.
        let detail_font = FontId::proportional(11.0 * scale);
        ui.painter().text(
            rect.right_top(),
            Align2::RIGHT_TOP,
            &legend.label,
            label_font,
            text_color,
        );
        let bar_rect = Rect::from_min_size(
            egui::pos2(rect.min.x, rect.min.y + 23.0 * scale),
            egui::vec2(rect.width(), bar_height),
        );
        let stops = [
            Color32::from_rgb(26, 51, 184),
            Color32::from_rgb(5, 173, 230),
            Color32::from_rgb(31, 209, 122),
            Color32::from_rgb(250, 209, 31),
            Color32::from_rgb(224, 26, 31),
        ];
        let steps = 80;
        for step in 0..steps {
            let t = step as f32 / (steps - 1) as f32;
            let scaled = t * (stops.len() - 1) as f32;
            let low = scaled.floor() as usize;
            let high = (low + 1).min(stops.len() - 1);
            let amount = scaled - low as f32;
            let low_rgba = stops[low].to_array();
            let high_rgba = stops[high].to_array();
            let color = Color32::from_rgb(
                (low_rgba[0] as f32 + (high_rgba[0] as f32 - low_rgba[0] as f32) * amount).round()
                    as u8,
                (low_rgba[1] as f32 + (high_rgba[1] as f32 - low_rgba[1] as f32) * amount).round()
                    as u8,
                (low_rgba[2] as f32 + (high_rgba[2] as f32 - low_rgba[2] as f32) * amount).round()
                    as u8,
            );
            let x0 = bar_rect.min.x + bar_rect.width() * step as f32 / steps as f32;
            let x1 = bar_rect.min.x + bar_rect.width() * (step + 1) as f32 / steps as f32;
            ui.painter().rect_filled(
                Rect::from_min_max(
                    egui::pos2(x0, bar_rect.min.y),
                    egui::pos2(x1, bar_rect.max.y),
                ),
                0.0,
                color,
            );
        }
        let precision = LegendPrecision::for_range(legend.minimum, legend.maximum);
        ui.painter().text(
            bar_rect.left_bottom() + egui::vec2(0.0, 3.0 * scale),
            Align2::LEFT_TOP,
            with_unit(precision.format(legend.minimum), &legend.unit),
            detail_font.clone(),
            muted,
        );
        ui.painter().text(
            bar_rect.right_bottom() + egui::vec2(0.0, 3.0 * scale),
            Align2::RIGHT_TOP,
            with_unit(precision.format(legend.maximum), &legend.unit),
            detail_font.clone(),
            muted,
        );
        // A limit inside the range is marked across the bar where it falls.
        if let Some(limit) = legend.limit.filter(|limit| {
            legend.maximum > legend.minimum && (legend.minimum..=legend.maximum).contains(limit)
        }) {
            let t = ((limit - legend.minimum) / (legend.maximum - legend.minimum)) as f32;
            let x = bar_rect.min.x + bar_rect.width() * t;
            ui.painter().line_segment(
                [
                    egui::pos2(x, bar_rect.min.y - 3.0 * scale),
                    egui::pos2(x, bar_rect.max.y + 3.0 * scale),
                ],
                Stroke::new(2.0 * scale, text_color),
            );
        }
        if let Some(note) = &legend.note {
            ui.painter().text(
                rect.right_bottom(),
                Align2::RIGHT_BOTTOM,
                note,
                detail_font,
                text_color,
            );
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
                FontId::proportional(Self::Y_LABEL_FONT_POINTS * scale),
                text_color,
            )
        });
        let size = galley.size();
        // Sit in the band's left zone reserved by `estimate_y_axis_band_width`, so the title
        // never reaches the tick labels however wide they are.
        let center = Pos2::new(rect.min.x + 2.0 * scale + size.y * 0.5, rect.center().y);
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
        cam.update_aspect_ratio(plot_rect.width().max(1.0) / plot_rect.height().max(1.0));
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

    /// The anchor for text with an explicit MATLAB alignment; `None` when neither is set.
    fn text_alignment_anchor(style: &TextStyle) -> Option<Align2> {
        if style.horizontal_alignment.is_none() && style.vertical_alignment.is_none() {
            return None;
        }
        let h = match style
            .horizontal_alignment
            .as_deref()
            .map(str::to_ascii_lowercase)
        {
            Some(ref a) if a == "center" => egui::Align::Center,
            Some(ref a) if a == "right" => egui::Align::Max,
            _ => egui::Align::Min,
        };
        let v = match style
            .vertical_alignment
            .as_deref()
            .map(str::to_ascii_lowercase)
        {
            Some(ref a) if a == "top" || a == "cap" => egui::Align::Min,
            Some(ref a) if a == "bottom" || a == "baseline" => egui::Align::Max,
            _ => egui::Align::Center,
        };
        Some(Align2([h, v]))
    }

    /// A decade tick label, `10^d`; `draw_tick_label` draws the exponent raised and smaller,
    /// as MATLAB draws log ticks.
    fn power_of_ten_label(exponent: i32) -> String {
        format!("10^{exponent}")
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
        // On a 2D axes with a log scale, the scene keeps data units and the shaders map them by
        // their logarithm; text maps the same way, so `text(1.5e5, y, …)` sits at 1.5e5.
        let log_2d = !is_3d
            && (plot_renderer.overlay_x_log_for_axes(axes_index)
                || plot_renderer.overlay_y_log_for_axes(axes_index));
        let to_screen = if log_2d {
            Self::data_to_screen(plot_renderer, axes_index, plot_rect)
        } else {
            None
        };
        for (position, text, style) in annotations {
            if !style.visible || text.trim().is_empty() {
                continue;
            }
            let screen = match &to_screen {
                Some(map) => map([position.x, position.y]),
                None => {
                    let position = plot_renderer
                        .axes_model_matrix(axes_index)
                        .transform_point3(position);
                    self.project_world_to_screen(plot_rect, camera, position)
                }
            };
            let Some(screen) = screen else {
                continue;
            };
            let color = Self::style_color(&style, self.theme_text_color());
            let font_size = Self::style_font_size(&style, 14.0, scale);
            let (offset, anchor) = match Self::text_alignment_anchor(&style) {
                Some(anchor) => (egui::Vec2::ZERO, anchor),
                None if is_3d => (
                    egui::vec2(0.0, -8.0 * scale.max(0.75)),
                    Align2::CENTER_CENTER,
                ),
                None => (
                    egui::vec2(0.0, 6.0 * scale.max(0.75)),
                    Align2::CENTER_CENTER,
                ),
            };
            Self::paint_styled_text(
                &ui.painter().with_clip_rect(plot_rect),
                screen + offset,
                anchor,
                &text,
                font_size,
                color,
                Self::style_is_bold(&style),
            );
        }
    }

    /// False after `axis off` (the axes style is hidden).
    fn axes_visible(plot_renderer: &PlotRenderer, axes_index: usize) -> bool {
        plot_renderer
            .overlay_axes_style_for_axes(axes_index)
            .is_none_or(|style| style.visible)
    }

    /// The colorbar for an axes, just right of its plot box, spanning the box height, with
    /// ticks at nice values of the axes' color range.
    fn draw_colorbar(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) {
        if !plot_renderer.overlay_colorbar_enabled_for_axes(axes_index) {
            return;
        }
        let scale = scale.max(0.75);
        let gap = 10.0 * scale;
        let bar_width = 14.0 * scale;
        let bar_rect = Rect::from_min_max(
            egui::pos2(plot_rect.max.x + gap, plot_rect.min.y),
            egui::pos2(plot_rect.max.x + gap + bar_width, plot_rect.max.y),
        );
        if bar_rect.height() < 4.0 {
            return;
        }
        let cmap = plot_renderer.overlay_colormap_for_axes(axes_index);
        let steps = 128;
        for i in 0..steps {
            let t0 = i as f32 / steps as f32;
            let t1 = (i + 1) as f32 / steps as f32;
            let y0 = bar_rect.max.y - t0 * bar_rect.height();
            let y1 = bar_rect.max.y - t1 * bar_rect.height();
            let c = cmap.map_value((t0 + t1) * 0.5);
            let col = Color32::from_rgb(
                (c.x.clamp(0.0, 1.0) * 255.0) as u8,
                (c.y.clamp(0.0, 1.0) * 255.0) as u8,
                (c.z.clamp(0.0, 1.0) * 255.0) as u8,
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
        let axis_color = self.theme_axis_color();
        let stroke = Stroke::new(1.0_f32, axis_color);
        ui.painter()
            .rect_stroke(bar_rect, 0.0, stroke, super::OUTLINE_STROKE_KIND);

        let Some((lo, hi)) = plot_renderer.overlay_color_range_for_axes(axes_index) else {
            return;
        };
        if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
            return;
        }
        let tick_len = 4.0 * scale;
        let font = FontId::proportional(11.0 * scale);
        let text_color = self.theme_text_color();
        for value in plot_utils::generate_major_ticks(lo, hi) {
            let t = ((value - lo) / (hi - lo)) as f32;
            if !(-1.0e-4..=1.0 + 1.0e-4).contains(&t) {
                continue;
            }
            let y = bar_rect.max.y - t * bar_rect.height();
            ui.painter().line_segment(
                [
                    egui::pos2(bar_rect.max.x - tick_len, y),
                    egui::pos2(bar_rect.max.x, y),
                ],
                stroke,
            );
            ui.painter().text(
                egui::pos2(bar_rect.max.x + 4.0 * scale, y),
                Align2::LEFT_CENTER,
                plot_utils::format_tick_label(value),
                font.clone(),
                text_color,
            );
        }
    }

    /// `xline` and `yline` labels in the line's color, as MATLAB places them by default:
    /// along a vertical line, reading upward from the top, on its right; above a horizontal
    /// line at its right end.
    fn draw_reference_line_labels(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) {
        let labels = plot_renderer.reference_line_labels_for_axes(axes_index);
        if labels.is_empty() {
            return;
        }
        let Some((x_min, x_max, y_min, y_max)) = plot_renderer.view_bounds_for_axes(axes_index)
        else {
            return;
        };
        // The label sits at its line's value through the axes' own scale, so a line on a log
        // axis is labelled where it is drawn.
        let Some(to_screen) = Self::data_to_screen(plot_renderer, axes_index, plot_rect) else {
            return;
        };
        let scale = scale.max(0.75);
        let pad = 4.0 * scale;
        let painter = ui.painter().with_clip_rect(plot_rect);
        for (orientation, value, label, color) in labels {
            let rgba = match color {
                Some(color) => Color32::from_rgba_unmultiplied(
                    (color.x.clamp(0.0, 1.0) * 255.0) as u8,
                    (color.y.clamp(0.0, 1.0) * 255.0) as u8,
                    (color.z.clamp(0.0, 1.0) * 255.0) as u8,
                    255,
                ),
                None => self.theme_axis_color(),
            };
            let galley = ui.fonts(|fonts| {
                fonts.layout_no_wrap(label, FontId::proportional(12.0 * scale), rgba)
            });
            let size = galley.size();
            match orientation {
                crate::plots::ReferenceLineOrientation::Vertical => {
                    let mid = [value as f32, ((y_min + y_max) * 0.5) as f32];
                    let Some(at) = to_screen(mid) else {
                        continue;
                    };
                    let pos = Pos2::new(at.x + 2.0 * scale, plot_rect.min.y + pad + size.x);
                    let mut shape = egui::epaint::TextShape::new(pos, galley, rgba);
                    shape.angle = -std::f32::consts::FRAC_PI_2;
                    shape.override_text_color = Some(rgba);
                    painter.add(shape);
                }
                crate::plots::ReferenceLineOrientation::Horizontal => {
                    let mid = [((x_min + x_max) * 0.5) as f32, value as f32];
                    let Some(at) = to_screen(mid) else {
                        continue;
                    };
                    let pos =
                        Pos2::new(plot_rect.max.x - pad - size.x, at.y - 2.0 * scale - size.y);
                    painter.galley(pos, galley, rgba);
                }
            }
        }
    }

    /// Heatmap cell values, centered in their cells, in black or white for contrast.
    fn draw_cell_labels(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        scale: f32,
    ) {
        let marks = plot_renderer.cell_label_marks_for_axes(axes_index);
        if marks.is_empty() {
            return;
        }
        let Some(camera) = plot_renderer
            .axes_camera(axes_index)
            .or_else(|| Some(plot_renderer.camera()))
        else {
            return;
        };
        let painter = ui.painter().with_clip_rect(plot_rect);
        let font = FontId::proportional(12.0 * scale.max(0.75));
        for (position, text, cell) in marks {
            let Some(screen) = self.project_world_to_screen(plot_rect, camera, position) else {
                continue;
            };
            let luminance = 0.2126 * cell.x + 0.7152 * cell.y + 0.0722 * cell.z;
            let color = if luminance > 0.5 {
                Color32::from_rgb(20, 20, 20)
            } else {
                Color32::WHITE
            };
            painter.text(screen, Align2::CENTER_CENTER, text, font.clone(), color);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_pie_label(
        &self,
        ui: &mut egui::Ui,
        plot_rect: Rect,
        plot_renderer: &PlotRenderer,
        axes_index: usize,
        label: &str,
        pos: glam::Vec2,
        scale: f32,
    ) {
        // Labels sit at data positions just outside each slice; place them through the axes
        // camera so they follow the pie wherever its box is drawn.
        let camera = plot_renderer
            .axes_camera(axes_index)
            .unwrap_or_else(|| plot_renderer.camera());
        let Some(screen) =
            self.project_world_to_screen(plot_rect, camera, glam::Vec3::new(pos.x, pos.y, 0.0))
        else {
            return;
        };
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

    pub(crate) fn take_cad_actions(&mut self) -> CadOverlayActions {
        std::mem::take(&mut self.cad_actions)
    }

    pub(crate) fn overlay_pointer_captured(&self) -> bool {
        self.cad_overlay.pointer_captured()
    }

    pub(crate) fn overlay_capture_regions(&self) -> Vec<[f32; 4]> {
        self.cad_overlay
            .capture_regions()
            .iter()
            .map(|rect| [rect.min.x, rect.min.y, rect.max.x, rect.max.y])
            .collect()
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

#[cfg(test)]
mod positioned_axes_tests {
    use super::{OverlayConfig, PlotOverlay};
    use crate::core::Camera;
    use egui::{pos2, vec2, Rect};
    use glam::Vec3;

    #[test]
    fn default_plot_toolbar_preserves_full_available_width() {
        let ctx = egui::Context::default();
        let screen = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 640.0));
        let mut overlay = PlotOverlay::new();
        let config = OverlayConfig::default();
        assert!(config.show_toolbar);
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default()
                    .frame(super::transparent_frame())
                    .show(ctx, |ui| {
                        overlay.render_toolbar(ui, config.show_grid, true);
                        let plot = ui.available_rect_before_wrap();
                        assert_eq!(plot.min.x, screen.min.x);
                        assert_eq!(plot.max.x, screen.max.x);
                        assert!(plot.height() > 0.0);
                    });
            },
        );
        let toolbar = overlay.toolbar_rect().expect("default toolbar is retained");
        assert!(toolbar.width() > 0.0);
        assert!(screen.contains_rect(toolbar));
    }

    #[test]
    fn overlay_annotations_are_absent_by_default_but_explicit_text_remains_visible() {
        let mut config = OverlayConfig::default();
        assert!(config.show_title);
        assert!(config.title.is_none());
        assert!(config.x_label.is_none());
        assert!(config.y_label.is_none());
        assert!(!PlotOverlay::has_visible_text(config.title.as_deref()));
        assert!(!PlotOverlay::has_visible_text(config.x_label.as_deref()));
        assert!(!PlotOverlay::has_visible_text(config.y_label.as_deref()));

        config.title = Some("Explicit title".to_string());
        config.x_label = Some("Horizontal".to_string());
        config.y_label = Some("Vertical".to_string());
        assert!(PlotOverlay::has_visible_text(config.title.as_deref()));
        assert!(PlotOverlay::has_visible_text(config.x_label.as_deref()));
        assert!(PlotOverlay::has_visible_text(config.y_label.as_deref()));
    }

    #[test]
    fn positioned_axes_accepts_a_surface_smaller_than_overlay_margins() {
        let outer = Rect::from_min_size(pos2(8.0, 8.0), vec2(0.0, 0.0));
        let rect = PlotOverlay::positioned_axes_rect(outer, [0.2, 0.2, 0.6, 0.6], "normalized");
        assert!(rect.min.x.is_finite() && rect.min.y.is_finite());
        assert!(rect.width() >= 1.0 && rect.height() >= 1.0);
    }

    #[test]
    fn positioned_axes_keeps_normalized_placement_on_a_regular_surface() {
        let outer = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
        let rect = PlotOverlay::positioned_axes_rect(outer, [0.2, 0.2, 0.6, 0.6], "normalized");
        assert!((rect.min.x - 20.0).abs() < 1.0e-4);
        assert!((rect.min.y - 20.0).abs() < 1.0e-4);
        assert!((rect.width() - 60.0).abs() < 1.0e-4);
        assert!((rect.height() - 60.0).abs() < 1.0e-4);
    }

    #[test]
    fn projected_world_text_uses_the_plot_rect_camera_aspect() {
        let overlay = PlotOverlay::new();
        let rect = Rect::from_min_size(pos2(20.0, 30.0), vec2(200.0, 700.0));
        let mut camera = Camera::new();
        let point = Vec3::new(0.3, 0.0, 0.0);
        let screen = overlay
            .project_world_to_screen(rect, &camera, point)
            .unwrap();
        camera.update_aspect_ratio(rect.width() / rect.height());
        let clip = camera.view_proj_matrix() * point.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let expected_x = rect.min.x + (ndc.x + 1.0) * 0.5 * rect.width();
        let expected_y = rect.min.y + (1.0 - (ndc.y + 1.0) * 0.5) * rect.height();
        assert!((screen.x - expected_x).abs() < 1.0e-4);
        assert!((screen.y - expected_y).abs() < 1.0e-4);
    }
}
