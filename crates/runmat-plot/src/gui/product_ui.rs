//! Product-styled egui primitives shared by plot/CAD overlays.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProductToolbarIcon {
    Reset,
    View,
    Grid,
    Xray,
    Tree,
    Info,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProductToolbarTheme {
    pub fill: egui::Color32,
    pub hover_fill: egui::Color32,
    pub pressed_fill: egui::Color32,
    pub stroke: egui::Color32,
    pub text: egui::Color32,
    pub active_fill: egui::Color32,
    pub active_hover_fill: egui::Color32,
    pub active_stroke: egui::Color32,
    pub active_text: egui::Color32,
    pub tooltip_fill: egui::Color32,
    pub tooltip_stroke: egui::Color32,
    pub tooltip_text: egui::Color32,
    pub primary_fill: egui::Color32,
    pub primary_hover_fill: egui::Color32,
    pub primary_pressed_fill: egui::Color32,
    pub primary_stroke: egui::Color32,
    pub primary_text: egui::Color32,
}

impl ProductToolbarTheme {
    pub(crate) fn from_context(ctx: &egui::Context) -> Self {
        if ctx.style().visuals.dark_mode {
            Self {
                fill: egui::Color32::from_rgba_unmultiplied(9, 9, 11, 188),
                hover_fill: egui::Color32::from_rgba_unmultiplied(39, 39, 42, 214),
                pressed_fill: egui::Color32::from_rgba_unmultiplied(63, 63, 70, 222),
                stroke: egui::Color32::from_rgba_unmultiplied(63, 63, 70, 190),
                text: egui::Color32::from_rgb(161, 161, 170),
                active_fill: egui::Color32::from_rgba_unmultiplied(39, 39, 42, 214),
                active_hover_fill: egui::Color32::from_rgba_unmultiplied(52, 52, 58, 222),
                active_stroke: egui::Color32::from_rgba_unmultiplied(82, 82, 91, 198),
                active_text: egui::Color32::from_rgb(212, 212, 216),
                tooltip_fill: egui::Color32::from_rgba_unmultiplied(17, 17, 19, 246),
                tooltip_stroke: egui::Color32::from_rgba_unmultiplied(63, 63, 70, 220),
                tooltip_text: egui::Color32::from_rgb(228, 228, 231),
                primary_fill: egui::Color32::from_rgb(192, 132, 252),
                primary_hover_fill: egui::Color32::from_rgb(216, 180, 254),
                primary_pressed_fill: egui::Color32::from_rgb(168, 85, 247),
                primary_stroke: egui::Color32::from_rgb(216, 180, 254),
                primary_text: egui::Color32::from_rgb(24, 24, 27),
            }
        } else {
            Self {
                fill: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 232),
                hover_fill: egui::Color32::from_rgba_unmultiplied(244, 244, 245, 244),
                pressed_fill: egui::Color32::from_rgba_unmultiplied(228, 228, 231, 248),
                stroke: egui::Color32::from_rgba_unmultiplied(228, 228, 231, 232),
                text: egui::Color32::from_rgb(113, 113, 122),
                active_fill: egui::Color32::from_rgba_unmultiplied(244, 244, 245, 244),
                active_hover_fill: egui::Color32::from_rgba_unmultiplied(239, 239, 241, 248),
                active_stroke: egui::Color32::from_rgba_unmultiplied(212, 212, 216, 232),
                active_text: egui::Color32::from_rgb(82, 82, 91),
                tooltip_fill: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 250),
                tooltip_stroke: egui::Color32::from_rgba_unmultiplied(212, 212, 216, 238),
                tooltip_text: egui::Color32::from_rgb(63, 63, 70),
                primary_fill: egui::Color32::from_rgb(168, 85, 247),
                primary_hover_fill: egui::Color32::from_rgb(147, 51, 234),
                primary_pressed_fill: egui::Color32::from_rgb(126, 34, 206),
                primary_stroke: egui::Color32::from_rgb(147, 51, 234),
                primary_text: egui::Color32::from_rgb(255, 255, 255),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ToolbarVisuals {
    fill: egui::Color32,
    stroke: egui::Stroke,
    content: egui::Color32,
}

pub(crate) fn toolbar_text_button(
    ui: &mut egui::Ui,
    label: &str,
    active: bool,
    scale: f32,
    theme: &ProductToolbarTheme,
    min_width: f32,
) -> egui::Response {
    let content_width = label.chars().count() as f32 * 6.2 + 18.0;
    let width = min_width.max(content_width) * scale;
    let size = egui::vec2(width, 26.0 * scale);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let visuals = toolbar_visuals(&response, active, theme);
    paint_toolbar_frame(ui, rect, scale, visuals);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(11.0 * scale),
        visuals.content,
    );
    response
}

pub(crate) fn toolbar_icon_button(
    ui: &mut egui::Ui,
    icon: ProductToolbarIcon,
    tooltip: &'static str,
    active: bool,
    scale: f32,
    theme: &ProductToolbarTheme,
) -> egui::Response {
    let size = egui::vec2(44.0 * scale, 36.0 * scale);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let visuals = toolbar_visuals(&response, active, theme);
    paint_toolbar_frame_with_rounding(ui, rect, 0.0, visuals);
    paint_toolbar_icon(ui, icon, rect, scale, visuals.content);
    if response.hovered() {
        show_toolbar_tooltip(ui.ctx(), response.id, rect, tooltip, scale, theme);
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(16));
    }
    response
}

pub(crate) fn toolbar_primary_text_button(
    ui: &mut egui::Ui,
    label: &'static str,
    tooltip: &'static str,
    scale: f32,
    theme: &ProductToolbarTheme,
) -> egui::Response {
    let content_width = label.chars().count() as f32 * 6.8 + 24.0;
    let size = egui::vec2(content_width.max(124.0) * scale, 36.0 * scale);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let fill = if response.is_pointer_button_down_on() {
        theme.primary_pressed_fill
    } else if response.hovered() {
        theme.primary_hover_fill
    } else {
        theme.primary_fill
    };
    paint_toolbar_frame_with_rounding(
        ui,
        rect,
        0.0,
        ToolbarVisuals {
            fill,
            stroke: egui::Stroke::new(1.0, theme.primary_stroke),
            content: theme.primary_text,
        },
    );
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(11.0 * scale),
        theme.primary_text,
    );
    if response.hovered() {
        show_toolbar_tooltip(ui.ctx(), response.id, rect, tooltip, scale, theme);
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(16));
    }
    response
}

fn show_toolbar_tooltip(
    ctx: &egui::Context,
    id: egui::Id,
    anchor: egui::Rect,
    text: &'static str,
    scale: f32,
    theme: &ProductToolbarTheme,
) {
    let margin_x = 10.0 * scale;
    let margin_y = 7.0 * scale;
    let pos = egui::pos2(anchor.left() - 8.0 * scale, anchor.bottom() + 7.0 * scale);
    egui::Area::new(id.with("product_tooltip"))
        .order(egui::Order::Tooltip)
        .fixed_pos(pos)
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::none()
                .fill(theme.tooltip_fill)
                .stroke(egui::Stroke::new(1.0, theme.tooltip_stroke))
                .rounding(scaled_corner_radius(2.0 * scale))
                .inner_margin(scaled_margin_symmetric(margin_x, margin_y))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(text)
                            .size(12.0 * scale)
                            .color(theme.tooltip_text),
                    );
                });
        });
}

fn toolbar_visuals(
    response: &egui::Response,
    active: bool,
    theme: &ProductToolbarTheme,
) -> ToolbarVisuals {
    let fill = if active {
        if response.hovered() {
            theme.active_hover_fill
        } else {
            theme.active_fill
        }
    } else if response.is_pointer_button_down_on() {
        theme.pressed_fill
    } else if response.hovered() {
        theme.hover_fill
    } else {
        theme.fill
    };
    let (stroke_color, content) = if active {
        (theme.active_stroke, theme.active_text)
    } else {
        (theme.stroke, theme.text)
    };
    ToolbarVisuals {
        fill,
        stroke: egui::Stroke::new(1.0, stroke_color),
        content,
    }
}

fn paint_toolbar_frame(ui: &egui::Ui, rect: egui::Rect, scale: f32, visuals: ToolbarVisuals) {
    paint_toolbar_frame_with_rounding(ui, rect, 4.0 * scale, visuals);
}

fn paint_toolbar_frame_with_rounding(
    ui: &egui::Ui,
    rect: egui::Rect,
    rounding: f32,
    visuals: ToolbarVisuals,
) {
    #[cfg(target_arch = "wasm32")]
    ui.painter().rect(
        rect,
        scaled_corner_radius(rounding),
        visuals.fill,
        visuals.stroke,
        egui::StrokeKind::Inside,
    );
    #[cfg(not(target_arch = "wasm32"))]
    ui.painter()
        .rect(rect, rounding, visuals.fill, visuals.stroke);
}

#[cfg(target_arch = "wasm32")]
fn scaled_corner_radius(radius: f32) -> egui::CornerRadius {
    egui::CornerRadius::same(radius.round().clamp(0.0, u8::MAX as f32) as u8)
}

#[cfg(not(target_arch = "wasm32"))]
fn scaled_corner_radius(radius: f32) -> egui::Rounding {
    egui::Rounding::same(radius)
}

#[cfg(target_arch = "wasm32")]
fn scaled_margin_symmetric(x: f32, y: f32) -> egui::Margin {
    egui::Margin::symmetric(
        x.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8,
        y.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn scaled_margin_symmetric(x: f32, y: f32) -> egui::Margin {
    egui::Margin::symmetric(x, y)
}

fn paint_toolbar_icon(
    ui: &egui::Ui,
    icon: ProductToolbarIcon,
    rect: egui::Rect,
    scale: f32,
    color: egui::Color32,
) {
    let stroke = egui::Stroke::new(1.45 * scale, color);
    let icon_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(15.0, 15.0) * scale);
    match icon {
        ProductToolbarIcon::Reset => paint_reset_icon(ui, icon_rect, scale, stroke),
        ProductToolbarIcon::View => paint_view_icon(ui, icon_rect, scale, stroke),
        ProductToolbarIcon::Grid => paint_grid_icon(ui, icon_rect, scale, stroke),
        ProductToolbarIcon::Xray => paint_xray_icon(ui, icon_rect, scale, stroke),
        ProductToolbarIcon::Tree => paint_tree_icon(ui, icon_rect, scale, stroke),
        ProductToolbarIcon::Info => paint_info_icon(ui, icon_rect, scale, stroke, color),
    }
}

fn paint_reset_icon(ui: &egui::Ui, rect: egui::Rect, scale: f32, stroke: egui::Stroke) {
    let center = rect.center();
    let radius = 6.4 * scale;
    let mut points = Vec::with_capacity(18);
    for index in 0..=16 {
        let angle = -0.15 + index as f32 / 16.0 * 1.65 * std::f32::consts::TAU;
        points.push(center + egui::vec2(angle.cos(), angle.sin()) * radius);
    }
    ui.painter().add(egui::Shape::line(points, stroke));

    let tip_angle: f32 = -0.15;
    let tip = center + egui::vec2(tip_angle.cos(), tip_angle.sin()) * radius;
    let tangent = egui::vec2(-tip_angle.sin(), tip_angle.cos());
    let inward = (center - tip).normalized();
    ui.painter().line_segment(
        [tip, tip - tangent * 4.0 * scale + inward * 1.8 * scale],
        stroke,
    );
    ui.painter().line_segment(
        [tip, tip - tangent * 1.0 * scale + inward * 4.2 * scale],
        stroke,
    );
}

fn paint_view_icon(ui: &egui::Ui, rect: egui::Rect, scale: f32, stroke: egui::Stroke) {
    let center = rect.center();
    let front = egui::Rect::from_center_size(
        center + egui::vec2(-1.7, 2.0) * scale,
        egui::vec2(8.5, 8.5) * scale,
    );
    let back = egui::Rect::from_center_size(
        center + egui::vec2(3.0, -2.6) * scale,
        egui::vec2(8.5, 8.5) * scale,
    );
    #[cfg(target_arch = "wasm32")]
    ui.painter().rect_stroke(
        back,
        scaled_corner_radius(0.0),
        stroke,
        egui::StrokeKind::Inside,
    );
    #[cfg(target_arch = "wasm32")]
    ui.painter().rect_stroke(
        front,
        scaled_corner_radius(0.0),
        stroke,
        egui::StrokeKind::Inside,
    );
    #[cfg(not(target_arch = "wasm32"))]
    ui.painter().rect_stroke(back, 0.0, stroke);
    #[cfg(not(target_arch = "wasm32"))]
    ui.painter().rect_stroke(front, 0.0, stroke);
    for (a, b) in [
        (front.left_top(), back.left_top()),
        (front.right_top(), back.right_top()),
        (front.left_bottom(), back.left_bottom()),
        (front.right_bottom(), back.right_bottom()),
    ] {
        ui.painter().line_segment([a, b], stroke);
    }
}

fn paint_grid_icon(ui: &egui::Ui, rect: egui::Rect, _scale: f32, stroke: egui::Stroke) {
    for factor in [0.0, 0.5, 1.0] {
        let x = egui::lerp(rect.left()..=rect.right(), factor);
        ui.painter().line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            stroke,
        );
        let y = egui::lerp(rect.top()..=rect.bottom(), factor);
        ui.painter().line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            stroke,
        );
    }
}

fn paint_xray_icon(ui: &egui::Ui, rect: egui::Rect, scale: f32, stroke: egui::Stroke) {
    let center = rect.center();
    let rx = rect.width() * 0.48;
    let ry = rect.height() * 0.31;
    let mut upper = Vec::with_capacity(11);
    let mut lower = Vec::with_capacity(11);
    for index in 0..=10 {
        let t = index as f32 / 10.0;
        let x = egui::lerp((center.x - rx)..=(center.x + rx), t);
        let y_offset = (1.0 - (2.0 * t - 1.0).abs()).powf(0.65) * ry;
        upper.push(egui::pos2(x, center.y - y_offset));
        lower.push(egui::pos2(x, center.y + y_offset));
    }
    ui.painter().add(egui::Shape::line(upper, stroke));
    ui.painter().add(egui::Shape::line(lower, stroke));
    ui.painter().circle_stroke(
        center,
        2.2 * scale,
        egui::Stroke::new(1.3 * scale, stroke.color),
    );
}

fn paint_tree_icon(ui: &egui::Ui, rect: egui::Rect, scale: f32, stroke: egui::Stroke) {
    let root = egui::pos2(rect.left() + 3.0 * scale, rect.center().y);
    let mid_x = rect.center().x - 1.0 * scale;
    let top = egui::pos2(rect.right() - 3.0 * scale, rect.top() + 3.0 * scale);
    let mid = egui::pos2(rect.right() - 3.0 * scale, rect.center().y);
    let bottom = egui::pos2(rect.right() - 3.0 * scale, rect.bottom() - 3.0 * scale);
    let branch = egui::pos2(mid_x, root.y);
    ui.painter().line_segment([root, branch], stroke);
    for target in [top, mid, bottom] {
        ui.painter().line_segment([branch, target], stroke);
        ui.painter()
            .circle_filled(target, 1.8 * scale, stroke.color);
    }
    ui.painter().circle_filled(root, 1.8 * scale, stroke.color);
}

fn paint_info_icon(
    ui: &egui::Ui,
    rect: egui::Rect,
    scale: f32,
    stroke: egui::Stroke,
    color: egui::Color32,
) {
    let center = rect.center();
    ui.painter().circle_stroke(center, 6.6 * scale, stroke);
    ui.painter()
        .circle_filled(center + egui::vec2(0.0, -3.3 * scale), 1.0 * scale, color);
    ui.painter().line_segment(
        [
            center + egui::vec2(0.0, -0.3 * scale),
            center + egui::vec2(0.0, 4.4 * scale),
        ],
        egui::Stroke::new(1.55 * scale, color),
    );
}
