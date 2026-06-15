use crate::core::CameraViewPreset;
use crate::geometry_scene::{
    GeometrySceneAssemblyNode, GeometrySceneCompleteness, GeometrySceneOverlay,
    GeometrySceneRegionSummary,
};
use std::collections::BTreeSet;

use super::product_ui::{self, ProductToolbarIcon, ProductToolbarTheme};

const STRIP_HEIGHT: f32 = 36.0;
const PANEL_CORNER_RADIUS: f32 = 2.0;
const TREE_ROW_HEIGHT: f32 = 22.0;

#[derive(Debug, Clone, Default)]
pub(crate) struct CadOverlayActions {
    pub reset_view: bool,
    pub view_preset: Option<CameraViewPreset>,
    pub grid_enabled: Option<bool>,
    pub xray_enabled: Option<bool>,
    pub owner_visibility: Vec<(String, bool)>,
    pub create_fea_study: bool,
}

impl CadOverlayActions {
    pub fn merge(&mut self, other: Self) {
        self.reset_view |= other.reset_view;
        self.create_fea_study |= other.create_fea_study;
        if other.view_preset.is_some() {
            self.view_preset = other.view_preset;
        }
        if other.grid_enabled.is_some() {
            self.grid_enabled = other.grid_enabled;
        }
        if other.xray_enabled.is_some() {
            self.xray_enabled = other.xray_enabled;
        }
        self.owner_visibility.extend(other.owner_visibility);
    }
}

#[derive(Debug)]
pub(crate) struct CadOverlayState {
    show_tree: bool,
    show_details: bool,
    show_diagnostics: bool,
    selected_node_id: Option<String>,
    selected_region_id: Option<String>,
    expanded_nodes: BTreeSet<String>,
    show_view_menu: bool,
    view_menu_anchor: Option<egui::Rect>,
    pointer_captured: bool,
    capture_regions: Vec<egui::Rect>,
}

impl Default for CadOverlayState {
    fn default() -> Self {
        Self {
            show_tree: false,
            show_details: false,
            show_diagnostics: false,
            selected_node_id: None,
            selected_region_id: None,
            expanded_nodes: BTreeSet::new(),
            show_view_menu: false,
            view_menu_anchor: None,
            pointer_captured: false,
            capture_regions: Vec::new(),
        }
    }
}

impl CadOverlayState {
    pub(crate) fn pointer_captured(&self) -> bool {
        self.pointer_captured
    }

    pub(crate) fn capture_regions(&self) -> &[egui::Rect] {
        &self.capture_regions
    }

    pub fn render(
        &mut self,
        ctx: &egui::Context,
        plot_rect: egui::Rect,
        overlay: &GeometrySceneOverlay,
        grid_enabled: bool,
        xray_enabled: bool,
        owner_visible: &dyn Fn(&str) -> bool,
        font_scale: f32,
    ) -> CadOverlayActions {
        let scale = font_scale.clamp(0.9, 1.0);
        let theme = CadOverlayTheme::from_context(ctx);
        let mut actions = CadOverlayActions::default();
        self.pointer_captured = false;
        self.capture_regions.clear();

        self.render_top_strip(
            ctx,
            plot_rect,
            overlay,
            grid_enabled,
            xray_enabled,
            scale,
            &theme,
            &mut actions,
        );
        self.render_view_menu(ctx, plot_rect, scale, &theme, &mut actions);

        let panel_top = plot_rect.min.y + strip_height(scale) + 12.0 * scale;
        if self.show_details {
            self.render_details_panel(ctx, plot_rect, panel_top, overlay, scale, &theme);
        }
        if self.show_tree && (!overlay.assembly_nodes.is_empty() || !overlay.regions.is_empty()) {
            self.render_assembly_tree(
                ctx,
                plot_rect,
                panel_top,
                overlay,
                owner_visible,
                scale,
                &theme,
                &mut actions,
            );
        }

        actions
    }

    #[allow(clippy::too_many_arguments)]
    fn render_top_strip(
        &mut self,
        ctx: &egui::Context,
        plot_rect: egui::Rect,
        overlay: &GeometrySceneOverlay,
        grid_enabled: bool,
        xray_enabled: bool,
        scale: f32,
        theme: &CadOverlayTheme,
        actions: &mut CadOverlayActions,
    ) {
        let height = strip_height(scale);
        egui::Area::new(egui::Id::new("runmat_cad_top_strip"))
            .order(egui::Order::Foreground)
            .fixed_pos(plot_rect.min)
            .show(ctx, |ui| {
                let (strip_rect, strip_response) = ui.allocate_exact_size(
                    egui::vec2(plot_rect.width(), height),
                    egui::Sense::hover(),
                );
                self.capture_regions.push(strip_rect);
                self.pointer_captured |= strip_response.hovered();
                ui.painter().rect_filled(strip_rect, 0.0, theme.strip_bg);
                ui.painter().line_segment(
                    [strip_rect.left_bottom(), strip_rect.right_bottom()],
                    egui::Stroke::new(1.0, theme.strip_border),
                );

                if matches!(
                    overlay.status,
                    GeometrySceneCompleteness::BoundedPreview | GeometrySceneCompleteness::Loading
                ) {
                    ctx.request_repaint_after(std::time::Duration::from_millis(16));
                    paint_loading_bar(ui, strip_rect, scale, theme);
                }

                let content_rect = strip_rect;
                ui.allocate_ui_at_rect(content_rect, |ui| {
                    ui.set_min_size(content_rect.size());
                    ui.spacing_mut().item_spacing = egui::vec2(8.0 * scale, 0.0);
                    ui.horizontal(|ui| {
                        ui.set_height(content_rect.height());
                        ui.add_space(12.0 * scale);
                        if overlay.status == GeometrySceneCompleteness::Loading {
                            ui.label(
                                egui::RichText::new(status_line_text(overlay))
                                    .size(12.0 * scale)
                                    .strong()
                                    .color(theme.status_text(overlay.status)),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new(geometry_status_label(overlay.status))
                                    .size(12.0 * scale)
                                    .strong()
                                    .color(theme.status_text(overlay.status)),
                            );
                            ui.label(
                                egui::RichText::new(status_metric_text(overlay))
                                    .size(12.0 * scale)
                                    .color(theme.muted_text),
                            );
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                            if cad_create_study_available(overlay)
                                && product_ui::toolbar_primary_text_button(
                                    ui,
                                    "Create FEA Study",
                                    "Create an FEA study from this geometry",
                                    scale,
                                    &theme.toolbar,
                                )
                                .clicked()
                            {
                                actions.create_fea_study = true;
                                ctx.request_repaint();
                            }
                            if strip_icon_button(
                                ui,
                                ProductToolbarIcon::Info,
                                "Geometry info",
                                self.show_details,
                                scale,
                                theme,
                            )
                            .clicked()
                            {
                                self.show_details = !self.show_details;
                                ctx.request_repaint();
                            }
                            if strip_icon_button(
                                ui,
                                ProductToolbarIcon::Tree,
                                "Assembly tree",
                                self.show_tree,
                                scale,
                                theme,
                            )
                            .clicked()
                            {
                                self.show_tree = !self.show_tree;
                                ctx.request_repaint();
                            }
                            let mut xray = xray_enabled;
                            if strip_icon_button(
                                ui,
                                ProductToolbarIcon::Xray,
                                "X-ray view",
                                xray,
                                scale,
                                theme,
                            )
                            .clicked()
                            {
                                xray = !xray;
                                actions.xray_enabled = Some(xray);
                                ctx.request_repaint();
                            }
                            let mut grid = grid_enabled;
                            if strip_icon_button(
                                ui,
                                ProductToolbarIcon::Grid,
                                "Show grid",
                                grid,
                                scale,
                                theme,
                            )
                            .clicked()
                            {
                                grid = !grid;
                                actions.grid_enabled = Some(grid);
                                ctx.request_repaint();
                            }
                            let view_response = strip_icon_button(
                                ui,
                                ProductToolbarIcon::View,
                                "View orientation",
                                self.show_view_menu,
                                scale,
                                theme,
                            );
                            if view_response.clicked() {
                                self.show_view_menu = !self.show_view_menu;
                                self.view_menu_anchor = Some(view_response.rect);
                                ctx.request_repaint();
                            }
                            if strip_icon_button(
                                ui,
                                ProductToolbarIcon::Reset,
                                "Reset view",
                                false,
                                scale,
                                theme,
                            )
                            .clicked()
                            {
                                actions.reset_view = true;
                                ctx.request_repaint();
                            }
                        });
                    });
                });
            });
    }

    fn render_view_menu(
        &mut self,
        ctx: &egui::Context,
        plot_rect: egui::Rect,
        scale: f32,
        theme: &CadOverlayTheme,
        actions: &mut CadOverlayActions,
    ) {
        if !self.show_view_menu {
            return;
        }

        let panel_width = 148.0 * scale;
        let anchor = self.view_menu_anchor.unwrap_or_else(|| {
            egui::Rect::from_min_size(
                egui::pos2(plot_rect.max.x - panel_width - 4.0 * scale, plot_rect.min.y),
                egui::vec2(44.0 * scale, strip_height(scale)),
            )
        });
        let x = anchor.left().clamp(
            plot_rect.min.x + 8.0 * scale,
            plot_rect.max.x - panel_width - 8.0 * scale,
        );
        let pos = egui::pos2(x, plot_rect.min.y + strip_height(scale) + 1.0);

        let response = egui::Area::new(egui::Id::new("runmat_cad_view_menu"))
            .order(egui::Order::Foreground)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                panel_frame(theme, scale).show(ui, |ui| {
                    ui.set_width(panel_width);
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                    for (label, preset) in [
                        ("Perspective", CameraViewPreset::Perspective),
                        ("Top", CameraViewPreset::Top),
                        ("Bottom", CameraViewPreset::Bottom),
                        ("Front", CameraViewPreset::Front),
                        ("Back", CameraViewPreset::Back),
                        ("Left", CameraViewPreset::Left),
                        ("Right", CameraViewPreset::Right),
                    ] {
                        if view_menu_row(ui, label, scale, theme).clicked() {
                            actions.view_preset = Some(preset);
                            self.show_view_menu = false;
                            ctx.request_repaint();
                        }
                    }
                });
            });
        self.capture_regions.push(response.response.rect);
        self.pointer_captured |= response.response.hovered();
    }

    fn render_details_panel(
        &mut self,
        ctx: &egui::Context,
        plot_rect: egui::Rect,
        panel_top: f32,
        overlay: &GeometrySceneOverlay,
        scale: f32,
        theme: &CadOverlayTheme,
    ) {
        let panel_width = (plot_rect.width() * 0.27).clamp(280.0 * scale, 390.0 * scale);
        let pos = egui::pos2(plot_rect.min.x + 12.0 * scale, panel_top);
        let response = egui::Area::new(egui::Id::new("runmat_cad_details"))
            .order(egui::Order::Foreground)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                panel_frame(theme, scale).show(ui, |ui| {
                    ui.set_width(panel_width);
                    ui.spacing_mut().item_spacing = egui::vec2(8.0 * scale, 7.0 * scale);
                    panel_header(ui, "Geometry", scale, theme, || {
                        self.show_details = false;
                    });
                    ui.separator();
                    ui.horizontal_wrapped(|ui| {
                        stat_column(ui, "Meshes", &overlay.mesh_count.to_string(), scale, theme);
                        stat_column(
                            ui,
                            "Triangles",
                            &format_count(overlay.triangle_count as u64),
                            scale,
                            theme,
                        );
                        stat_column(
                            ui,
                            "Regions",
                            &format_mapped_count(overlay.region_count, overlay.mapped_region_count),
                            scale,
                            theme,
                        );
                        if let Some(bytes) = overlay.byte_count {
                            stat_column(ui, "Size", &format_bytes(bytes), scale, theme);
                        }
                    });

                    ui.separator();
                    ui.horizontal_wrapped(|ui| {
                        if let Some(format) = overlay.format.as_deref() {
                            ui.label(detail_text(format!("Format {format}"), scale, theme));
                        }
                        if let Some(source) = overlay.source_label.as_deref() {
                            ui.label(detail_text(source.to_string(), scale, theme));
                        }
                        if let Some(quality) = overlay.quality_label.as_deref() {
                            ui.label(detail_text(format!("Quality {quality}"), scale, theme));
                        }
                    });

                    if let Some(selected) = self.selected_node_label(overlay) {
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!(
                                "Selected {}",
                                truncate_label(selected, 44)
                            ))
                            .size(10.5 * scale)
                            .color(theme.primary_text),
                        );
                    }

                    if !overlay.warnings.is_empty()
                        && panel_button(
                            ui,
                            &format!("Diagnostics ({})", overlay.warnings.len()),
                            self.show_diagnostics,
                            scale,
                            theme,
                        )
                        .clicked()
                    {
                        self.show_diagnostics = !self.show_diagnostics;
                    }
                    if self.show_diagnostics {
                        for warning in overlay.warnings.iter().take(4) {
                            ui.label(detail_text(truncate_label(warning, 78), scale, theme));
                        }
                    }
                });
            });
        self.capture_regions.push(response.response.rect);
        self.pointer_captured |= response.response.hovered();
    }

    #[allow(clippy::too_many_arguments)]
    fn render_assembly_tree(
        &mut self,
        ctx: &egui::Context,
        plot_rect: egui::Rect,
        panel_top: f32,
        overlay: &GeometrySceneOverlay,
        owner_visible: &dyn Fn(&str) -> bool,
        scale: f32,
        theme: &CadOverlayTheme,
        actions: &mut CadOverlayActions,
    ) {
        let panel_width = (plot_rect.width() * 0.28).clamp(300.0 * scale, 420.0 * scale);
        let panel_height = (plot_rect.height() - strip_height(scale) - 32.0 * scale)
            .clamp(180.0 * scale, 440.0 * scale);
        let pos = egui::pos2(plot_rect.max.x - panel_width - 12.0 * scale, panel_top);
        let response = egui::Area::new(egui::Id::new("runmat_cad_assembly_tree"))
            .order(egui::Order::Foreground)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                panel_frame(theme, scale).show(ui, |ui| {
                    ui.set_width(panel_width);
                    ui.set_height(panel_height);
                    panel_header(ui, "Assembly", scale, theme, || {
                        self.show_tree = false;
                    });
                    ui.add_space(3.0 * scale);
                    egui::ScrollArea::vertical()
                        .id_source("runmat_cad_assembly_tree_scroll")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            if overlay.assembly_nodes.is_empty() {
                                ui.label(detail_text("No assembly tree".to_string(), scale, theme));
                            } else {
                                section_label(ui, "Components", scale, theme);
                                for node in &overlay.assembly_nodes {
                                    self.render_tree_node(
                                        ui,
                                        node,
                                        0,
                                        owner_visible,
                                        scale,
                                        theme,
                                        actions,
                                    );
                                }
                            }

                            if !overlay.regions.is_empty() {
                                ui.add_space(8.0 * scale);
                                section_label(ui, "Regions", scale, theme);
                                let row_height = TREE_ROW_HEIGHT * scale;
                                egui::ScrollArea::vertical()
                                    .id_source("runmat_cad_region_list")
                                    .max_height(210.0 * scale)
                                    .auto_shrink([false, false])
                                    .show_rows(
                                        ui,
                                        row_height,
                                        overlay.regions.len(),
                                        |ui, range| {
                                            for index in range {
                                                if let Some(region) = overlay.regions.get(index) {
                                                    self.render_region_row(
                                                        ui, region, scale, theme,
                                                    );
                                                }
                                            }
                                        },
                                    );
                            }
                        });
                });
            });
        self.capture_regions.push(response.response.rect);
        self.pointer_captured |= response.response.hovered();
    }

    #[allow(clippy::too_many_arguments)]
    fn render_tree_node(
        &mut self,
        ui: &mut egui::Ui,
        node: &GeometrySceneAssemblyNode,
        depth: usize,
        owner_visible: &dyn Fn(&str) -> bool,
        scale: f32,
        theme: &CadOverlayTheme,
        actions: &mut CadOverlayActions,
    ) {
        let id = if node.node_id.is_empty() {
            node.label.clone()
        } else {
            node.node_id.clone()
        };
        let has_children = !node.children.is_empty();
        let mut expanded = has_children && (depth == 0 || self.expanded_nodes.contains(&id));
        let selected = self.selected_node_id.as_deref() == Some(id.as_str());
        let visibility = assembly_visibility_state(node, owner_visible);

        let row_response = ui
            .horizontal(|ui| {
                ui.set_height(TREE_ROW_HEIGHT * scale);
                ui.add_space(depth as f32 * 14.0 * scale);
                if has_children {
                    let chevron = if expanded { "v" } else { ">" };
                    if tree_icon_button(ui, chevron, scale, theme).clicked() {
                        expanded = !expanded;
                        if expanded {
                            self.expanded_nodes.insert(id.clone());
                        } else {
                            self.expanded_nodes.remove(&id);
                        }
                    }
                } else {
                    ui.add_space(22.0 * scale);
                }

                if visibility_checkbox(ui, visibility, scale, theme).clicked() {
                    actions
                        .owner_visibility
                        .push((id.clone(), !visibility.is_fully_visible()));
                }

                let label = egui::RichText::new(truncate_label(&node.label, 38))
                    .size(11.5 * scale)
                    .color(if selected {
                        theme.primary_text
                    } else {
                        theme.muted_text
                    });
                ui.selectable_label(selected, label)
            })
            .inner;
        if row_response.clicked() {
            self.selected_node_id = Some(id.clone());
            self.selected_region_id = None;
        }

        if has_children && expanded {
            for child in &node.children {
                self.render_tree_node(ui, child, depth + 1, owner_visible, scale, theme, actions);
            }
        }
    }

    fn render_region_row(
        &mut self,
        ui: &mut egui::Ui,
        region: &GeometrySceneRegionSummary,
        scale: f32,
        theme: &CadOverlayTheme,
    ) {
        let selected = self.selected_region_id.as_deref() == Some(region.region_id.as_str());
        let response = ui
            .horizontal(|ui| {
                ui.set_height(TREE_ROW_HEIGHT * scale);
                ui.add_space(4.0 * scale);
                region_icon(ui, scale, theme);
                let label = region_label(region);
                let text = egui::RichText::new(truncate_label(&label, 36))
                    .size(11.5 * scale)
                    .color(if selected {
                        theme.primary_text
                    } else {
                        theme.muted_text
                    });
                ui.selectable_label(selected, text)
            })
            .inner;
        if response.clicked() {
            self.selected_region_id = Some(region.region_id.clone());
            self.selected_node_id = None;
        }
    }

    fn selected_node_label<'a>(&self, overlay: &'a GeometrySceneOverlay) -> Option<&'a str> {
        let selected = self.selected_node_id.as_deref()?;
        overlay
            .assembly_nodes
            .iter()
            .find_map(|node| find_node_label(node, selected))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisibilityState {
    Visible,
    Hidden,
    Mixed,
}

impl VisibilityState {
    fn is_fully_visible(self) -> bool {
        matches!(self, Self::Visible)
    }
}

fn assembly_visibility_state(
    node: &GeometrySceneAssemblyNode,
    owner_visible: &dyn Fn(&str) -> bool,
) -> VisibilityState {
    let id = if node.node_id.is_empty() {
        node.label.as_str()
    } else {
        node.node_id.as_str()
    };
    let own_visible = owner_visible(id);
    if !own_visible {
        return VisibilityState::Hidden;
    }
    if node.children.is_empty() {
        return VisibilityState::Visible;
    }

    let mut saw_visible = false;
    let mut saw_hidden = false;
    for child in &node.children {
        match assembly_visibility_state(child, owner_visible) {
            VisibilityState::Visible => saw_visible = true,
            VisibilityState::Hidden => saw_hidden = true,
            VisibilityState::Mixed => {
                saw_visible = true;
                saw_hidden = true;
            }
        }
    }
    match (saw_visible, saw_hidden) {
        (_, false) => VisibilityState::Visible,
        (false, true) => VisibilityState::Hidden,
        _ => VisibilityState::Mixed,
    }
}

fn region_label(region: &GeometrySceneRegionSummary) -> String {
    let mut label = region.label.clone();
    if label.trim().is_empty() {
        label = region.region_id.clone();
    }
    if let Some(kind) = region.kind.as_deref().filter(|kind| !kind.is_empty()) {
        label = format!("{label}  {kind}");
    }
    if region.triangle_count > 0 {
        label = format!("{label}  {}", format_count(region.triangle_count as u64));
    }
    label
}

fn section_label(ui: &mut egui::Ui, label: &str, scale: f32, theme: &CadOverlayTheme) {
    ui.label(
        egui::RichText::new(label.to_ascii_uppercase())
            .size(9.0 * scale)
            .strong()
            .color(theme.section_text),
    );
    ui.add_space(2.0 * scale);
}

fn tree_icon_button(
    ui: &mut egui::Ui,
    label: &str,
    scale: f32,
    theme: &CadOverlayTheme,
) -> egui::Response {
    ui.add(
        egui::Button::new(
            egui::RichText::new(label)
                .size(10.0 * scale)
                .color(theme.toolbar.text),
        )
        .min_size(egui::vec2(18.0 * scale, 18.0 * scale))
        .fill(egui::Color32::TRANSPARENT)
        .stroke(egui::Stroke::NONE),
    )
}

fn visibility_checkbox(
    ui: &mut egui::Ui,
    state: VisibilityState,
    scale: f32,
    theme: &CadOverlayTheme,
) -> egui::Response {
    let size = egui::vec2(14.0 * scale, 14.0 * scale);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let stroke = egui::Stroke::new(1.0, theme.checkbox_border);
    ui.painter()
        .rect(rect, PANEL_CORNER_RADIUS * scale, theme.checkbox_bg, stroke);
    match state {
        VisibilityState::Visible => {
            let p0 = egui::pos2(rect.left() + 3.0 * scale, rect.center().y);
            let p1 = egui::pos2(rect.left() + 6.0 * scale, rect.bottom() - 3.0 * scale);
            let p2 = egui::pos2(rect.right() - 3.0 * scale, rect.top() + 3.0 * scale);
            ui.painter()
                .line_segment([p0, p1], egui::Stroke::new(1.6, theme.checkbox_mark));
            ui.painter()
                .line_segment([p1, p2], egui::Stroke::new(1.6, theme.checkbox_mark));
        }
        VisibilityState::Mixed => {
            let y = rect.center().y;
            ui.painter().line_segment(
                [
                    egui::pos2(rect.left() + 3.0 * scale, y),
                    egui::pos2(rect.right() - 3.0 * scale, y),
                ],
                egui::Stroke::new(1.6, theme.checkbox_mark),
            );
        }
        VisibilityState::Hidden => {}
    }
    response
}

fn region_icon(ui: &mut egui::Ui, scale: f32, theme: &CadOverlayTheme) {
    let size = egui::vec2(14.0 * scale, 14.0 * scale);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let center = rect.center();
    ui.painter().circle_stroke(
        center,
        4.5 * scale,
        egui::Stroke::new(1.0, theme.checkbox_border),
    );
    ui.painter()
        .circle_filled(center, 2.0 * scale, theme.region_dot);
}

#[derive(Debug, Clone, Copy)]
struct CadOverlayTheme {
    strip_bg: egui::Color32,
    strip_border: egui::Color32,
    panel_bg: egui::Color32,
    panel_border: egui::Color32,
    primary_text: egui::Color32,
    muted_text: egui::Color32,
    toolbar: ProductToolbarTheme,
    section_text: egui::Color32,
    checkbox_bg: egui::Color32,
    checkbox_border: egui::Color32,
    checkbox_mark: egui::Color32,
    region_dot: egui::Color32,
    preview_text: egui::Color32,
    complete_text: egui::Color32,
    failed_text: egui::Color32,
}

impl CadOverlayTheme {
    fn from_context(ctx: &egui::Context) -> Self {
        if ctx.style().visuals.dark_mode {
            Self {
                strip_bg: egui::Color32::from_rgba_unmultiplied(11, 13, 18, 234),
                strip_border: egui::Color32::from_rgba_unmultiplied(52, 58, 68, 210),
                panel_bg: egui::Color32::from_rgba_unmultiplied(12, 15, 22, 214),
                panel_border: egui::Color32::from_rgba_unmultiplied(136, 150, 170, 92),
                primary_text: egui::Color32::from_rgb(226, 232, 240),
                muted_text: egui::Color32::from_rgb(157, 168, 184),
                toolbar: ProductToolbarTheme::from_context(ctx),
                section_text: egui::Color32::from_rgb(126, 138, 156),
                checkbox_bg: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 10),
                checkbox_border: egui::Color32::from_rgba_unmultiplied(180, 194, 214, 124),
                checkbox_mark: egui::Color32::from_rgb(139, 223, 176),
                region_dot: egui::Color32::from_rgb(139, 180, 255),
                preview_text: egui::Color32::from_rgb(238, 196, 94),
                complete_text: egui::Color32::from_rgb(125, 211, 158),
                failed_text: egui::Color32::from_rgb(239, 138, 118),
            }
        } else {
            Self {
                strip_bg: egui::Color32::from_rgba_unmultiplied(246, 248, 251, 244),
                strip_border: egui::Color32::from_rgba_unmultiplied(196, 205, 218, 230),
                panel_bg: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 226),
                panel_border: egui::Color32::from_rgba_unmultiplied(148, 163, 184, 120),
                primary_text: egui::Color32::from_rgb(24, 31, 42),
                muted_text: egui::Color32::from_rgb(91, 103, 119),
                toolbar: ProductToolbarTheme::from_context(ctx),
                section_text: egui::Color32::from_rgb(92, 105, 123),
                checkbox_bg: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210),
                checkbox_border: egui::Color32::from_rgba_unmultiplied(110, 126, 146, 156),
                checkbox_mark: egui::Color32::from_rgb(37, 124, 76),
                region_dot: egui::Color32::from_rgb(61, 111, 190),
                preview_text: egui::Color32::from_rgb(158, 112, 18),
                complete_text: egui::Color32::from_rgb(36, 128, 73),
                failed_text: egui::Color32::from_rgb(170, 62, 46),
            }
        }
    }

    fn status_text(&self, status: GeometrySceneCompleteness) -> egui::Color32 {
        match status {
            GeometrySceneCompleteness::Complete => self.complete_text,
            GeometrySceneCompleteness::Loading => self.preview_text,
            GeometrySceneCompleteness::BoundedPreview => self.preview_text,
            GeometrySceneCompleteness::FailedComplete => self.failed_text,
        }
    }
}

fn strip_height(scale: f32) -> f32 {
    STRIP_HEIGHT * scale
}

fn panel_frame(theme: &CadOverlayTheme, scale: f32) -> egui::Frame {
    egui::Frame::none()
        .fill(theme.panel_bg)
        .stroke(egui::Stroke::new(1.0, theme.panel_border))
        .rounding(egui::Rounding::same(PANEL_CORNER_RADIUS * scale))
        .inner_margin(egui::Margin::same(10.0 * scale))
}

fn panel_header(
    ui: &mut egui::Ui,
    title: &str,
    scale: f32,
    theme: &CadOverlayTheme,
    mut close: impl FnMut(),
) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(title)
                .strong()
                .size(12.0 * scale)
                .color(theme.primary_text),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if panel_button(ui, "Close", false, scale, theme).clicked() {
                close();
            }
        });
    });
}

fn strip_icon_button(
    ui: &mut egui::Ui,
    icon: ProductToolbarIcon,
    tooltip: &'static str,
    active: bool,
    scale: f32,
    theme: &CadOverlayTheme,
) -> egui::Response {
    product_ui::toolbar_icon_button(ui, icon, tooltip, active, scale, &theme.toolbar)
}

fn panel_button(
    ui: &mut egui::Ui,
    label: &str,
    active: bool,
    scale: f32,
    theme: &CadOverlayTheme,
) -> egui::Response {
    product_ui::toolbar_text_button(ui, label, active, scale, &theme.toolbar, 0.0)
}

fn view_menu_row(
    ui: &mut egui::Ui,
    label: &str,
    scale: f32,
    theme: &CadOverlayTheme,
) -> egui::Response {
    let size = egui::vec2(ui.available_width().max(1.0), 24.0 * scale);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    if response.hovered() {
        ui.painter()
            .rect_filled(rect, 0.0, theme.toolbar.hover_fill);
    }
    ui.painter().text(
        rect.left_center() + egui::vec2(8.0 * scale, 0.0),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(11.0 * scale),
        theme.primary_text,
    );
    response
}

fn paint_loading_bar(ui: &egui::Ui, strip_rect: egui::Rect, scale: f32, theme: &CadOverlayTheme) {
    let height = 2.0_f32.max(2.0 * scale);
    let rect = egui::Rect::from_min_max(
        egui::pos2(strip_rect.min.x, strip_rect.max.y - height),
        strip_rect.max,
    );
    ui.painter().rect_filled(
        rect,
        0.0,
        egui::Color32::from_rgba_unmultiplied(242, 198, 91, 42),
    );
    let time = ui.ctx().input(|input| input.time as f32);
    let width = rect.width();
    let segment_width = width * 0.28;
    let phase = (time * 0.55).fract();
    let x0 = rect.min.x + (width + segment_width) * phase - segment_width;
    let x1 = (x0 + segment_width).min(rect.max.x);
    let x0 = x0.max(rect.min.x);
    if x1 > x0 {
        ui.painter().rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, rect.min.y), egui::pos2(x1, rect.max.y)),
            0.0,
            theme.preview_text,
        );
    }
}

fn status_metric_text(overlay: &GeometrySceneOverlay) -> String {
    match overlay.status {
        GeometrySceneCompleteness::Loading => {
            let loaded = format_count(overlay.triangle_count as u64);
            if let Some(percent) = overlay.progress_percent {
                format!(
                    "geometry. {loaded} triangles loaded ({})",
                    format_percent(percent)
                )
            } else {
                format!("geometry. {loaded} triangles loaded")
            }
        }
        GeometrySceneCompleteness::BoundedPreview => {
            format!(
                "{} tris shown; bounded preview",
                format_count(overlay.triangle_count as u64)
            )
        }
        GeometrySceneCompleteness::Complete => {
            format!(
                "{} tris, {} regions",
                format_count(overlay.triangle_count as u64),
                format_count(overlay.region_count as u64)
            )
        }
        GeometrySceneCompleteness::FailedComplete => {
            format!(
                "{} tris available; complete load failed",
                format_count(overlay.triangle_count as u64)
            )
        }
    }
}

fn status_line_text(overlay: &GeometrySceneOverlay) -> String {
    format!(
        "{} {}",
        geometry_status_label(overlay.status),
        status_metric_text(overlay)
    )
}

fn cad_create_study_available(overlay: &GeometrySceneOverlay) -> bool {
    overlay.allow_create_fea_study
        && overlay
            .source_label
            .as_deref()
            .map(|label| label.starts_with("cad /"))
            .unwrap_or(false)
}

fn stat_column(ui: &mut egui::Ui, label: &str, value: &str, scale: f32, theme: &CadOverlayTheme) {
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new(label.to_ascii_uppercase())
                .size(8.0 * scale)
                .color(theme.muted_text),
        );
        ui.label(
            egui::RichText::new(value)
                .monospace()
                .size(11.0 * scale)
                .color(theme.primary_text),
        );
    });
}

fn detail_text(text: String, scale: f32, theme: &CadOverlayTheme) -> egui::RichText {
    egui::RichText::new(text)
        .monospace()
        .size(10.0 * scale)
        .color(theme.muted_text)
}

fn geometry_status_label(status: GeometrySceneCompleteness) -> &'static str {
    match status {
        GeometrySceneCompleteness::Complete => "Complete",
        GeometrySceneCompleteness::Loading => "Loading",
        GeometrySceneCompleteness::BoundedPreview => "Preview",
        GeometrySceneCompleteness::FailedComplete => "Limited",
    }
}

fn format_percent(value: f64) -> String {
    format!("{:.0}%", (value.clamp(0.0, 1.0) * 100.0).round())
}

fn find_node_label<'a>(node: &'a GeometrySceneAssemblyNode, id: &str) -> Option<&'a str> {
    if node.node_id == id || (node.node_id.is_empty() && node.label == id) {
        return Some(node.label.as_str());
    }
    node.children
        .iter()
        .find_map(|child| find_node_label(child, id))
}

fn truncate_label(label: &str, max_len: usize) -> String {
    if label.chars().count() <= max_len {
        return label.to_string();
    }
    let mut out = String::new();
    for (i, ch) in label.chars().enumerate() {
        if i >= max_len.saturating_sub(1) {
            break;
        }
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn format_count(value: u64) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    for (index, ch) in text.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn format_mapped_count(total: usize, mapped: usize) -> String {
    if mapped == 0 {
        format_count(total as u64)
    } else {
        format!(
            "{} / {}",
            format_count(mapped as u64),
            format_count(total as u64)
        )
    }
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let value = bytes as f64;
    if value >= GIB {
        format!("{:.1} GB", value / GIB)
    } else if value >= MIB {
        format!("{:.1} MB", value / MIB)
    } else if value >= KIB {
        format!("{:.1} KB", value / KIB)
    } else {
        format!("{bytes} B")
    }
}
