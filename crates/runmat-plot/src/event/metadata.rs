//! Figure layout and metadata, with their serialized text, legend and axes forms.

use super::*;

/// Layout metadata describing subplot arrangement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureLayout {
    pub axes_rows: u32,
    pub axes_cols: u32,
    pub axes_indices: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub axes_grid_cells: Vec<Option<crate::plots::figure::AxesGridCell>>,
}

/// Figure-level metadata (title, labels, theming).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub number_title: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub visible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sg_title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sg_title_style: Option<SerializedTextStyle>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y_label: Option<String>,
    pub grid_enabled: bool,
    #[serde(default)]
    pub minor_grid_enabled: bool,
    pub legend_enabled: bool,
    pub colorbar_enabled: bool,
    pub axis_equal: bool,
    #[serde(
        default = "default_data_aspect_ratio",
        skip_serializing_if = "is_default_data_aspect_ratio"
    )]
    pub data_aspect_ratio: [f64; 3],
    #[serde(
        default = "default_data_aspect_ratio_mode",
        skip_serializing_if = "is_auto_data_aspect_ratio_mode"
    )]
    pub data_aspect_ratio_mode: String,
    pub background_rgba: [f32; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub colormap: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_colormap: Option<String>,
    #[serde(default = "default_colormap_length")]
    pub default_colormap_length: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_limits: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z_limits: Option<[f64; 2]>,
    pub legend_entries: Vec<FigureLegendEntry>,
    #[serde(default)]
    pub active_axes_index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub axes_metadata: Option<Vec<SerializedAxesMetadata>>,
}

impl FigureMetadata {
    pub(super) fn from_figure(figure: &Figure) -> Self {
        let legend_entries = figure
            .legend_entries()
            .into_iter()
            .map(FigureLegendEntry::from)
            .collect();

        Self {
            name: figure.name.clone(),
            number_title: figure.number_title,
            visible: figure.visible,
            position: Some(figure.position),
            title: figure.title.clone(),
            sg_title: figure.sg_title.clone(),
            sg_title_style: figure
                .sg_title
                .as_ref()
                .map(|_| figure.sg_title_style.clone().into()),
            x_label: figure.x_label.clone(),
            y_label: figure.y_label.clone(),
            grid_enabled: figure.grid_enabled,
            minor_grid_enabled: figure.minor_grid_enabled,
            legend_enabled: figure.legend_enabled,
            colorbar_enabled: figure.colorbar_enabled,
            axis_equal: figure.axis_equal,
            data_aspect_ratio: figure
                .axes_metadata(figure.active_axes_index)
                .map(|meta| meta.data_aspect_ratio)
                .unwrap_or([1.0, 1.0, 1.0]),
            data_aspect_ratio_mode: figure
                .axes_metadata(figure.active_axes_index)
                .map(|meta| meta.data_aspect_ratio_mode.clone())
                .unwrap_or_else(|| "auto".into()),
            background_rgba: vec4_to_rgba(figure.background_color),
            colormap: Some(figure.colormap.to_serialized_token()),
            default_colormap: Some(figure.default_colormap.to_serialized_token()),
            default_colormap_length: figure.default_colormap_length,
            color_limits: figure.color_limits.map(|(lo, hi)| [lo, hi]),
            z_limits: figure.z_limits.map(|(lo, hi)| [lo, hi]),
            legend_entries,
            active_axes_index: figure.active_axes_index as u32,
            axes_metadata: Some(
                figure
                    .axes_metadata
                    .iter()
                    .cloned()
                    .map(SerializedAxesMetadata::from)
                    .collect(),
            ),
        }
    }
}

pub(super) fn default_true() -> bool {
    true
}

pub(super) fn default_colormap_length() -> usize {
    256
}

pub(super) fn is_true(value: &bool) -> bool {
    *value
}

pub(super) fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedTextStyle {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_rgba: Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_angle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interpreter: Option<String>,
    pub visible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizontal_alignment: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical_alignment: Option<String>,
}

impl Default for SerializedTextStyle {
    fn default() -> Self {
        TextStyle::default().into()
    }
}

impl From<TextStyle> for SerializedTextStyle {
    fn from(value: TextStyle) -> Self {
        Self {
            color_rgba: value.color.map(vec4_to_rgba),
            font_size: value.font_size,
            font_weight: value.font_weight,
            font_angle: value.font_angle,
            interpreter: value.interpreter,
            visible: value.visible,
            horizontal_alignment: value.horizontal_alignment,
            vertical_alignment: value.vertical_alignment,
        }
    }
}

impl From<SerializedTextStyle> for TextStyle {
    fn from(value: SerializedTextStyle) -> Self {
        Self {
            color: value.color_rgba.map(rgba_to_vec4),
            font_size: value.font_size,
            font_weight: value.font_weight,
            font_angle: value.font_angle,
            interpreter: value.interpreter,
            visible: value.visible,
            horizontal_alignment: value.horizontal_alignment,
            vertical_alignment: value.vertical_alignment,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedLegendStyle {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    pub visible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_angle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interpreter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub box_visible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_color_rgba: Option<[f32; 4]>,
}

impl From<LegendStyle> for SerializedLegendStyle {
    fn from(value: LegendStyle) -> Self {
        Self {
            location: value.location,
            visible: value.visible,
            font_size: value.font_size,
            font_weight: value.font_weight,
            font_angle: value.font_angle,
            interpreter: value.interpreter,
            box_visible: value.box_visible,
            orientation: value.orientation,
            text_color_rgba: value.text_color.map(vec4_to_rgba),
        }
    }
}

impl From<SerializedLegendStyle> for LegendStyle {
    fn from(value: SerializedLegendStyle) -> Self {
        Self {
            location: value.location,
            visible: value.visible,
            font_size: value.font_size,
            font_weight: value.font_weight,
            font_angle: value.font_angle,
            interpreter: value.interpreter,
            box_visible: value.box_visible,
            orientation: value.orientation,
            text_color: value.text_color_rgba.map(rgba_to_vec4),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedAxesMetadata {
    #[serde(default)]
    pub y_direction: crate::plots::figure::YDirection,
    #[serde(default, skip_serializing_if = "is_cartesian_axes_kind")]
    pub axes_kind: SerializedAxesKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay_parent: Option<usize>,
    #[serde(
        default = "default_y_axis_location",
        skip_serializing_if = "is_left_axis"
    )]
    pub y_axis_location: String,
    #[serde(
        default = "default_axes_position",
        skip_serializing_if = "is_default_axes_position"
    )]
    pub position: [f64; 4],
    #[serde(default, skip_serializing_if = "is_false")]
    pub position_explicit: bool,
    #[serde(
        default = "default_axes_units",
        skip_serializing_if = "is_normalized_units"
    )]
    pub units: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_ticks: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_ticks: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_tick_labels: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_tick_labels: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_tick_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_tick_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_tick_label_rotation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_tick_label_rotation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_limits: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y_limits: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z_limits: Option<[f64; 2]>,
    #[serde(
        default,
        skip_serializing_if = "crate::plots::figure::AxisLimitMethod::is_default"
    )]
    pub x_limit_method: crate::plots::figure::AxisLimitMethod,
    #[serde(
        default,
        skip_serializing_if = "crate::plots::figure::AxisLimitMethod::is_default"
    )]
    pub y_limit_method: crate::plots::figure::AxisLimitMethod,
    #[serde(
        default,
        skip_serializing_if = "crate::plots::figure::AxisLimitMethod::is_default"
    )]
    pub z_limit_method: crate::plots::figure::AxisLimitMethod,
    #[serde(default)]
    pub x_log: bool,
    #[serde(default)]
    pub y_log: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub view_azimuth_deg: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub view_elevation_deg: Option<f32>,
    #[serde(default)]
    pub grid_enabled: bool,
    #[serde(default)]
    pub minor_grid_enabled: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub minor_grid_explicit: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub hidden_line_removal: bool,
    #[serde(default)]
    pub box_enabled: bool,
    #[serde(default)]
    pub axis_equal: bool,
    #[serde(
        default = "default_data_aspect_ratio",
        skip_serializing_if = "is_default_data_aspect_ratio"
    )]
    pub data_aspect_ratio: [f64; 3],
    #[serde(
        default = "default_data_aspect_ratio_mode",
        skip_serializing_if = "is_auto_data_aspect_ratio_mode"
    )]
    pub data_aspect_ratio_mode: String,
    pub legend_enabled: bool,
    #[serde(default)]
    pub colorbar_enabled: bool,
    pub colormap: String,
    #[serde(default = "default_colormap_length")]
    pub colormap_length: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_order: Option<Vec<[f32; 3]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_limits: Option<[f64; 2]>,
    #[serde(default)]
    pub axes_style: SerializedTextStyle,
    pub title_style: SerializedTextStyle,
    #[serde(default)]
    pub subtitle_style: SerializedTextStyle,
    pub x_label_style: SerializedTextStyle,
    pub y_label_style: SerializedTextStyle,
    pub z_label_style: SerializedTextStyle,
    pub legend_style: SerializedLegendStyle,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub world_text_annotations: Vec<SerializedTextAnnotation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum SerializedAxesKind {
    #[default]
    Cartesian,
    Polar,
}

pub(super) fn is_cartesian_axes_kind(value: &SerializedAxesKind) -> bool {
    *value == SerializedAxesKind::Cartesian
}

pub(super) fn default_y_axis_location() -> String {
    "left".into()
}

pub(super) fn is_left_axis(value: &str) -> bool {
    value == "left"
}

pub(super) fn default_axes_position() -> [f64; 4] {
    [0.13, 0.11, 0.775, 0.815]
}

pub(super) fn is_default_axes_position(value: &[f64; 4]) -> bool {
    *value == default_axes_position()
}

pub(super) fn default_data_aspect_ratio() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}

pub(super) fn is_default_data_aspect_ratio(value: &[f64; 3]) -> bool {
    *value == default_data_aspect_ratio()
}

pub(super) fn default_data_aspect_ratio_mode() -> String {
    "auto".into()
}

pub(super) fn is_auto_data_aspect_ratio_mode(value: &str) -> bool {
    value == "auto"
}

pub(super) fn default_axes_units() -> String {
    "normalized".into()
}

pub(super) fn is_normalized_units(value: &str) -> bool {
    value == "normalized"
}

impl From<AxesKind> for SerializedAxesKind {
    fn from(value: AxesKind) -> Self {
        match value {
            AxesKind::Cartesian => Self::Cartesian,
            AxesKind::Polar => Self::Polar,
        }
    }
}

impl From<SerializedAxesKind> for AxesKind {
    fn from(value: SerializedAxesKind) -> Self {
        match value {
            SerializedAxesKind::Cartesian => Self::Cartesian,
            SerializedAxesKind::Polar => Self::Polar,
        }
    }
}
