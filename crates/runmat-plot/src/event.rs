use crate::core::{BoundingBox, Vertex};
use crate::plots::{
    AreaPlot, AxesMetadata, BarChart, ColorMap, ContourFillPlot, ContourPlot, ErrorBar, Figure,
    LegendEntry, LegendStyle, Line3Plot, LinePlot, MarkerStyle, PlotElement, PlotType, QuiverPlot,
    Scatter3Plot, ScatterPlot, ShadingMode, StairsPlot, StemPlot, SurfacePlot, TextStyle,
};
use glam::{Vec3, Vec4};
use serde::{Deserialize, Serialize};

/// High-level event emitted whenever a figure changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureEvent {
    pub handle: u32,
    pub kind: FigureEventKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub figure: Option<FigureSnapshot>,
}

/// Event kind for figure lifecycle + updates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FigureEventKind {
    Created,
    Updated,
    Cleared,
    Closed,
}

/// Snapshot of the figure state describing layout + plots.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureSnapshot {
    pub layout: FigureLayout,
    pub metadata: FigureMetadata,
    pub plots: Vec<PlotDescriptor>,
}

/// Full replay scene payload capable of reconstructing a figure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureScene {
    pub schema_version: u32,
    pub layout: FigureLayout,
    pub metadata: FigureMetadata,
    pub plots: Vec<ScenePlot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScenePlot {
    Line {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Scatter {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        color_rgba: [f32; 4],
        marker_size: f32,
        marker_style: String,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Bar {
        labels: Vec<String>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        values: Vec<f64>,
        #[serde(default, deserialize_with = "deserialize_option_vec_f64_lossy")]
        histogram_bin_edges: Option<Vec<f64>>,
        color_rgba: [f32; 4],
        #[serde(default)]
        outline_color_rgba: Option<[f32; 4]>,
        bar_width: f32,
        outline_width: f32,
        orientation: String,
        group_index: u32,
        group_count: u32,
        #[serde(default, deserialize_with = "deserialize_option_vec_f64_lossy")]
        stack_offsets: Option<Vec<f64>>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    ErrorBar {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        err_low: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        err_high: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x_err_low: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x_err_high: Vec<f64>,
        orientation: String,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        cap_width: f32,
        marker_style: Option<String>,
        marker_size: Option<f32>,
        marker_face_color: Option<[f32; 4]>,
        marker_edge_color: Option<[f32; 4]>,
        marker_filled: Option<bool>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Stairs {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        color_rgba: [f32; 4],
        line_width: f32,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Stem {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(deserialize_with = "deserialize_f64_lossy")]
        baseline: f64,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        baseline_color_rgba: [f32; 4],
        baseline_visible: bool,
        marker_color_rgba: [f32; 4],
        marker_size: f32,
        marker_filled: bool,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Area {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(default, deserialize_with = "deserialize_option_vec_f64_lossy")]
        lower_y: Option<Vec<f64>>,
        #[serde(deserialize_with = "deserialize_f64_lossy")]
        baseline: f64,
        color_rgba: [f32; 4],
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Quiver {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        u: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        v: Vec<f64>,
        color_rgba: [f32; 4],
        line_width: f32,
        scale: f32,
        head_size: f32,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Surface {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(deserialize_with = "deserialize_matrix_f64_lossy")]
        z: Vec<Vec<f64>>,
        colormap: String,
        shading_mode: String,
        wireframe: bool,
        alpha: f32,
        flatten_z: bool,
        #[serde(default)]
        image_mode: bool,
        #[serde(default)]
        color_grid_rgba: Option<Vec<Vec<[f32; 4]>>>,
        #[serde(default, deserialize_with = "deserialize_option_pair_f64_lossy")]
        color_limits: Option<[f64; 2]>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Line3 {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        z: Vec<f64>,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Scatter3 {
        #[serde(deserialize_with = "deserialize_vec_xyz_f32_lossy")]
        points: Vec<[f32; 3]>,
        #[serde(default, deserialize_with = "deserialize_vec_rgba_f32_lossy")]
        colors_rgba: Vec<[f32; 4]>,
        point_size: f32,
        #[serde(default, deserialize_with = "deserialize_option_vec_f32_lossy")]
        point_sizes: Option<Vec<f32>>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Contour {
        vertices: Vec<SerializedVertex>,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        base_z: f32,
        line_width: f32,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    ContourFill {
        vertices: Vec<SerializedVertex>,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Pie {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        values: Vec<f64>,
        colors_rgba: Vec<[f32; 4]>,
        slice_labels: Vec<String>,
        label_format: Option<String>,
        explode: Vec<bool>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Unsupported {
        plot_kind: PlotKind,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
}

impl FigureSnapshot {
    /// Capture a snapshot from a [`Figure`] reference.
    pub fn capture(figure: &Figure) -> Self {
        let (rows, cols) = figure.axes_grid();
        let layout = FigureLayout {
            axes_rows: rows as u32,
            axes_cols: cols as u32,
            axes_indices: figure
                .plot_axes_indices()
                .iter()
                .map(|idx| *idx as u32)
                .collect(),
        };

        let metadata = FigureMetadata::from_figure(figure);

        let plots = figure
            .plots()
            .enumerate()
            .map(|(idx, plot)| PlotDescriptor::from_plot(plot, figure_axis_index(figure, idx)))
            .collect();

        Self {
            layout,
            metadata,
            plots,
        }
    }
}

impl FigureScene {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn capture(figure: &Figure) -> Self {
        let snapshot = FigureSnapshot::capture(figure);
        let plots = figure
            .plots()
            .enumerate()
            .map(|(idx, plot)| ScenePlot::from_plot(plot, figure_axis_index(figure, idx)))
            .collect();

        Self {
            schema_version: Self::SCHEMA_VERSION,
            layout: snapshot.layout,
            metadata: snapshot.metadata,
            plots,
        }
    }

    pub fn into_figure(self) -> Result<Figure, String> {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Err(format!(
                "unsupported figure scene schema version {}",
                self.schema_version
            ));
        }

        let mut figure = Figure::new();
        figure.set_subplot_grid(
            self.layout.axes_rows as usize,
            self.layout.axes_cols as usize,
        );
        figure.active_axes_index = self.metadata.active_axes_index as usize;
        if let Some(axes_metadata) = self.metadata.axes_metadata.clone() {
            figure.axes_metadata = axes_metadata.into_iter().map(AxesMetadata::from).collect();
            figure.set_active_axes_index(figure.active_axes_index);
        } else {
            figure.title = self.metadata.title;
            figure.x_label = self.metadata.x_label;
            figure.y_label = self.metadata.y_label;
            figure.legend_enabled = self.metadata.legend_enabled;
        }
        figure.grid_enabled = self.metadata.grid_enabled;
        figure.z_limits = self.metadata.z_limits.map(|[lo, hi]| (lo, hi));
        figure.colorbar_enabled = self.metadata.colorbar_enabled;
        figure.axis_equal = self.metadata.axis_equal;
        figure.background_color = rgba_to_vec4(self.metadata.background_rgba);

        for plot in self.plots {
            plot.apply_to_figure(&mut figure)?;
        }

        Ok(figure)
    }
}

fn figure_axis_index(figure: &Figure, plot_index: usize) -> u32 {
    figure
        .plot_axes_indices()
        .get(plot_index)
        .copied()
        .unwrap_or(0) as u32
}

/// Layout metadata describing subplot arrangement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureLayout {
    pub axes_rows: u32,
    pub axes_cols: u32,
    pub axes_indices: Vec<u32>,
}

/// Figure-level metadata (title, labels, theming).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y_label: Option<String>,
    pub grid_enabled: bool,
    pub legend_enabled: bool,
    pub colorbar_enabled: bool,
    pub axis_equal: bool,
    pub background_rgba: [f32; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub colormap: Option<String>,
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
    fn from_figure(figure: &Figure) -> Self {
        let legend_entries = figure
            .legend_entries()
            .into_iter()
            .map(FigureLegendEntry::from)
            .collect();

        Self {
            title: figure.title.clone(),
            x_label: figure.x_label.clone(),
            y_label: figure.y_label.clone(),
            grid_enabled: figure.grid_enabled,
            legend_enabled: figure.legend_enabled,
            colorbar_enabled: figure.colorbar_enabled,
            axis_equal: figure.axis_equal,
            background_rgba: vec4_to_rgba(figure.background_color),
            colormap: Some(format!("{:?}", figure.colormap)),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_limits: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y_limits: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z_limits: Option<[f64; 2]>,
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
    pub box_enabled: bool,
    #[serde(default)]
    pub axis_equal: bool,
    pub legend_enabled: bool,
    #[serde(default)]
    pub colorbar_enabled: bool,
    pub colormap: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_limits: Option<[f64; 2]>,
    pub title_style: SerializedTextStyle,
    pub x_label_style: SerializedTextStyle,
    pub y_label_style: SerializedTextStyle,
    pub z_label_style: SerializedTextStyle,
    pub legend_style: SerializedLegendStyle,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub world_text_annotations: Vec<SerializedTextAnnotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedTextAnnotation {
    pub position: [f32; 3],
    pub text: String,
    pub style: SerializedTextStyle,
}

impl From<AxesMetadata> for SerializedAxesMetadata {
    fn from(value: AxesMetadata) -> Self {
        Self {
            title: value.title,
            x_label: value.x_label,
            y_label: value.y_label,
            z_label: value.z_label,
            x_limits: value.x_limits.map(|(a, b)| [a, b]),
            y_limits: value.y_limits.map(|(a, b)| [a, b]),
            z_limits: value.z_limits.map(|(a, b)| [a, b]),
            x_log: value.x_log,
            y_log: value.y_log,
            view_azimuth_deg: value.view_azimuth_deg,
            view_elevation_deg: value.view_elevation_deg,
            grid_enabled: value.grid_enabled,
            box_enabled: value.box_enabled,
            axis_equal: value.axis_equal,
            legend_enabled: value.legend_enabled,
            colorbar_enabled: value.colorbar_enabled,
            colormap: format!("{:?}", value.colormap),
            color_limits: value.color_limits.map(|(a, b)| [a, b]),
            title_style: value.title_style.into(),
            x_label_style: value.x_label_style.into(),
            y_label_style: value.y_label_style.into(),
            z_label_style: value.z_label_style.into(),
            legend_style: value.legend_style.into(),
            world_text_annotations: value
                .world_text_annotations
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<SerializedAxesMetadata> for AxesMetadata {
    fn from(value: SerializedAxesMetadata) -> Self {
        Self {
            title: value.title,
            x_label: value.x_label,
            y_label: value.y_label,
            z_label: value.z_label,
            x_limits: value.x_limits.map(|[a, b]| (a, b)),
            y_limits: value.y_limits.map(|[a, b]| (a, b)),
            z_limits: value.z_limits.map(|[a, b]| (a, b)),
            x_log: value.x_log,
            y_log: value.y_log,
            view_azimuth_deg: value.view_azimuth_deg,
            view_elevation_deg: value.view_elevation_deg,
            grid_enabled: value.grid_enabled,
            box_enabled: value.box_enabled,
            axis_equal: value.axis_equal,
            legend_enabled: value.legend_enabled,
            colorbar_enabled: value.colorbar_enabled,
            colormap: parse_colormap_name(&value.colormap),
            color_limits: value.color_limits.map(|[a, b]| (a, b)),
            title_style: value.title_style.into(),
            x_label_style: value.x_label_style.into(),
            y_label_style: value.y_label_style.into(),
            z_label_style: value.z_label_style.into(),
            legend_style: value.legend_style.into(),
            world_text_annotations: value
                .world_text_annotations
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<crate::plots::figure::TextAnnotation> for SerializedTextAnnotation {
    fn from(value: crate::plots::figure::TextAnnotation) -> Self {
        Self {
            position: value.position.to_array(),
            text: value.text,
            style: value.style.into(),
        }
    }
}

impl From<SerializedTextAnnotation> for crate::plots::figure::TextAnnotation {
    fn from(value: SerializedTextAnnotation) -> Self {
        Self {
            position: glam::Vec3::from_array(value.position),
            text: value.text,
            style: value.style.into(),
        }
    }
}

/// Descriptor for a single plot element within the figure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlotDescriptor {
    pub kind: PlotKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub axes_index: u32,
    pub color_rgba: [f32; 4],
    pub visible: bool,
}

impl PlotDescriptor {
    fn from_plot(plot: &PlotElement, axes_index: u32) -> Self {
        Self {
            kind: PlotKind::from(plot.plot_type()),
            label: plot.label(),
            axes_index,
            color_rgba: vec4_to_rgba(plot.color()),
            visible: plot.is_visible(),
        }
    }
}

impl ScenePlot {
    fn from_plot(plot: &PlotElement, axes_index: u32) -> Self {
        match plot {
            PlotElement::Line(line) => Self::Line {
                x: line.x_data.clone(),
                y: line.y_data.clone(),
                color_rgba: vec4_to_rgba(line.color),
                line_width: line.line_width,
                line_style: format!("{:?}", line.line_style),
                axes_index,
                label: line.label.clone(),
                visible: line.visible,
            },
            PlotElement::Scatter(scatter) => Self::Scatter {
                x: scatter.x_data.clone(),
                y: scatter.y_data.clone(),
                color_rgba: vec4_to_rgba(scatter.color),
                marker_size: scatter.marker_size,
                marker_style: format!("{:?}", scatter.marker_style),
                axes_index,
                label: scatter.label.clone(),
                visible: scatter.visible,
            },
            PlotElement::Bar(bar) => Self::Bar {
                labels: bar.labels.clone(),
                values: bar.values().unwrap_or(&[]).to_vec(),
                histogram_bin_edges: bar.histogram_bin_edges().map(|edges| edges.to_vec()),
                color_rgba: vec4_to_rgba(bar.color),
                outline_color_rgba: bar.outline_color.map(vec4_to_rgba),
                bar_width: bar.bar_width,
                outline_width: bar.outline_width,
                orientation: format!("{:?}", bar.orientation),
                group_index: bar.group_index as u32,
                group_count: bar.group_count as u32,
                stack_offsets: bar.stack_offsets().map(|offsets| offsets.to_vec()),
                axes_index,
                label: bar.label.clone(),
                visible: bar.visible,
            },
            PlotElement::ErrorBar(error) => Self::ErrorBar {
                x: error.x.clone(),
                y: error.y.clone(),
                err_low: error.y_neg.clone(),
                err_high: error.y_pos.clone(),
                x_err_low: error.x_neg.clone(),
                x_err_high: error.x_pos.clone(),
                orientation: format!("{:?}", error.orientation),
                color_rgba: vec4_to_rgba(error.color),
                line_width: error.line_width,
                line_style: format!("{:?}", error.line_style),
                cap_width: error.cap_size,
                marker_style: error.marker.as_ref().map(|m| format!("{:?}", m.kind)),
                marker_size: error.marker.as_ref().map(|m| m.size),
                marker_face_color: error.marker.as_ref().map(|m| vec4_to_rgba(m.face_color)),
                marker_edge_color: error.marker.as_ref().map(|m| vec4_to_rgba(m.edge_color)),
                marker_filled: error.marker.as_ref().map(|m| m.filled),
                axes_index,
                label: error.label.clone(),
                visible: error.visible,
            },
            PlotElement::Stairs(stairs) => Self::Stairs {
                x: stairs.x.clone(),
                y: stairs.y.clone(),
                color_rgba: vec4_to_rgba(stairs.color),
                line_width: stairs.line_width,
                axes_index,
                label: stairs.label.clone(),
                visible: stairs.visible,
            },
            PlotElement::Stem(stem) => Self::Stem {
                x: stem.x.clone(),
                y: stem.y.clone(),
                baseline: stem.baseline,
                color_rgba: vec4_to_rgba(stem.color),
                line_width: stem.line_width,
                line_style: format!("{:?}", stem.line_style),
                baseline_color_rgba: vec4_to_rgba(stem.baseline_color),
                baseline_visible: stem.baseline_visible,
                marker_color_rgba: vec4_to_rgba(
                    stem.marker
                        .as_ref()
                        .map(|m| m.face_color)
                        .unwrap_or(stem.color),
                ),
                marker_size: stem.marker.as_ref().map(|m| m.size).unwrap_or(0.0),
                marker_filled: stem.marker.as_ref().map(|m| m.filled).unwrap_or(false),
                axes_index,
                label: stem.label.clone(),
                visible: stem.visible,
            },
            PlotElement::Area(area) => Self::Area {
                x: area.x.clone(),
                y: area.y.clone(),
                lower_y: area.lower_y.clone(),
                baseline: area.baseline,
                color_rgba: vec4_to_rgba(area.color),
                axes_index,
                label: area.label.clone(),
                visible: area.visible,
            },
            PlotElement::Quiver(quiver) => Self::Quiver {
                x: quiver.x.clone(),
                y: quiver.y.clone(),
                u: quiver.u.clone(),
                v: quiver.v.clone(),
                color_rgba: vec4_to_rgba(quiver.color),
                line_width: quiver.line_width,
                scale: quiver.scale,
                head_size: quiver.head_size,
                axes_index,
                label: quiver.label.clone(),
                visible: quiver.visible,
            },
            PlotElement::Surface(surface) => Self::Surface {
                x: surface.x_data.clone(),
                y: surface.y_data.clone(),
                z: surface.z_data.clone().unwrap_or_default(),
                colormap: format!("{:?}", surface.colormap),
                shading_mode: format!("{:?}", surface.shading_mode),
                wireframe: surface.wireframe,
                alpha: surface.alpha,
                flatten_z: surface.flatten_z,
                image_mode: surface.image_mode,
                color_grid_rgba: surface.color_grid.as_ref().map(|grid| {
                    grid.iter()
                        .map(|row| row.iter().map(|color| vec4_to_rgba(*color)).collect())
                        .collect()
                }),
                color_limits: surface.color_limits.map(|(lo, hi)| [lo, hi]),
                axes_index,
                label: surface.label.clone(),
                visible: surface.visible,
            },
            PlotElement::Line3(line) => Self::Line3 {
                x: line.x_data.clone(),
                y: line.y_data.clone(),
                z: line.z_data.clone(),
                color_rgba: vec4_to_rgba(line.color),
                line_width: line.line_width,
                line_style: format!("{:?}", line.line_style),
                axes_index,
                label: line.label.clone(),
                visible: line.visible,
            },
            PlotElement::Scatter3(scatter3) => Self::Scatter3 {
                points: scatter3
                    .points
                    .iter()
                    .map(|point| vec3_to_xyz(*point))
                    .collect(),
                colors_rgba: scatter3
                    .colors
                    .iter()
                    .map(|color| vec4_to_rgba(*color))
                    .collect(),
                point_size: scatter3.point_size,
                point_sizes: scatter3.point_sizes.clone(),
                axes_index,
                label: scatter3.label.clone(),
                visible: scatter3.visible,
            },
            PlotElement::Contour(contour) => Self::Contour {
                vertices: contour
                    .cpu_vertices()
                    .unwrap_or(&[])
                    .iter()
                    .cloned()
                    .map(Into::into)
                    .collect(),
                bounds_min: vec3_to_xyz(contour.bounds().min),
                bounds_max: vec3_to_xyz(contour.bounds().max),
                base_z: contour.base_z,
                line_width: contour.line_width,
                axes_index,
                label: contour.label.clone(),
                visible: contour.visible,
            },
            PlotElement::ContourFill(fill) => Self::ContourFill {
                vertices: fill
                    .cpu_vertices()
                    .unwrap_or(&[])
                    .iter()
                    .cloned()
                    .map(Into::into)
                    .collect(),
                bounds_min: vec3_to_xyz(fill.bounds().min),
                bounds_max: vec3_to_xyz(fill.bounds().max),
                axes_index,
                label: fill.label.clone(),
                visible: fill.visible,
            },
            PlotElement::Pie(pie) => Self::Pie {
                values: pie.values.clone(),
                colors_rgba: pie.colors.iter().map(|c| vec4_to_rgba(*c)).collect(),
                slice_labels: pie.slice_labels.clone(),
                label_format: pie.label_format.clone(),
                explode: pie.explode.clone(),
                axes_index,
                label: pie.label.clone(),
                visible: pie.visible,
            },
        }
    }

    fn apply_to_figure(self, figure: &mut Figure) -> Result<(), String> {
        match self {
            ScenePlot::Line {
                x,
                y,
                color_rgba,
                line_width,
                line_style,
                axes_index,
                label,
                visible,
            } => {
                let mut line = LinePlot::new(x, y)?;
                line.set_color(rgba_to_vec4(color_rgba));
                line.set_line_width(line_width);
                line.set_line_style(parse_line_style(&line_style));
                line.label = label;
                line.set_visible(visible);
                figure.add_line_plot_on_axes(line, axes_index as usize);
            }
            ScenePlot::Scatter {
                x,
                y,
                color_rgba,
                marker_size,
                marker_style,
                axes_index,
                label,
                visible,
            } => {
                let mut scatter = ScatterPlot::new(x, y)?;
                scatter.set_color(rgba_to_vec4(color_rgba));
                scatter.set_marker_size(marker_size);
                scatter.set_marker_style(parse_marker_style(&marker_style));
                scatter.label = label;
                scatter.set_visible(visible);
                figure.add_scatter_plot_on_axes(scatter, axes_index as usize);
            }
            ScenePlot::Bar {
                labels,
                values,
                histogram_bin_edges,
                color_rgba,
                outline_color_rgba,
                bar_width,
                outline_width,
                orientation,
                group_index,
                group_count,
                stack_offsets,
                axes_index,
                label,
                visible,
            } => {
                let mut bar = BarChart::new(labels, values)?
                    .with_style(rgba_to_vec4(color_rgba), bar_width)
                    .with_orientation(parse_bar_orientation(&orientation))
                    .with_group(group_index as usize, group_count as usize);
                if let Some(edges) = histogram_bin_edges {
                    bar.set_histogram_bin_edges(edges);
                }
                if let Some(offsets) = stack_offsets {
                    bar = bar.with_stack_offsets(offsets);
                }
                if let Some(outline) = outline_color_rgba {
                    bar = bar.with_outline(rgba_to_vec4(outline), outline_width);
                }
                bar.label = label;
                bar.set_visible(visible);
                figure.add_bar_chart_on_axes(bar, axes_index as usize);
            }
            ScenePlot::ErrorBar {
                x,
                y,
                err_low,
                err_high,
                x_err_low,
                x_err_high,
                orientation,
                color_rgba,
                line_width,
                line_style,
                cap_width,
                marker_style,
                marker_size,
                marker_face_color,
                marker_edge_color,
                marker_filled,
                axes_index,
                label,
                visible,
            } => {
                let mut error = if orientation.eq_ignore_ascii_case("Both") {
                    ErrorBar::new_both(x, y, x_err_low, x_err_high, err_low, err_high)?
                } else {
                    ErrorBar::new_vertical(x, y, err_low, err_high)?
                }
                .with_style(
                    rgba_to_vec4(color_rgba),
                    line_width,
                    parse_line_style_name(&line_style),
                    cap_width,
                );
                if let Some(size) = marker_size {
                    error.set_marker(Some(crate::plots::line::LineMarkerAppearance {
                        kind: parse_marker_style(marker_style.as_deref().unwrap_or("Circle")),
                        size,
                        edge_color: marker_edge_color
                            .map(rgba_to_vec4)
                            .unwrap_or(rgba_to_vec4(color_rgba)),
                        face_color: marker_face_color
                            .map(rgba_to_vec4)
                            .unwrap_or(rgba_to_vec4(color_rgba)),
                        filled: marker_filled.unwrap_or(false),
                    }));
                }
                error.label = label;
                error.set_visible(visible);
                figure.add_errorbar_on_axes(error, axes_index as usize);
            }
            ScenePlot::Stairs {
                x,
                y,
                color_rgba,
                line_width,
                axes_index,
                label,
                visible,
            } => {
                let mut stairs = StairsPlot::new(x, y)?;
                stairs.color = rgba_to_vec4(color_rgba);
                stairs.line_width = line_width;
                stairs.label = label;
                stairs.set_visible(visible);
                figure.add_stairs_plot_on_axes(stairs, axes_index as usize);
            }
            ScenePlot::Stem {
                x,
                y,
                baseline,
                color_rgba,
                line_width,
                line_style,
                baseline_color_rgba,
                baseline_visible,
                marker_color_rgba,
                marker_size,
                marker_filled,
                axes_index,
                label,
                visible,
            } => {
                let mut stem = StemPlot::new(x, y)?;
                stem = stem
                    .with_style(
                        rgba_to_vec4(color_rgba),
                        line_width,
                        parse_line_style_name(&line_style),
                        baseline,
                    )
                    .with_baseline_style(rgba_to_vec4(baseline_color_rgba), baseline_visible);
                if marker_size > 0.0 {
                    stem.set_marker(Some(crate::plots::line::LineMarkerAppearance {
                        kind: crate::plots::scatter::MarkerStyle::Circle,
                        size: marker_size,
                        edge_color: rgba_to_vec4(marker_color_rgba),
                        face_color: rgba_to_vec4(marker_color_rgba),
                        filled: marker_filled,
                    }));
                }
                stem.label = label;
                stem.set_visible(visible);
                figure.add_stem_plot_on_axes(stem, axes_index as usize);
            }
            ScenePlot::Area {
                x,
                y,
                lower_y,
                baseline,
                color_rgba,
                axes_index,
                label,
                visible,
            } => {
                let mut area = AreaPlot::new(x, y)?;
                if let Some(lower_y) = lower_y {
                    area = area.with_lower_curve(lower_y);
                }
                area.baseline = baseline;
                area.color = rgba_to_vec4(color_rgba);
                area.label = label;
                area.set_visible(visible);
                figure.add_area_plot_on_axes(area, axes_index as usize);
            }
            ScenePlot::Quiver {
                x,
                y,
                u,
                v,
                color_rgba,
                line_width,
                scale,
                head_size,
                axes_index,
                label,
                visible,
            } => {
                let mut quiver = QuiverPlot::new(x, y, u, v)?
                    .with_style(rgba_to_vec4(color_rgba), line_width, scale, head_size)
                    .with_label(label.unwrap_or_else(|| "Data".to_string()));
                quiver.set_visible(visible);
                figure.add_quiver_plot_on_axes(quiver, axes_index as usize);
            }
            ScenePlot::Surface {
                x,
                y,
                z,
                colormap,
                shading_mode,
                wireframe,
                alpha,
                flatten_z,
                image_mode,
                color_grid_rgba,
                color_limits,
                axes_index,
                label,
                visible,
            } => {
                let mut surface = SurfacePlot::new(x, y, z)?;
                surface.colormap = parse_colormap(&colormap);
                surface.shading_mode = parse_shading_mode(&shading_mode);
                surface.wireframe = wireframe;
                surface.alpha = alpha.clamp(0.0, 1.0);
                surface.flatten_z = flatten_z;
                surface.image_mode = image_mode;
                surface.color_grid = color_grid_rgba.map(|grid| {
                    grid.into_iter()
                        .map(|row| row.into_iter().map(rgba_to_vec4).collect())
                        .collect()
                });
                surface.color_limits = color_limits.map(|[lo, hi]| (lo, hi));
                surface.label = label;
                surface.visible = visible;
                figure.add_surface_plot_on_axes(surface, axes_index as usize);
            }
            ScenePlot::Line3 {
                x,
                y,
                z,
                color_rgba,
                line_width,
                line_style,
                axes_index,
                label,
                visible,
            } => {
                let mut plot = Line3Plot::new(x, y, z)?
                    .with_style(
                        rgba_to_vec4(color_rgba),
                        line_width,
                        parse_line_style_name(&line_style),
                    )
                    .with_label(label.unwrap_or_else(|| "Data".to_string()));
                plot.set_visible(visible);
                figure.add_line3_plot_on_axes(plot, axes_index as usize);
            }
            ScenePlot::Scatter3 {
                points,
                colors_rgba,
                point_size,
                point_sizes,
                axes_index,
                label,
                visible,
            } => {
                let points: Vec<Vec3> = points.into_iter().map(xyz_to_vec3).collect();
                let colors: Vec<Vec4> = colors_rgba.into_iter().map(rgba_to_vec4).collect();
                let mut scatter3 = Scatter3Plot::new(points)?;
                if !colors.is_empty() {
                    scatter3 = scatter3.with_colors(colors)?;
                }
                scatter3.point_size = point_size.max(1.0);
                scatter3.point_sizes = point_sizes;
                scatter3.label = label;
                scatter3.visible = visible;
                figure.add_scatter3_plot_on_axes(scatter3, axes_index as usize);
            }
            ScenePlot::Contour {
                vertices,
                bounds_min,
                bounds_max,
                base_z,
                line_width,
                axes_index,
                label,
                visible,
            } => {
                let mut contour = ContourPlot::from_vertices(
                    vertices.into_iter().map(Into::into).collect(),
                    base_z,
                    serialized_bounds(bounds_min, bounds_max),
                )
                .with_line_width(line_width);
                contour.label = label;
                contour.set_visible(visible);
                figure.add_contour_plot_on_axes(contour, axes_index as usize);
            }
            ScenePlot::ContourFill {
                vertices,
                bounds_min,
                bounds_max,
                axes_index,
                label,
                visible,
            } => {
                let mut fill = ContourFillPlot::from_vertices(
                    vertices.into_iter().map(Into::into).collect(),
                    serialized_bounds(bounds_min, bounds_max),
                );
                fill.label = label;
                fill.set_visible(visible);
                figure.add_contour_fill_plot_on_axes(fill, axes_index as usize);
            }
            ScenePlot::Pie {
                values,
                colors_rgba,
                slice_labels,
                label_format,
                explode,
                axes_index,
                label,
                visible,
            } => {
                let mut pie = crate::plots::PieChart::new(
                    values,
                    Some(colors_rgba.into_iter().map(rgba_to_vec4).collect()),
                )?
                .with_slice_labels(slice_labels)
                .with_explode(explode);
                if let Some(fmt) = label_format {
                    pie = pie.with_label_format(fmt);
                }
                pie.label = label;
                pie.set_visible(visible);
                figure.add_pie_chart_on_axes(pie, axes_index as usize);
            }
            ScenePlot::Unsupported { .. } => {}
        }
        Ok(())
    }
}

fn parse_line_style(value: &str) -> crate::plots::LineStyle {
    match value {
        "Dashed" => crate::plots::LineStyle::Dashed,
        "Dotted" => crate::plots::LineStyle::Dotted,
        "DashDot" => crate::plots::LineStyle::DashDot,
        _ => crate::plots::LineStyle::Solid,
    }
}

fn parse_bar_orientation(value: &str) -> crate::plots::bar::Orientation {
    match value {
        "Horizontal" => crate::plots::bar::Orientation::Horizontal,
        _ => crate::plots::bar::Orientation::Vertical,
    }
}

fn parse_marker_style(value: &str) -> MarkerStyle {
    match value {
        "Square" => MarkerStyle::Square,
        "Triangle" => MarkerStyle::Triangle,
        "Diamond" => MarkerStyle::Diamond,
        "Plus" => MarkerStyle::Plus,
        "Cross" => MarkerStyle::Cross,
        "Star" => MarkerStyle::Star,
        "Hexagon" => MarkerStyle::Hexagon,
        _ => MarkerStyle::Circle,
    }
}

fn parse_colormap(value: &str) -> ColorMap {
    match value {
        "Jet" => ColorMap::Jet,
        "Hot" => ColorMap::Hot,
        "Cool" => ColorMap::Cool,
        "Spring" => ColorMap::Spring,
        "Summer" => ColorMap::Summer,
        "Autumn" => ColorMap::Autumn,
        "Winter" => ColorMap::Winter,
        "Gray" => ColorMap::Gray,
        "Bone" => ColorMap::Bone,
        "Copper" => ColorMap::Copper,
        "Pink" => ColorMap::Pink,
        "Lines" => ColorMap::Lines,
        "Viridis" => ColorMap::Viridis,
        "Plasma" => ColorMap::Plasma,
        "Inferno" => ColorMap::Inferno,
        "Magma" => ColorMap::Magma,
        "Turbo" => ColorMap::Turbo,
        "Parula" => ColorMap::Parula,
        _ => ColorMap::Parula,
    }
}

fn parse_shading_mode(value: &str) -> ShadingMode {
    match value {
        "Flat" => ShadingMode::Flat,
        "Smooth" => ShadingMode::Smooth,
        "Faceted" => ShadingMode::Faceted,
        "None" => ShadingMode::None,
        _ => ShadingMode::Smooth,
    }
}

fn xyz_to_vec3(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

fn serialized_bounds(min: [f32; 3], max: [f32; 3]) -> BoundingBox {
    BoundingBox::new(xyz_to_vec3(min), xyz_to_vec3(max))
}

fn vec3_to_xyz(value: Vec3) -> [f32; 3] {
    [value.x, value.y, value.z]
}

fn rgba_to_vec4(value: [f32; 4]) -> Vec4 {
    Vec4::new(value[0], value[1], value[2], value[3])
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedVertex {
    position: [f32; 3],
    color_rgba: [f32; 4],
    normal: [f32; 3],
    tex_coords: [f32; 2],
}

impl From<Vertex> for SerializedVertex {
    fn from(value: Vertex) -> Self {
        Self {
            position: value.position,
            color_rgba: value.color,
            normal: value.normal,
            tex_coords: value.tex_coords,
        }
    }
}

impl From<SerializedVertex> for Vertex {
    fn from(value: SerializedVertex) -> Self {
        Self {
            position: value.position,
            color: value.color_rgba,
            normal: value.normal,
            tex_coords: value.tex_coords,
        }
    }
}

/// Serialized legend entry for frontend rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureLegendEntry {
    pub label: String,
    pub plot_type: PlotKind,
    pub color_rgba: [f32; 4],
}

impl From<LegendEntry> for FigureLegendEntry {
    fn from(entry: LegendEntry) -> Self {
        Self {
            label: entry.label,
            plot_type: PlotKind::from(entry.plot_type),
            color_rgba: vec4_to_rgba(entry.color),
        }
    }
}

/// Serializable plot kind values consumed by UI + transports.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlotKind {
    Line,
    Line3,
    Scatter,
    Bar,
    ErrorBar,
    Stairs,
    Stem,
    Area,
    Quiver,
    Pie,
    Image,
    Surface,
    Scatter3,
    Contour,
    ContourFill,
}

impl From<PlotType> for PlotKind {
    fn from(value: PlotType) -> Self {
        match value {
            PlotType::Line => Self::Line,
            PlotType::Line3 => Self::Line3,
            PlotType::Scatter => Self::Scatter,
            PlotType::Bar => Self::Bar,
            PlotType::ErrorBar => Self::ErrorBar,
            PlotType::Stairs => Self::Stairs,
            PlotType::Stem => Self::Stem,
            PlotType::Area => Self::Area,
            PlotType::Quiver => Self::Quiver,
            PlotType::Pie => Self::Pie,
            PlotType::Surface => Self::Surface,
            PlotType::Scatter3 => Self::Scatter3,
            PlotType::Contour => Self::Contour,
            PlotType::ContourFill => Self::ContourFill,
        }
    }
}

fn parse_line_style_name(name: &str) -> crate::plots::line::LineStyle {
    match name.to_ascii_lowercase().as_str() {
        "dashed" => crate::plots::line::LineStyle::Dashed,
        "dotted" => crate::plots::line::LineStyle::Dotted,
        "dashdot" => crate::plots::line::LineStyle::DashDot,
        _ => crate::plots::line::LineStyle::Solid,
    }
}

fn parse_colormap_name(name: &str) -> crate::plots::surface::ColorMap {
    match name.trim().to_ascii_lowercase().as_str() {
        "viridis" => crate::plots::surface::ColorMap::Viridis,
        "plasma" => crate::plots::surface::ColorMap::Plasma,
        "inferno" => crate::plots::surface::ColorMap::Inferno,
        "magma" => crate::plots::surface::ColorMap::Magma,
        "turbo" => crate::plots::surface::ColorMap::Turbo,
        "jet" => crate::plots::surface::ColorMap::Jet,
        "hot" => crate::plots::surface::ColorMap::Hot,
        "cool" => crate::plots::surface::ColorMap::Cool,
        "spring" => crate::plots::surface::ColorMap::Spring,
        "summer" => crate::plots::surface::ColorMap::Summer,
        "autumn" => crate::plots::surface::ColorMap::Autumn,
        "winter" => crate::plots::surface::ColorMap::Winter,
        "gray" | "grey" => crate::plots::surface::ColorMap::Gray,
        "bone" => crate::plots::surface::ColorMap::Bone,
        "copper" => crate::plots::surface::ColorMap::Copper,
        "pink" => crate::plots::surface::ColorMap::Pink,
        "lines" => crate::plots::surface::ColorMap::Lines,
        _ => crate::plots::surface::ColorMap::Parula,
    }
}

fn vec4_to_rgba(value: Vec4) -> [f32; 4] {
    [value.x, value.y, value.z, value.w]
}

fn deserialize_f64_lossy<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<f64>::deserialize(deserializer)?;
    Ok(value.unwrap_or(f64::NAN))
}

fn deserialize_vec_f64_lossy<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Vec::<Option<f64>>::deserialize(deserializer)?;
    Ok(values
        .into_iter()
        .map(|value| value.unwrap_or(f64::NAN))
        .collect())
}

fn deserialize_option_vec_f64_lossy<'de, D>(deserializer: D) -> Result<Option<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Option::<Vec<Option<f64>>>::deserialize(deserializer)?;
    Ok(values.map(|items| {
        items
            .into_iter()
            .map(|value| value.unwrap_or(f64::NAN))
            .collect()
    }))
}

fn deserialize_matrix_f64_lossy<'de, D>(deserializer: D) -> Result<Vec<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let rows = Vec::<Vec<Option<f64>>>::deserialize(deserializer)?;
    Ok(rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect()
        })
        .collect())
}

fn deserialize_option_pair_f64_lossy<'de, D>(deserializer: D) -> Result<Option<[f64; 2]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<[Option<f64>; 2]>::deserialize(deserializer)?;
    Ok(value.map(|pair| [pair[0].unwrap_or(f64::NAN), pair[1].unwrap_or(f64::NAN)]))
}

fn deserialize_option_vec_f32_lossy<'de, D>(deserializer: D) -> Result<Option<Vec<f32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Option::<Vec<Option<f32>>>::deserialize(deserializer)?;
    Ok(values.map(|items| {
        items
            .into_iter()
            .map(|value| value.unwrap_or(f32::NAN))
            .collect()
    }))
}

fn deserialize_vec_xyz_f32_lossy<'de, D>(deserializer: D) -> Result<Vec<[f32; 3]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Vec::<[Option<f32>; 3]>::deserialize(deserializer)?;
    Ok(values
        .into_iter()
        .map(|xyz| {
            [
                xyz[0].unwrap_or(f32::NAN),
                xyz[1].unwrap_or(f32::NAN),
                xyz[2].unwrap_or(f32::NAN),
            ]
        })
        .collect())
}

fn deserialize_vec_rgba_f32_lossy<'de, D>(deserializer: D) -> Result<Vec<[f32; 4]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Vec::<[Option<f32>; 4]>::deserialize(deserializer)?;
    Ok(values
        .into_iter()
        .map(|rgba| {
            [
                rgba[0].unwrap_or(f32::NAN),
                rgba[1].unwrap_or(f32::NAN),
                rgba[2].unwrap_or(f32::NAN),
                rgba[3].unwrap_or(f32::NAN),
            ]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plots::{Figure, Line3Plot, LinePlot, Scatter3Plot, ScatterPlot, SurfacePlot};
    use glam::Vec3;

    #[test]
    fn capture_snapshot_reflects_layout_and_metadata() {
        let mut figure = Figure::new()
            .with_title("Demo")
            .with_labels("X", "Y")
            .with_grid(false)
            .with_subplot_grid(1, 2);
        let line = LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        figure.add_line_plot_on_axes(line, 1);

        let snapshot = FigureSnapshot::capture(&figure);
        assert_eq!(snapshot.layout.axes_rows, 1);
        assert_eq!(snapshot.layout.axes_cols, 2);
        assert_eq!(snapshot.metadata.title.as_deref(), Some("Demo"));
        assert_eq!(snapshot.metadata.legend_entries.len(), 0);
        assert_eq!(snapshot.plots.len(), 1);
        assert_eq!(snapshot.plots[0].axes_index, 1);
        assert!(!snapshot.metadata.grid_enabled);
    }

    #[test]
    fn figure_scene_roundtrip_reconstructs_supported_plots() {
        let mut figure = Figure::new().with_title("Replay").with_subplot_grid(1, 2);
        let mut line = LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
        line.label = Some("line".to_string());
        figure.add_line_plot_on_axes(line, 0);
        let mut scatter = ScatterPlot::new(vec![0.0, 1.0, 2.0], vec![2.0, 3.0, 4.0]).unwrap();
        scatter.label = Some("scatter".to_string());
        figure.add_scatter_plot_on_axes(scatter, 1);

        let scene = FigureScene::capture(&figure);
        let rebuilt = scene.into_figure().expect("scene restore should succeed");
        assert_eq!(rebuilt.axes_grid(), (1, 2));
        assert_eq!(rebuilt.plots().count(), 2);
        assert_eq!(rebuilt.title.as_deref(), Some("Replay"));
    }

    #[test]
    fn figure_scene_roundtrip_reconstructs_surface_and_scatter3() {
        let mut figure = Figure::new().with_title("Replay3D").with_subplot_grid(1, 2);
        let mut surface = SurfacePlot::new(
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![vec![0.0, 1.0], vec![1.0, 2.0]],
        )
        .expect("surface data should be valid");
        surface.label = Some("surface".to_string());
        figure.add_surface_plot_on_axes(surface, 0);

        let mut scatter3 = Scatter3Plot::new(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(2.0, 3.0, 4.0),
        ])
        .expect("scatter3 data should be valid");
        scatter3.label = Some("scatter3".to_string());
        figure.add_scatter3_plot_on_axes(scatter3, 1);

        let scene = FigureScene::capture(&figure);
        let rebuilt = scene.into_figure().expect("scene restore should succeed");
        assert_eq!(rebuilt.axes_grid(), (1, 2));
        assert_eq!(rebuilt.plots().count(), 2);
        assert_eq!(rebuilt.title.as_deref(), Some("Replay3D"));
        assert!(matches!(
            rebuilt.plots().next(),
            Some(PlotElement::Surface(_))
        ));
        assert!(matches!(
            rebuilt.plots().nth(1),
            Some(PlotElement::Scatter3(_))
        ));
    }

    #[test]
    fn figure_scene_roundtrip_preserves_line3_plot() {
        let mut figure = Figure::new();
        let line3 = Line3Plot::new(vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0])
            .unwrap()
            .with_label("Trajectory");
        figure.add_line3_plot(line3);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");

        let PlotElement::Line3(line3) = rebuilt.plots().next().unwrap() else {
            panic!("expected line3")
        };
        assert_eq!(line3.x_data, vec![0.0, 1.0]);
        assert_eq!(line3.z_data, vec![2.0, 3.0]);
        assert_eq!(line3.label.as_deref(), Some("Trajectory"));
    }

    #[test]
    fn figure_scene_roundtrip_preserves_contour_and_fill_plots() {
        let mut figure = Figure::new();
        let bounds = BoundingBox::new(Vec3::new(-1.0, -2.0, 0.0), Vec3::new(3.0, 4.0, 0.0));
        let vertices = vec![Vertex {
            position: [0.0, 0.0, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            tex_coords: [0.0, 0.0],
        }];
        let fill = ContourFillPlot::from_vertices(vertices.clone(), bounds).with_label("fill");
        let contour = ContourPlot::from_vertices(vertices, 0.0, bounds)
            .with_label("lines")
            .with_line_width(2.0);
        figure.add_contour_fill_plot(fill);
        figure.add_contour_plot(contour);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        assert!(matches!(
            rebuilt.plots().next(),
            Some(PlotElement::ContourFill(_))
        ));
        let Some(PlotElement::Contour(contour)) = rebuilt.plots().nth(1) else {
            panic!("expected contour")
        };
        assert_eq!(contour.line_width, 2.0);
    }

    #[test]
    fn figure_scene_roundtrip_preserves_stem_style_surface() {
        let mut figure = Figure::new();
        let mut stem = StemPlot::new(vec![0.0, 1.0], vec![1.0, 2.0])
            .unwrap()
            .with_style(
                Vec4::new(1.0, 0.0, 0.0, 1.0),
                2.0,
                crate::plots::line::LineStyle::Dashed,
                -1.0,
            )
            .with_baseline_style(Vec4::new(0.0, 0.0, 0.0, 1.0), false)
            .with_label("Impulse");
        stem.set_marker(Some(crate::plots::line::LineMarkerAppearance {
            kind: crate::plots::scatter::MarkerStyle::Square,
            size: 8.0,
            edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            face_color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            filled: true,
        }));
        figure.add_stem_plot(stem);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let PlotElement::Stem(stem) = rebuilt.plots().next().unwrap() else {
            panic!("expected stem")
        };
        assert_eq!(stem.baseline, -1.0);
        assert_eq!(stem.line_width, 2.0);
        assert_eq!(stem.label.as_deref(), Some("Impulse"));
        assert!(!stem.baseline_visible);
        assert!(stem.marker.as_ref().map(|m| m.filled).unwrap_or(false));
        assert_eq!(stem.marker.as_ref().map(|m| m.size), Some(8.0));
    }

    #[test]
    fn figure_scene_roundtrip_preserves_bar_plot() {
        let mut figure = Figure::new();
        let bar = BarChart::new(vec!["A".into(), "B".into()], vec![2.0, 3.5])
            .unwrap()
            .with_style(Vec4::new(0.2, 0.4, 0.8, 1.0), 0.95)
            .with_outline(Vec4::new(0.1, 0.1, 0.1, 1.0), 1.5)
            .with_label("Histogram")
            .with_stack_offsets(vec![1.0, 0.5]);
        figure.add_bar_chart(bar);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let PlotElement::Bar(bar) = rebuilt.plots().next().unwrap() else {
            panic!("expected bar")
        };
        assert_eq!(bar.labels, vec!["A", "B"]);
        assert_eq!(bar.values().unwrap_or(&[]), &[2.0, 3.5]);
        assert_eq!(bar.bar_width, 0.95);
        assert_eq!(bar.outline_width, 1.5);
        assert_eq!(bar.label.as_deref(), Some("Histogram"));
        assert_eq!(bar.stack_offsets().unwrap_or(&[]), &[1.0, 0.5]);
        assert!(bar.histogram_bin_edges().is_none());
    }

    #[test]
    fn figure_scene_roundtrip_preserves_histogram_bin_edges() {
        let mut figure = Figure::new();
        let mut bar = BarChart::new(vec!["bin1".into(), "bin2".into()], vec![4.0, 5.0]).unwrap();
        bar.set_histogram_bin_edges(vec![0.0, 0.5, 1.0]);
        figure.add_bar_chart(bar);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let PlotElement::Bar(bar) = rebuilt.plots().next().unwrap() else {
            panic!("expected bar")
        };
        assert_eq!(bar.histogram_bin_edges().unwrap_or(&[]), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn figure_scene_roundtrip_preserves_errorbar_style_surface() {
        let mut figure = Figure::new();
        let mut error = ErrorBar::new_vertical(
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![0.1, 0.2],
            vec![0.2, 0.3],
        )
        .unwrap()
        .with_style(
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            2.0,
            crate::plots::line::LineStyle::Dashed,
            10.0,
        )
        .with_label("Err");
        error.set_marker(Some(crate::plots::line::LineMarkerAppearance {
            kind: crate::plots::scatter::MarkerStyle::Triangle,
            size: 8.0,
            edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            face_color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            filled: true,
        }));
        figure.add_errorbar(error);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let PlotElement::ErrorBar(error) = rebuilt.plots().next().unwrap() else {
            panic!("expected errorbar")
        };
        assert_eq!(error.line_width, 2.0);
        assert_eq!(error.cap_size, 10.0);
        assert_eq!(error.label.as_deref(), Some("Err"));
        assert_eq!(error.line_style, crate::plots::line::LineStyle::Dashed);
        assert!(error.marker.as_ref().map(|m| m.filled).unwrap_or(false));
    }

    #[test]
    fn figure_scene_roundtrip_preserves_errorbar_both_direction() {
        let mut figure = Figure::new();
        let error = ErrorBar::new_both(
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![0.1, 0.2],
            vec![0.2, 0.3],
            vec![0.3, 0.4],
            vec![0.4, 0.5],
        )
        .unwrap();
        figure.add_errorbar(error);
        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let PlotElement::ErrorBar(error) = rebuilt.plots().next().unwrap() else {
            panic!("expected errorbar")
        };
        assert_eq!(
            error.orientation,
            crate::plots::errorbar::ErrorBarOrientation::Both
        );
        assert_eq!(error.x_neg, vec![0.1, 0.2]);
        assert_eq!(error.x_pos, vec![0.2, 0.3]);
    }

    #[test]
    fn figure_scene_roundtrip_preserves_quiver_plot() {
        let mut figure = Figure::new();
        let quiver = QuiverPlot::new(
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![0.5, -0.5],
            vec![1.0, 0.25],
        )
        .unwrap()
        .with_style(Vec4::new(0.2, 0.3, 0.4, 1.0), 2.0, 1.5, 0.2)
        .with_label("Field");
        figure.add_quiver_plot(quiver);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let PlotElement::Quiver(quiver) = rebuilt.plots().next().unwrap() else {
            panic!("expected quiver")
        };
        assert_eq!(quiver.u, vec![0.5, -0.5]);
        assert_eq!(quiver.v, vec![1.0, 0.25]);
        assert_eq!(quiver.line_width, 2.0);
        assert_eq!(quiver.scale, 1.5);
        assert_eq!(quiver.head_size, 0.2);
        assert_eq!(quiver.label.as_deref(), Some("Field"));
    }

    #[test]
    fn figure_scene_roundtrip_preserves_image_surface_mode_and_color_grid() {
        let mut figure = Figure::new();
        let surface = SurfacePlot::new(
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        )
        .unwrap()
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_color_grid(vec![
            vec![Vec4::new(1.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0)],
            vec![Vec4::new(0.0, 0.0, 1.0, 1.0), Vec4::new(1.0, 1.0, 1.0, 1.0)],
        ]);
        figure.add_surface_plot(surface);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let PlotElement::Surface(surface) = rebuilt.plots().next().unwrap() else {
            panic!("expected surface")
        };
        assert!(surface.flatten_z);
        assert!(surface.image_mode);
        assert!(surface.color_grid.is_some());
        assert_eq!(
            surface.color_grid.as_ref().unwrap()[0][0],
            Vec4::new(1.0, 0.0, 0.0, 1.0)
        );
    }

    #[test]
    fn figure_scene_roundtrip_preserves_area_lower_curve() {
        let mut figure = Figure::new();
        let area = AreaPlot::new(vec![1.0, 2.0], vec![2.0, 3.0])
            .unwrap()
            .with_lower_curve(vec![0.5, 1.0])
            .with_label("Stacked");
        figure.add_area_plot(area);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let PlotElement::Area(area) = rebuilt.plots().next().unwrap() else {
            panic!("expected area")
        };
        assert_eq!(area.lower_y, Some(vec![0.5, 1.0]));
        assert_eq!(area.label.as_deref(), Some("Stacked"));
    }

    #[test]
    fn figure_scene_roundtrip_preserves_axes_local_limits_and_colormap_state() {
        let mut figure = Figure::new().with_subplot_grid(1, 2);
        figure.set_axes_limits(1, Some((1.0, 2.0)), Some((3.0, 4.0)));
        figure.set_axes_z_limits(1, Some((5.0, 6.0)));
        figure.set_axes_grid_enabled(1, false);
        figure.set_axes_box_enabled(1, false);
        figure.set_axes_axis_equal(1, true);
        figure.set_axes_colorbar_enabled(1, true);
        figure.set_axes_colormap(1, ColorMap::Hot);
        figure.set_axes_color_limits(1, Some((0.0, 10.0)));
        figure.set_active_axes_index(1);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let meta = rebuilt.axes_metadata(1).unwrap();
        assert_eq!(meta.x_limits, Some((1.0, 2.0)));
        assert_eq!(meta.y_limits, Some((3.0, 4.0)));
        assert_eq!(meta.z_limits, Some((5.0, 6.0)));
        assert!(!meta.grid_enabled);
        assert!(!meta.box_enabled);
        assert!(meta.axis_equal);
        assert!(meta.colorbar_enabled);
        assert_eq!(format!("{:?}", meta.colormap), "Hot");
        assert_eq!(meta.color_limits, Some((0.0, 10.0)));
    }

    #[test]
    fn figure_scene_roundtrip_preserves_axes_local_annotation_metadata() {
        let mut figure = Figure::new().with_subplot_grid(1, 2);
        figure.set_active_axes_index(0);
        figure.set_axes_title(0, "Left");
        figure.set_axes_xlabel(0, "LX");
        figure.set_axes_ylabel(0, "LY");
        figure.set_axes_legend_enabled(0, false);
        figure.set_axes_title(1, "Right");
        figure.set_axes_xlabel(1, "RX");
        figure.set_axes_ylabel(1, "RY");
        figure.set_axes_legend_enabled(1, true);
        figure.set_axes_legend_style(
            1,
            LegendStyle {
                location: Some("northeast".into()),
                font_weight: Some("bold".into()),
                orientation: Some("horizontal".into()),
                ..Default::default()
            },
        );
        if let Some(meta) = figure.axes_metadata.get_mut(0) {
            meta.title_style.font_weight = Some("bold".into());
            meta.title_style.font_angle = Some("italic".into());
        }
        figure.set_active_axes_index(1);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");

        assert_eq!(rebuilt.active_axes_index, 1);
        assert_eq!(
            rebuilt.axes_metadata(0).and_then(|m| m.title.as_deref()),
            Some("Left")
        );
        assert_eq!(
            rebuilt.axes_metadata(0).and_then(|m| m.x_label.as_deref()),
            Some("LX")
        );
        assert_eq!(
            rebuilt.axes_metadata(0).and_then(|m| m.y_label.as_deref()),
            Some("LY")
        );
        assert!(!rebuilt.axes_metadata(0).unwrap().legend_enabled);
        assert_eq!(
            rebuilt
                .axes_metadata(0)
                .unwrap()
                .title_style
                .font_weight
                .as_deref(),
            Some("bold")
        );
        assert_eq!(
            rebuilt
                .axes_metadata(0)
                .unwrap()
                .title_style
                .font_angle
                .as_deref(),
            Some("italic")
        );
        assert_eq!(
            rebuilt.axes_metadata(1).and_then(|m| m.title.as_deref()),
            Some("Right")
        );
        assert_eq!(
            rebuilt.axes_metadata(1).and_then(|m| m.x_label.as_deref()),
            Some("RX")
        );
        assert_eq!(
            rebuilt.axes_metadata(1).and_then(|m| m.y_label.as_deref()),
            Some("RY")
        );
        assert_eq!(
            rebuilt
                .axes_metadata(1)
                .unwrap()
                .legend_style
                .location
                .as_deref(),
            Some("northeast")
        );
        assert_eq!(
            rebuilt
                .axes_metadata(1)
                .unwrap()
                .legend_style
                .font_weight
                .as_deref(),
            Some("bold")
        );
        assert_eq!(
            rebuilt
                .axes_metadata(1)
                .unwrap()
                .legend_style
                .orientation
                .as_deref(),
            Some("horizontal")
        );
    }

    #[test]
    fn figure_scene_roundtrip_preserves_axes_local_log_modes() {
        let mut figure = Figure::new().with_subplot_grid(1, 2);
        figure.set_axes_log_modes(0, true, false);
        figure.set_axes_log_modes(1, false, true);
        figure.set_active_axes_index(1);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");

        assert!(rebuilt.axes_metadata(0).unwrap().x_log);
        assert!(!rebuilt.axes_metadata(0).unwrap().y_log);
        assert!(!rebuilt.axes_metadata(1).unwrap().x_log);
        assert!(rebuilt.axes_metadata(1).unwrap().y_log);
        assert!(!rebuilt.x_log);
        assert!(rebuilt.y_log);
    }

    #[test]
    fn figure_scene_roundtrip_preserves_zlabel_and_view_state() {
        let mut figure = Figure::new().with_subplot_grid(1, 2);
        figure.set_axes_zlabel(1, "Height");
        figure.set_axes_view(1, 45.0, 20.0);
        figure.set_active_axes_index(1);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");

        assert_eq!(
            rebuilt.axes_metadata(1).unwrap().z_label.as_deref(),
            Some("Height")
        );
        assert_eq!(
            rebuilt.axes_metadata(1).unwrap().view_azimuth_deg,
            Some(45.0)
        );
        assert_eq!(
            rebuilt.axes_metadata(1).unwrap().view_elevation_deg,
            Some(20.0)
        );
        assert_eq!(rebuilt.z_label.as_deref(), Some("Height"));
    }

    #[test]
    fn figure_scene_roundtrip_preserves_pie_metadata() {
        let mut figure = Figure::new();
        let pie = crate::plots::PieChart::new(vec![1.0, 2.0], None)
            .unwrap()
            .with_slice_labels(vec!["A".into(), "B".into()])
            .with_explode(vec![false, true]);
        figure.add_pie_chart(pie);

        let rebuilt = FigureScene::capture(&figure)
            .into_figure()
            .expect("scene restore should succeed");
        let crate::plots::figure::PlotElement::Pie(pie) = rebuilt.plots().next().unwrap() else {
            panic!("expected pie")
        };
        assert_eq!(pie.slice_labels, vec!["A", "B"]);
        assert_eq!(pie.explode, vec![false, true]);
    }

    #[test]
    fn scene_plot_deserialize_maps_null_numeric_values_to_nan() {
        let json = r#"{
          "schemaVersion": 1,
          "layout": { "axesRows": 1, "axesCols": 1, "axesIndices": [0] },
          "metadata": {
            "gridEnabled": true,
            "legendEnabled": false,
            "colorbarEnabled": false,
            "axisEqual": false,
            "backgroundRgba": [1,1,1,1],
            "legendEntries": []
          },
          "plots": [
            {
              "kind": "surface",
              "x": [0.0, null],
              "y": [0.0, 1.0],
              "z": [[0.0, null], [1.0, 2.0]],
              "colormap": "Parula",
              "shading_mode": "Smooth",
              "wireframe": false,
              "alpha": 1.0,
              "flatten_z": false,
              "color_limits": null,
              "axes_index": 0,
              "label": null,
              "visible": true
            }
          ]
        }"#;
        let scene: FigureScene = serde_json::from_str(json).expect("scene should deserialize");
        let ScenePlot::Surface { x, z, .. } = &scene.plots[0] else {
            panic!("expected surface plot");
        };
        assert!(x[1].is_nan());
        assert!(z[0][1].is_nan());
    }

    #[test]
    fn scene_plot_deserialize_maps_null_scatter3_components_to_nan() {
        let json = r#"{
          "schemaVersion": 1,
          "layout": { "axesRows": 1, "axesCols": 1, "axesIndices": [0] },
          "metadata": {
            "gridEnabled": true,
            "legendEnabled": false,
            "colorbarEnabled": false,
            "axisEqual": false,
            "backgroundRgba": [1,1,1,1],
            "legendEntries": []
          },
          "plots": [
            {
              "kind": "scatter3",
              "points": [[0.0, 1.0, null], [1.0, null, 2.0]],
              "colors_rgba": [[0.2, 0.4, 0.6, 1.0], [0.1, 0.2, 0.3, 1.0]],
              "point_size": 6.0,
              "point_sizes": [3.0, null],
              "axes_index": 0,
              "label": null,
              "visible": true
            }
          ]
        }"#;
        let scene: FigureScene = serde_json::from_str(json).expect("scene should deserialize");
        let ScenePlot::Scatter3 {
            points,
            point_sizes,
            ..
        } = &scene.plots[0]
        else {
            panic!("expected scatter3 plot");
        };
        assert!(points[0][2].is_nan());
        assert!(points[1][1].is_nan());
        assert!(point_sizes.as_ref().unwrap()[1].is_nan());
    }
}
