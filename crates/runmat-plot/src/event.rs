use crate::plots::{
    AreaPlot, ErrorBar, Figure, LegendEntry, LinePlot, MarkerStyle, PlotElement, PlotType,
    ScatterPlot, StairsPlot, StemPlot,
};
use glam::Vec4;
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
        x: Vec<f64>,
        y: Vec<f64>,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Scatter {
        x: Vec<f64>,
        y: Vec<f64>,
        color_rgba: [f32; 4],
        marker_size: f32,
        marker_style: String,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    ErrorBar {
        x: Vec<f64>,
        y: Vec<f64>,
        err_low: Vec<f64>,
        err_high: Vec<f64>,
        color_rgba: [f32; 4],
        line_width: f32,
        cap_width: f32,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Stairs {
        x: Vec<f64>,
        y: Vec<f64>,
        color_rgba: [f32; 4],
        line_width: f32,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Stem {
        x: Vec<f64>,
        y: Vec<f64>,
        baseline: f64,
        color_rgba: [f32; 4],
        marker_color_rgba: [f32; 4],
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Area {
        x: Vec<f64>,
        y: Vec<f64>,
        baseline: f64,
        color_rgba: [f32; 4],
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
        figure.title = self.metadata.title;
        figure.x_label = self.metadata.x_label;
        figure.y_label = self.metadata.y_label;
        figure.grid_enabled = self.metadata.grid_enabled;
        figure.legend_enabled = self.metadata.legend_enabled;
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
    pub legend_entries: Vec<FigureLegendEntry>,
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
            legend_entries,
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
            PlotElement::ErrorBar(error) => Self::ErrorBar {
                x: error.x.clone(),
                y: error.y.clone(),
                err_low: error.err_low.clone(),
                err_high: error.err_high.clone(),
                color_rgba: vec4_to_rgba(error.color),
                line_width: error.line_width,
                cap_width: error.cap_width,
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
                marker_color_rgba: vec4_to_rgba(stem.marker_color),
                axes_index,
                label: stem.label.clone(),
                visible: stem.visible,
            },
            PlotElement::Area(area) => Self::Area {
                x: area.x.clone(),
                y: area.y.clone(),
                baseline: area.baseline,
                color_rgba: vec4_to_rgba(area.color),
                axes_index,
                label: area.label.clone(),
                visible: area.visible,
            },
            _ => Self::Unsupported {
                plot_kind: PlotKind::from(plot.plot_type()),
                axes_index,
                label: plot.label(),
                visible: plot.is_visible(),
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
            ScenePlot::ErrorBar {
                x,
                y,
                err_low,
                err_high,
                color_rgba,
                line_width,
                cap_width,
                axes_index,
                label,
                visible,
            } => {
                let mut error = ErrorBar::new(x, y, err_low, err_high)?;
                error.color = rgba_to_vec4(color_rgba);
                error.line_width = line_width;
                error.cap_width = cap_width;
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
                marker_color_rgba,
                axes_index,
                label,
                visible,
            } => {
                let mut stem = StemPlot::new(x, y)?;
                stem.baseline = baseline;
                stem.color = rgba_to_vec4(color_rgba);
                stem.marker_color = rgba_to_vec4(marker_color_rgba);
                stem.label = label;
                stem.set_visible(visible);
                figure.add_stem_plot_on_axes(stem, axes_index as usize);
            }
            ScenePlot::Area {
                x,
                y,
                baseline,
                color_rgba,
                axes_index,
                label,
                visible,
            } => {
                let mut area = AreaPlot::new(x, y)?;
                area.baseline = baseline;
                area.color = rgba_to_vec4(color_rgba);
                area.label = label;
                area.set_visible(visible);
                figure.add_area_plot_on_axes(area, axes_index as usize);
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

fn rgba_to_vec4(value: [f32; 4]) -> Vec4 {
    Vec4::new(value[0], value[1], value[2], value[3])
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
            PlotType::Scatter => Self::Scatter,
            PlotType::Bar => Self::Bar,
            PlotType::ErrorBar => Self::ErrorBar,
            PlotType::Stairs => Self::Stairs,
            PlotType::Stem => Self::Stem,
            PlotType::Area => Self::Area,
            PlotType::Quiver => Self::Quiver,
            PlotType::Pie => Self::Pie,
            PlotType::Image => Self::Image,
            PlotType::Surface => Self::Surface,
            PlotType::Scatter3 => Self::Scatter3,
            PlotType::Contour => Self::Contour,
            PlotType::ContourFill => Self::ContourFill,
        }
    }
}

fn vec4_to_rgba(value: Vec4) -> [f32; 4] {
    [value.x, value.y, value.z, value.w]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plots::{Figure, LinePlot, ScatterPlot};

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
}
