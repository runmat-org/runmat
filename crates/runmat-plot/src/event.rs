use crate::plots::{Figure, LegendEntry, PlotElement, PlotType};
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
    use crate::plots::{Figure, LinePlot};

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
}
