//! Figure management for multiple overlaid plots
//!
//! This module provides the `Figure` struct that manages multiple plots in a single
//! coordinate system, handling overlays, legends, and proper rendering order.

use crate::core::{BoundingBox, RenderData};
use crate::plots::{BarChart, Histogram, LinePlot, ScatterPlot};
use glam::Vec4;
use std::collections::HashMap;

/// A figure that can contain multiple overlaid plots
#[derive(Debug, Clone)]
pub struct Figure {
    /// All plots in this figure
    plots: Vec<PlotElement>,

    /// Figure-level settings
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub legend_enabled: bool,
    pub grid_enabled: bool,
    pub background_color: Vec4,

    /// Axis limits (None = auto-scale)
    pub x_limits: Option<(f64, f64)>,
    pub y_limits: Option<(f64, f64)>,

    /// Cached data
    bounds: Option<BoundingBox>,
    dirty: bool,
}

/// A plot element that can be any type of plot
#[derive(Debug, Clone)]
pub enum PlotElement {
    Line(LinePlot),
    Scatter(ScatterPlot),
    Bar(BarChart),
    Histogram(Histogram),
}

/// Legend entry for a plot
#[derive(Debug, Clone)]
pub struct LegendEntry {
    pub label: String,
    pub color: Vec4,
    pub plot_type: PlotType,
}

/// Type of plot for legend rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlotType {
    Line,
    Scatter,
    Bar,
    Histogram,
}

impl Figure {
    /// Create a new empty figure
    pub fn new() -> Self {
        Self {
            plots: Vec::new(),
            title: None,
            x_label: None,
            y_label: None,
            legend_enabled: true,
            grid_enabled: true,
            background_color: Vec4::new(1.0, 1.0, 1.0, 1.0), // White background
            x_limits: None,
            y_limits: None,
            bounds: None,
            dirty: true,
        }
    }

    /// Set the figure title
    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set axis labels
    pub fn with_labels<S: Into<String>>(mut self, x_label: S, y_label: S) -> Self {
        self.x_label = Some(x_label.into());
        self.y_label = Some(y_label.into());
        self
    }

    /// Set axis limits manually
    pub fn with_limits(mut self, x_limits: (f64, f64), y_limits: (f64, f64)) -> Self {
        self.x_limits = Some(x_limits);
        self.y_limits = Some(y_limits);
        self.dirty = true;
        self
    }

    /// Enable or disable the legend
    pub fn with_legend(mut self, enabled: bool) -> Self {
        self.legend_enabled = enabled;
        self
    }

    /// Enable or disable the grid
    pub fn with_grid(mut self, enabled: bool) -> Self {
        self.grid_enabled = enabled;
        self
    }

    /// Set background color
    pub fn with_background_color(mut self, color: Vec4) -> Self {
        self.background_color = color;
        self
    }

    /// Add a line plot to the figure
    pub fn add_line_plot(&mut self, plot: LinePlot) -> usize {
        self.plots.push(PlotElement::Line(plot));
        self.dirty = true;
        self.plots.len() - 1
    }

    /// Add a scatter plot to the figure
    pub fn add_scatter_plot(&mut self, plot: ScatterPlot) -> usize {
        self.plots.push(PlotElement::Scatter(plot));
        self.dirty = true;
        self.plots.len() - 1
    }

    /// Add a bar chart to the figure
    pub fn add_bar_chart(&mut self, plot: BarChart) -> usize {
        self.plots.push(PlotElement::Bar(plot));
        self.dirty = true;
        self.plots.len() - 1
    }

    /// Add a histogram to the figure
    pub fn add_histogram(&mut self, plot: Histogram) -> usize {
        self.plots.push(PlotElement::Histogram(plot));
        self.dirty = true;
        self.plots.len() - 1
    }

    /// Remove a plot by index
    pub fn remove_plot(&mut self, index: usize) -> Result<(), String> {
        if index >= self.plots.len() {
            return Err(format!("Plot index {} out of bounds", index));
        }
        self.plots.remove(index);
        self.dirty = true;
        Ok(())
    }

    /// Clear all plots
    pub fn clear(&mut self) {
        self.plots.clear();
        self.dirty = true;
    }

    /// Get the number of plots
    pub fn len(&self) -> usize {
        self.plots.len()
    }

    /// Check if figure has no plots
    pub fn is_empty(&self) -> bool {
        self.plots.is_empty()
    }

    /// Get an iterator over all plots in this figure
    pub fn plots(&self) -> impl Iterator<Item = &PlotElement> {
        self.plots.iter()
    }

    /// Get a mutable reference to a plot
    pub fn get_plot_mut(&mut self, index: usize) -> Option<&mut PlotElement> {
        self.dirty = true;
        self.plots.get_mut(index)
    }

    /// Get the combined bounds of all visible plots
    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            self.compute_bounds();
        }
        self.bounds.unwrap()
    }

    /// Compute the combined bounds from all plots
    fn compute_bounds(&mut self) {
        if self.plots.is_empty() {
            self.bounds = Some(BoundingBox::default());
            return;
        }

        let mut combined_bounds = None;

        for plot in &mut self.plots {
            if !plot.is_visible() {
                continue;
            }

            let plot_bounds = plot.bounds();

            combined_bounds = match combined_bounds {
                None => Some(plot_bounds),
                Some(existing) => Some(existing.union(&plot_bounds)),
            };
        }

        self.bounds = combined_bounds.or_else(|| Some(BoundingBox::default()));
        self.dirty = false;
    }

    /// Generate all render data for all visible plots
    pub fn render_data(&mut self) -> Vec<RenderData> {
        let mut render_data = Vec::new();

        for plot in &mut self.plots {
            if plot.is_visible() {
                render_data.push(plot.render_data());
            }
        }

        render_data
    }

    /// Get legend entries for all labeled plots
    pub fn legend_entries(&self) -> Vec<LegendEntry> {
        let mut entries = Vec::new();

        for plot in &self.plots {
            if let Some(label) = plot.label() {
                entries.push(LegendEntry {
                    label,
                    color: plot.color(),
                    plot_type: plot.plot_type(),
                });
            }
        }

        entries
    }

    /// Get figure statistics
    pub fn statistics(&self) -> FigureStatistics {
        let plot_counts = self.plots.iter().fold(HashMap::new(), |mut acc, plot| {
            let plot_type = plot.plot_type();
            *acc.entry(plot_type).or_insert(0) += 1;
            acc
        });

        let total_memory: usize = self
            .plots
            .iter()
            .map(|plot| plot.estimated_memory_usage())
            .sum();

        let visible_count = self.plots.iter().filter(|plot| plot.is_visible()).count();

        FigureStatistics {
            total_plots: self.plots.len(),
            visible_plots: visible_count,
            plot_type_counts: plot_counts,
            total_memory_usage: total_memory,
            has_legend: self.legend_enabled && !self.legend_entries().is_empty(),
        }
    }
}

impl Default for Figure {
    fn default() -> Self {
        Self::new()
    }
}

impl PlotElement {
    /// Check if the plot is visible
    pub fn is_visible(&self) -> bool {
        match self {
            PlotElement::Line(plot) => plot.visible,
            PlotElement::Scatter(plot) => plot.visible,
            PlotElement::Bar(plot) => plot.visible,
            PlotElement::Histogram(plot) => plot.visible,
        }
    }

    /// Get the plot's label
    pub fn label(&self) -> Option<String> {
        match self {
            PlotElement::Line(plot) => plot.label.clone(),
            PlotElement::Scatter(plot) => plot.label.clone(),
            PlotElement::Bar(plot) => plot.label.clone(),
            PlotElement::Histogram(plot) => plot.label.clone(),
        }
    }

    /// Get the plot's primary color
    pub fn color(&self) -> Vec4 {
        match self {
            PlotElement::Line(plot) => plot.color,
            PlotElement::Scatter(plot) => plot.color,
            PlotElement::Bar(plot) => plot.color,
            PlotElement::Histogram(plot) => plot.color,
        }
    }

    /// Get the plot type
    pub fn plot_type(&self) -> PlotType {
        match self {
            PlotElement::Line(_) => PlotType::Line,
            PlotElement::Scatter(_) => PlotType::Scatter,
            PlotElement::Bar(_) => PlotType::Bar,
            PlotElement::Histogram(_) => PlotType::Histogram,
        }
    }

    /// Get the plot's bounds
    pub fn bounds(&mut self) -> BoundingBox {
        match self {
            PlotElement::Line(plot) => plot.bounds(),
            PlotElement::Scatter(plot) => plot.bounds(),
            PlotElement::Bar(plot) => plot.bounds(),
            PlotElement::Histogram(plot) => plot.bounds(),
        }
    }

    /// Generate render data for this plot
    pub fn render_data(&mut self) -> RenderData {
        match self {
            PlotElement::Line(plot) => plot.render_data(),
            PlotElement::Scatter(plot) => plot.render_data(),
            PlotElement::Bar(plot) => plot.render_data(),
            PlotElement::Histogram(plot) => plot.render_data(),
        }
    }

    /// Estimate memory usage
    pub fn estimated_memory_usage(&self) -> usize {
        match self {
            PlotElement::Line(plot) => plot.estimated_memory_usage(),
            PlotElement::Scatter(plot) => plot.estimated_memory_usage(),
            PlotElement::Bar(plot) => plot.estimated_memory_usage(),
            PlotElement::Histogram(plot) => plot.estimated_memory_usage(),
        }
    }
}

/// Figure statistics for debugging and optimization
#[derive(Debug)]
pub struct FigureStatistics {
    pub total_plots: usize,
    pub visible_plots: usize,
    pub plot_type_counts: HashMap<PlotType, usize>,
    pub total_memory_usage: usize,
    pub has_legend: bool,
}

/// MATLAB-compatible figure creation utilities
pub mod matlab_compat {
    use super::*;
    use crate::plots::{LinePlot, ScatterPlot};

    /// Create a new figure (equivalent to MATLAB's `figure`)
    pub fn figure() -> Figure {
        Figure::new()
    }

    /// Create a figure with a title
    pub fn figure_with_title<S: Into<String>>(title: S) -> Figure {
        Figure::new().with_title(title)
    }

    /// Add multiple line plots to a figure (`hold on` behavior)
    pub fn plot_multiple_lines(
        figure: &mut Figure,
        data_sets: Vec<(Vec<f64>, Vec<f64>, Option<String>)>,
    ) -> Result<Vec<usize>, String> {
        let mut indices = Vec::new();

        for (i, (x, y, label)) in data_sets.into_iter().enumerate() {
            let mut line = LinePlot::new(x, y)?;

            // Automatic color cycling (similar to MATLAB)
            let colors = [
                Vec4::new(0.0, 0.4470, 0.7410, 1.0),    // Blue
                Vec4::new(0.8500, 0.3250, 0.0980, 1.0), // Orange
                Vec4::new(0.9290, 0.6940, 0.1250, 1.0), // Yellow
                Vec4::new(0.4940, 0.1840, 0.5560, 1.0), // Purple
                Vec4::new(0.4660, 0.6740, 0.1880, 1.0), // Green
                Vec4::new(0.3010, 0.7450, 0.9330, 1.0), // Cyan
                Vec4::new(0.6350, 0.0780, 0.1840, 1.0), // Red
            ];
            let color = colors[i % colors.len()];
            line.set_color(color);

            if let Some(label) = label {
                line = line.with_label(label);
            }

            indices.push(figure.add_line_plot(line));
        }

        Ok(indices)
    }

    /// Add multiple scatter plots to a figure
    pub fn scatter_multiple(
        figure: &mut Figure,
        data_sets: Vec<(Vec<f64>, Vec<f64>, Option<String>)>,
    ) -> Result<Vec<usize>, String> {
        let mut indices = Vec::new();

        for (i, (x, y, label)) in data_sets.into_iter().enumerate() {
            let mut scatter = ScatterPlot::new(x, y)?;

            // Automatic color cycling
            let colors = [
                Vec4::new(1.0, 0.0, 0.0, 1.0), // Red
                Vec4::new(0.0, 1.0, 0.0, 1.0), // Green
                Vec4::new(0.0, 0.0, 1.0, 1.0), // Blue
                Vec4::new(1.0, 1.0, 0.0, 1.0), // Yellow
                Vec4::new(1.0, 0.0, 1.0, 1.0), // Magenta
                Vec4::new(0.0, 1.0, 1.0, 1.0), // Cyan
                Vec4::new(0.5, 0.5, 0.5, 1.0), // Gray
            ];
            let color = colors[i % colors.len()];
            scatter.set_color(color);

            if let Some(label) = label {
                scatter = scatter.with_label(label);
            }

            indices.push(figure.add_scatter_plot(scatter));
        }

        Ok(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plots::line::LineStyle;

    #[test]
    fn test_figure_creation() {
        let figure = Figure::new();

        assert_eq!(figure.len(), 0);
        assert!(figure.is_empty());
        assert!(figure.legend_enabled);
        assert!(figure.grid_enabled);
    }

    #[test]
    fn test_figure_styling() {
        let figure = Figure::new()
            .with_title("Test Figure")
            .with_labels("X Axis", "Y Axis")
            .with_legend(false)
            .with_grid(false);

        assert_eq!(figure.title, Some("Test Figure".to_string()));
        assert_eq!(figure.x_label, Some("X Axis".to_string()));
        assert_eq!(figure.y_label, Some("Y Axis".to_string()));
        assert!(!figure.legend_enabled);
        assert!(!figure.grid_enabled);
    }

    #[test]
    fn test_multiple_line_plots() {
        let mut figure = Figure::new();

        // Add first line plot
        let line1 = LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 4.0])
            .unwrap()
            .with_label("Quadratic");
        let index1 = figure.add_line_plot(line1);

        // Add second line plot
        let line2 = LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0])
            .unwrap()
            .with_style(Vec4::new(1.0, 0.0, 0.0, 1.0), 2.0, LineStyle::Dashed)
            .with_label("Linear");
        let index2 = figure.add_line_plot(line2);

        assert_eq!(figure.len(), 2);
        assert_eq!(index1, 0);
        assert_eq!(index2, 1);

        // Test legend entries
        let legend = figure.legend_entries();
        assert_eq!(legend.len(), 2);
        assert_eq!(legend[0].label, "Quadratic");
        assert_eq!(legend[1].label, "Linear");
    }

    #[test]
    fn test_mixed_plot_types() {
        let mut figure = Figure::new();

        // Add different plot types
        let line = LinePlot::new(vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0])
            .unwrap()
            .with_label("Line");
        figure.add_line_plot(line);

        let scatter = ScatterPlot::new(vec![0.5, 1.5, 2.5], vec![1.5, 2.5, 3.5])
            .unwrap()
            .with_label("Scatter");
        figure.add_scatter_plot(scatter);

        let bar = BarChart::new(vec!["A".to_string(), "B".to_string()], vec![2.0, 4.0])
            .unwrap()
            .with_label("Bar");
        figure.add_bar_chart(bar);

        assert_eq!(figure.len(), 3);

        // Test render data generation
        let render_data = figure.render_data();
        assert_eq!(render_data.len(), 3);

        // Test statistics
        let stats = figure.statistics();
        assert_eq!(stats.total_plots, 3);
        assert_eq!(stats.visible_plots, 3);
        assert!(stats.has_legend);
    }

    #[test]
    fn test_plot_visibility() {
        let mut figure = Figure::new();

        let mut line = LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        line.set_visible(false); // Hide this plot
        figure.add_line_plot(line);

        let scatter = ScatterPlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
        figure.add_scatter_plot(scatter);

        // Only one plot should be visible
        let render_data = figure.render_data();
        assert_eq!(render_data.len(), 1);

        let stats = figure.statistics();
        assert_eq!(stats.total_plots, 2);
        assert_eq!(stats.visible_plots, 1);
    }

    #[test]
    fn test_bounds_computation() {
        let mut figure = Figure::new();

        // Add plots with different ranges
        let line = LinePlot::new(vec![-1.0, 0.0, 1.0], vec![-2.0, 0.0, 2.0]).unwrap();
        figure.add_line_plot(line);

        let scatter = ScatterPlot::new(vec![2.0, 3.0, 4.0], vec![1.0, 3.0, 5.0]).unwrap();
        figure.add_scatter_plot(scatter);

        let bounds = figure.bounds();

        // Bounds should encompass all plots
        assert!(bounds.min.x <= -1.0);
        assert!(bounds.max.x >= 4.0);
        assert!(bounds.min.y <= -2.0);
        assert!(bounds.max.y >= 5.0);
    }

    #[test]
    fn test_matlab_compat_multiple_lines() {
        use super::matlab_compat::*;

        let mut figure = figure_with_title("Multiple Lines Test");

        let data_sets = vec![
            (
                vec![0.0, 1.0, 2.0],
                vec![0.0, 1.0, 4.0],
                Some("Quadratic".to_string()),
            ),
            (
                vec![0.0, 1.0, 2.0],
                vec![0.0, 1.0, 2.0],
                Some("Linear".to_string()),
            ),
            (
                vec![0.0, 1.0, 2.0],
                vec![1.0, 1.0, 1.0],
                Some("Constant".to_string()),
            ),
        ];

        let indices = plot_multiple_lines(&mut figure, data_sets).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(figure.len(), 3);

        // Each plot should have different colors
        let legend = figure.legend_entries();
        assert_eq!(legend.len(), 3);
        assert_ne!(legend[0].color, legend[1].color);
        assert_ne!(legend[1].color, legend[2].color);
    }
}
