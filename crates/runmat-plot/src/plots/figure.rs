//! Figure management for multiple overlaid plots
//!
//! This module provides the `Figure` struct that manages multiple plots in a single
//! coordinate system, handling overlays, legends, and proper rendering order.

use crate::core::{BoundingBox, GpuPackContext, RenderData};
use crate::plots::surface::ColorMap;
use crate::plots::{
    AreaPlot, BarChart, ContourFillPlot, ContourPlot, ErrorBar, ImagePlot, LinePlot, PieChart,
    QuiverPlot, Scatter3Plot, ScatterPlot, StairsPlot, StemPlot, SurfacePlot,
};
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

    /// Axis scales
    pub x_log: bool,
    pub y_log: bool,

    /// Axis aspect handling
    pub axis_equal: bool,

    /// Global colormap and colorbar
    pub colormap: ColorMap,
    pub colorbar_enabled: bool,

    /// Color mapping limits for all color-mapped plots in this figure (caxis)
    pub color_limits: Option<(f64, f64)>,

    /// Cached data
    bounds: Option<BoundingBox>,
    dirty: bool,

    /// Subplot grid configuration (rows x cols). Defaults to 1x1.
    pub axes_rows: usize,
    pub axes_cols: usize,
    /// For each plot element, the axes index (row-major, 0..rows*cols-1)
    plot_axes_indices: Vec<usize>,
}

/// A plot element that can be any type of plot
#[derive(Debug, Clone)]
pub enum PlotElement {
    Line(LinePlot),
    Scatter(ScatterPlot),
    Bar(BarChart),
    ErrorBar(ErrorBar),
    Stairs(StairsPlot),
    Stem(StemPlot),
    Area(AreaPlot),
    Quiver(QuiverPlot),
    Pie(PieChart),
    Image(ImagePlot),
    Surface(SurfacePlot),
    Scatter3(Scatter3Plot),
    Contour(ContourPlot),
    ContourFill(ContourFillPlot),
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
            x_log: false,
            y_log: false,
            axis_equal: false,
            colormap: ColorMap::Parula,
            colorbar_enabled: false,
            color_limits: None,
            bounds: None,
            dirty: true,
            axes_rows: 1,
            axes_cols: 1,
            plot_axes_indices: Vec::new(),
        }
    }

    /// Set the figure title
    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.set_title(title);
        self
    }

    /// Set the figure title in-place
    pub fn set_title<S: Into<String>>(&mut self, title: S) {
        self.title = Some(title.into());
        self.dirty = true;
    }

    /// Set axis labels
    pub fn with_labels<S: Into<String>>(mut self, x_label: S, y_label: S) -> Self {
        self.set_axis_labels(x_label, y_label);
        self
    }

    /// Set axis labels in-place
    pub fn set_axis_labels<S: Into<String>>(&mut self, x_label: S, y_label: S) {
        self.x_label = Some(x_label.into());
        self.y_label = Some(y_label.into());
        self.dirty = true;
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
        self.set_grid(enabled);
        self
    }

    pub fn set_grid(&mut self, enabled: bool) {
        self.grid_enabled = enabled;
        self.dirty = true;
    }

    /// Set background color
    pub fn with_background_color(mut self, color: Vec4) -> Self {
        self.background_color = color;
        self
    }

    /// Set log scale flags
    pub fn with_xlog(mut self, enabled: bool) -> Self {
        self.x_log = enabled;
        self
    }
    pub fn with_ylog(mut self, enabled: bool) -> Self {
        self.y_log = enabled;
        self
    }
    pub fn with_axis_equal(mut self, enabled: bool) -> Self {
        self.set_axis_equal(enabled);
        self
    }

    pub fn set_axis_equal(&mut self, enabled: bool) {
        self.axis_equal = enabled;
        self.dirty = true;
    }
    pub fn with_colormap(mut self, cmap: ColorMap) -> Self {
        self.colormap = cmap;
        self
    }
    pub fn with_colorbar(mut self, enabled: bool) -> Self {
        self.colorbar_enabled = enabled;
        self
    }
    pub fn with_color_limits(mut self, limits: Option<(f64, f64)>) -> Self {
        self.color_limits = limits;
        self
    }

    /// Configure subplot grid (rows x cols). Axes are indexed row-major starting at 0.
    pub fn with_subplot_grid(mut self, rows: usize, cols: usize) -> Self {
        self.set_subplot_grid(rows, cols);
        self
    }

    /// Return subplot grid (rows, cols)
    pub fn axes_grid(&self) -> (usize, usize) {
        (self.axes_rows, self.axes_cols)
    }

    /// Axes index mapping for plots (length equals number of plots)
    pub fn plot_axes_indices(&self) -> &[usize] {
        &self.plot_axes_indices
    }

    /// Assign a specific plot (by index) to an axes index in the subplot grid
    pub fn assign_plot_to_axes(
        &mut self,
        plot_index: usize,
        axes_index: usize,
    ) -> Result<(), String> {
        if plot_index >= self.plot_axes_indices.len() {
            return Err(format!(
                "assign_plot_to_axes: index {plot_index} out of bounds"
            ));
        }
        let max_axes = self.axes_rows.max(1) * self.axes_cols.max(1);
        let ai = axes_index.min(max_axes.saturating_sub(1));
        self.plot_axes_indices[plot_index] = ai;
        self.dirty = true;
        Ok(())
    }
    /// Mutably set subplot grid (rows x cols)
    pub fn set_subplot_grid(&mut self, rows: usize, cols: usize) {
        self.axes_rows = rows.max(1);
        self.axes_cols = cols.max(1);
        self.dirty = true;
    }

    /// Set color limits and propagate to existing surface plots
    pub fn set_color_limits(&mut self, limits: Option<(f64, f64)>) {
        self.color_limits = limits;
        for plot in &mut self.plots {
            if let PlotElement::Surface(s) = plot {
                s.set_color_limits(limits);
            }
        }
        self.dirty = true;
    }

    fn total_axes(&self) -> usize {
        self.axes_rows.max(1) * self.axes_cols.max(1)
    }

    fn normalize_axes_index(&self, axes_index: usize) -> usize {
        let total = self.total_axes().max(1);
        axes_index.min(total - 1)
    }

    fn push_plot(&mut self, element: PlotElement, axes_index: usize) -> usize {
        let idx = self.normalize_axes_index(axes_index);
        self.plots.push(element);
        self.plot_axes_indices.push(idx);
        self.dirty = true;
        self.plots.len() - 1
    }

    /// Add a line plot to the figure
    pub fn add_line_plot(&mut self, plot: LinePlot) -> usize {
        self.add_line_plot_on_axes(plot, 0)
    }

    pub fn add_line_plot_on_axes(&mut self, plot: LinePlot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Line(plot), axes_index)
    }

    /// Add a scatter plot to the figure
    pub fn add_scatter_plot(&mut self, plot: ScatterPlot) -> usize {
        self.add_scatter_plot_on_axes(plot, 0)
    }

    pub fn add_scatter_plot_on_axes(&mut self, plot: ScatterPlot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Scatter(plot), axes_index)
    }

    /// Add a bar chart to the figure
    pub fn add_bar_chart(&mut self, plot: BarChart) -> usize {
        self.add_bar_chart_on_axes(plot, 0)
    }

    pub fn add_bar_chart_on_axes(&mut self, plot: BarChart, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Bar(plot), axes_index)
    }

    /// Add an errorbar plot
    pub fn add_errorbar(&mut self, plot: ErrorBar) -> usize {
        self.add_errorbar_on_axes(plot, 0)
    }

    pub fn add_errorbar_on_axes(&mut self, plot: ErrorBar, axes_index: usize) -> usize {
        self.push_plot(PlotElement::ErrorBar(plot), axes_index)
    }

    /// Add a stairs plot
    pub fn add_stairs_plot(&mut self, plot: StairsPlot) -> usize {
        self.add_stairs_plot_on_axes(plot, 0)
    }

    pub fn add_stairs_plot_on_axes(&mut self, plot: StairsPlot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Stairs(plot), axes_index)
    }

    /// Add a stem plot
    pub fn add_stem_plot(&mut self, plot: StemPlot) -> usize {
        self.add_stem_plot_on_axes(plot, 0)
    }

    pub fn add_stem_plot_on_axes(&mut self, plot: StemPlot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Stem(plot), axes_index)
    }

    /// Add an area plot
    pub fn add_area_plot(&mut self, plot: AreaPlot) -> usize {
        self.add_area_plot_on_axes(plot, 0)
    }

    pub fn add_area_plot_on_axes(&mut self, plot: AreaPlot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Area(plot), axes_index)
    }

    pub fn add_image_plot(&mut self, plot: ImagePlot) -> usize {
        self.add_image_plot_on_axes(plot, 0)
    }

    pub fn add_image_plot_on_axes(&mut self, plot: ImagePlot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Image(plot), axes_index)
    }

    pub fn add_quiver_plot(&mut self, plot: QuiverPlot) -> usize {
        self.add_quiver_plot_on_axes(plot, 0)
    }

    pub fn add_quiver_plot_on_axes(&mut self, plot: QuiverPlot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Quiver(plot), axes_index)
    }

    pub fn add_pie_chart(&mut self, plot: PieChart) -> usize {
        self.add_pie_chart_on_axes(plot, 0)
    }

    pub fn add_pie_chart_on_axes(&mut self, plot: PieChart, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Pie(plot), axes_index)
    }

    /// Add a surface plot to the figure
    pub fn add_surface_plot(&mut self, plot: SurfacePlot) -> usize {
        self.add_surface_plot_on_axes(plot, 0)
    }

    pub fn add_surface_plot_on_axes(&mut self, plot: SurfacePlot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Surface(plot), axes_index)
    }

    /// Add a 3D scatter plot to the figure
    pub fn add_scatter3_plot(&mut self, plot: Scatter3Plot) -> usize {
        self.add_scatter3_plot_on_axes(plot, 0)
    }

    pub fn add_scatter3_plot_on_axes(&mut self, plot: Scatter3Plot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Scatter3(plot), axes_index)
    }

    pub fn add_contour_plot(&mut self, plot: ContourPlot) -> usize {
        self.add_contour_plot_on_axes(plot, 0)
    }

    pub fn add_contour_plot_on_axes(&mut self, plot: ContourPlot, axes_index: usize) -> usize {
        self.push_plot(PlotElement::Contour(plot), axes_index)
    }

    pub fn add_contour_fill_plot(&mut self, plot: ContourFillPlot) -> usize {
        self.add_contour_fill_plot_on_axes(plot, 0)
    }

    pub fn add_contour_fill_plot_on_axes(
        &mut self,
        plot: ContourFillPlot,
        axes_index: usize,
    ) -> usize {
        self.push_plot(PlotElement::ContourFill(plot), axes_index)
    }

    /// Remove a plot by index
    pub fn remove_plot(&mut self, index: usize) -> Result<(), String> {
        if index >= self.plots.len() {
            return Err(format!("Plot index {index} out of bounds"));
        }
        self.plots.remove(index);
        self.plot_axes_indices.remove(index);
        self.dirty = true;
        Ok(())
    }

    /// Clear all plots
    pub fn clear(&mut self) {
        self.plots.clear();
        self.plot_axes_indices.clear();
        self.dirty = true;
    }

    /// Clear all plots assigned to a specific axes index
    pub fn clear_axes(&mut self, axes_index: usize) {
        let mut i = 0usize;
        while i < self.plots.len() {
            let ax = *self.plot_axes_indices.get(i).unwrap_or(&0);
            if ax == axes_index {
                self.plots.remove(i);
                self.plot_axes_indices.remove(i);
            } else {
                i += 1;
            }
        }
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
        self.render_data_with_viewport(None)
    }

    /// Generate all render data for all visible plots, optionally providing the
    /// pixel size of the target viewport (width, height).
    ///
    /// Some plot types (notably thick 2D lines) need a viewport hint to convert
    /// pixel-based style parameters (e.g. `LineWidth`) into data-space geometry.
    pub fn render_data_with_viewport(
        &mut self,
        viewport_px: Option<(u32, u32)>,
    ) -> Vec<RenderData> {
        self.render_data_with_viewport_and_gpu(viewport_px, None)
    }

    pub fn render_data_with_viewport_and_gpu(
        &mut self,
        viewport_px: Option<(u32, u32)>,
        gpu: Option<&GpuPackContext<'_>>,
    ) -> Vec<RenderData> {
        let mut out = Vec::new();
        for p in self.plots.iter_mut() {
            if !p.is_visible() {
                continue;
            }
            // Apply figure-level color limits to surfaces before generating
            if let PlotElement::Surface(s) = p {
                if self.color_limits.is_some() {
                    s.set_color_limits(self.color_limits);
                }
            }

            match p {
                PlotElement::Line(plot) => {
                    out.push(plot.render_data_with_viewport_gpu(viewport_px, gpu));
                    if let Some(marker_data) = plot.marker_render_data() {
                        out.push(marker_data);
                    }
                }
                PlotElement::Stairs(plot) => {
                    out.push(plot.render_data());
                    if let Some(marker_data) = plot.marker_render_data() {
                        out.push(marker_data);
                    }
                }
                _ => out.push(p.render_data()),
            }
        }
        out
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

    /// Assign labels to visible plots in order
    pub fn set_labels(&mut self, labels: &[String]) {
        let mut idx = 0usize;
        for plot in &mut self.plots {
            if !plot.is_visible() {
                continue;
            }
            if idx >= labels.len() {
                break;
            }
            plot.set_label(Some(labels[idx].clone()));
            idx += 1;
        }
        self.dirty = true;
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

    /// If the figure contains a bar/barh plot, return its categorical axis labels.
    /// Returns (is_x_axis, labels) where is_x_axis=true means X is categorical (vertical bars),
    /// false means Y is categorical (horizontal bars).
    pub fn categorical_axis_labels(&self) -> Option<(bool, Vec<String>)> {
        for plot in &self.plots {
            if let PlotElement::Bar(b) = plot {
                let is_x = matches!(b.orientation, crate::plots::bar::Orientation::Vertical);
                return Some((is_x, b.labels.clone()));
            }
        }
        None
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
            PlotElement::ErrorBar(plot) => plot.visible,
            PlotElement::Stairs(plot) => plot.visible,
            PlotElement::Stem(plot) => plot.visible,
            PlotElement::Area(plot) => plot.visible,
            PlotElement::Quiver(plot) => plot.visible,
            PlotElement::Pie(plot) => plot.visible,
            PlotElement::Image(plot) => plot.visible,
            PlotElement::Surface(plot) => plot.visible,
            PlotElement::Scatter3(plot) => plot.visible,
            PlotElement::Contour(plot) => plot.visible,
            PlotElement::ContourFill(plot) => plot.visible,
        }
    }

    /// Get the plot's label
    pub fn label(&self) -> Option<String> {
        match self {
            PlotElement::Line(plot) => plot.label.clone(),
            PlotElement::Scatter(plot) => plot.label.clone(),
            PlotElement::Bar(plot) => plot.label.clone(),
            PlotElement::ErrorBar(plot) => plot.label.clone(),
            PlotElement::Stairs(plot) => plot.label.clone(),
            PlotElement::Stem(plot) => plot.label.clone(),
            PlotElement::Area(plot) => plot.label.clone(),
            PlotElement::Quiver(plot) => plot.label.clone(),
            PlotElement::Pie(plot) => plot.label.clone(),
            PlotElement::Image(plot) => plot.label.clone(),
            PlotElement::Surface(plot) => plot.label.clone(),
            PlotElement::Scatter3(plot) => plot.label.clone(),
            PlotElement::Contour(plot) => plot.label.clone(),
            PlotElement::ContourFill(plot) => plot.label.clone(),
        }
    }

    /// Mutate label
    pub fn set_label(&mut self, label: Option<String>) {
        match self {
            PlotElement::Line(plot) => plot.label = label,
            PlotElement::Scatter(plot) => plot.label = label,
            PlotElement::Bar(plot) => plot.label = label,
            PlotElement::ErrorBar(plot) => plot.label = label,
            PlotElement::Stairs(plot) => plot.label = label,
            PlotElement::Stem(plot) => plot.label = label,
            PlotElement::Area(plot) => plot.label = label,
            PlotElement::Quiver(plot) => plot.label = label,
            PlotElement::Pie(plot) => plot.label = label,
            PlotElement::Image(plot) => plot.label = label,
            PlotElement::Surface(plot) => plot.label = label,
            PlotElement::Scatter3(plot) => plot.label = label,
            PlotElement::Contour(plot) => plot.label = label,
            PlotElement::ContourFill(plot) => plot.label = label,
        }
    }

    /// Get the plot's primary color
    pub fn color(&self) -> Vec4 {
        match self {
            PlotElement::Line(plot) => plot.color,
            PlotElement::Scatter(plot) => plot.color,
            PlotElement::Bar(plot) => plot.color,
            PlotElement::ErrorBar(plot) => plot.color,
            PlotElement::Stairs(plot) => plot.color,
            PlotElement::Stem(plot) => plot.color,
            PlotElement::Area(plot) => plot.color,
            PlotElement::Quiver(plot) => plot.color,
            PlotElement::Pie(_plot) => Vec4::new(1.0, 1.0, 1.0, 1.0),
            PlotElement::Image(_plot) => Vec4::new(1.0, 1.0, 1.0, 1.0),
            PlotElement::Surface(_plot) => Vec4::new(1.0, 1.0, 1.0, 1.0),
            PlotElement::Scatter3(plot) => plot.colors.first().copied().unwrap_or(Vec4::ONE),
            PlotElement::Contour(_plot) => Vec4::new(1.0, 1.0, 1.0, 1.0),
            PlotElement::ContourFill(_plot) => Vec4::new(0.9, 0.9, 0.9, 1.0),
        }
    }

    /// Get the plot type
    pub fn plot_type(&self) -> PlotType {
        match self {
            PlotElement::Line(_) => PlotType::Line,
            PlotElement::Scatter(_) => PlotType::Scatter,
            PlotElement::Bar(_) => PlotType::Bar,
            PlotElement::ErrorBar(_) => PlotType::ErrorBar,
            PlotElement::Stairs(_) => PlotType::Stairs,
            PlotElement::Stem(_) => PlotType::Stem,
            PlotElement::Area(_) => PlotType::Area,
            PlotElement::Quiver(_) => PlotType::Quiver,
            PlotElement::Pie(_) => PlotType::Pie,
            PlotElement::Image(_) => PlotType::Image,
            PlotElement::Surface(_) => PlotType::Surface,
            PlotElement::Scatter3(_) => PlotType::Scatter3,
            PlotElement::Contour(_) => PlotType::Contour,
            PlotElement::ContourFill(_) => PlotType::ContourFill,
        }
    }

    /// Get the plot's bounds
    pub fn bounds(&mut self) -> BoundingBox {
        match self {
            PlotElement::Line(plot) => plot.bounds(),
            PlotElement::Scatter(plot) => plot.bounds(),
            PlotElement::Bar(plot) => plot.bounds(),
            PlotElement::ErrorBar(plot) => plot.bounds(),
            PlotElement::Stairs(plot) => plot.bounds(),
            PlotElement::Stem(plot) => plot.bounds(),
            PlotElement::Area(plot) => plot.bounds(),
            PlotElement::Quiver(plot) => plot.bounds(),
            PlotElement::Pie(plot) => plot.bounds(),
            PlotElement::Image(plot) => plot.bounds(),
            PlotElement::Surface(_plot) => BoundingBox::default(),
            PlotElement::Scatter3(plot) => plot.bounds(),
            PlotElement::Contour(plot) => plot.bounds(),
            PlotElement::ContourFill(plot) => plot.bounds(),
        }
    }

    /// Generate render data for this plot
    pub fn render_data(&mut self) -> RenderData {
        match self {
            PlotElement::Line(plot) => plot.render_data(),
            PlotElement::Scatter(plot) => plot.render_data(),
            PlotElement::Bar(plot) => plot.render_data(),
            PlotElement::ErrorBar(plot) => plot.render_data(),
            PlotElement::Stairs(plot) => plot.render_data(),
            PlotElement::Stem(plot) => plot.render_data(),
            PlotElement::Area(plot) => plot.render_data(),
            PlotElement::Quiver(plot) => plot.render_data(),
            PlotElement::Pie(plot) => plot.render_data(),
            PlotElement::Image(plot) => plot.render_data(),
            PlotElement::Surface(plot) => plot.render_data(),
            PlotElement::Scatter3(plot) => plot.render_data(),
            PlotElement::Contour(plot) => plot.render_data(),
            PlotElement::ContourFill(plot) => plot.render_data(),
        }
    }

    /// Estimate memory usage
    pub fn estimated_memory_usage(&self) -> usize {
        match self {
            PlotElement::Line(plot) => plot.estimated_memory_usage(),
            PlotElement::Scatter(plot) => plot.estimated_memory_usage(),
            PlotElement::Bar(plot) => plot.estimated_memory_usage(),
            PlotElement::ErrorBar(plot) => plot.estimated_memory_usage(),
            PlotElement::Stairs(plot) => plot.estimated_memory_usage(),
            PlotElement::Stem(plot) => plot.estimated_memory_usage(),
            PlotElement::Area(plot) => plot.estimated_memory_usage(),
            PlotElement::Quiver(plot) => plot.estimated_memory_usage(),
            PlotElement::Pie(plot) => plot.estimated_memory_usage(),
            PlotElement::Image(plot) => plot.estimated_memory_usage(),
            PlotElement::Surface(_plot) => 0,
            PlotElement::Scatter3(plot) => plot.estimated_memory_usage(),
            PlotElement::Contour(plot) => plot.estimated_memory_usage(),
            PlotElement::ContourFill(plot) => plot.estimated_memory_usage(),
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
                Vec4::new(std::f64::consts::LOG10_2 as f32, 0.7450, 0.9330, 1.0), // Cyan
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
