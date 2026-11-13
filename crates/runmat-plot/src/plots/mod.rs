//! High-level plot type implementations
//!
//! This module contains implementations of specific plot types like
//! line plots, scatter plots, surfaces, etc.

pub mod area;
pub mod bar;
mod colormap_tables;
mod clone_budget;
mod color_data;
pub mod contour;
pub mod contour_fill;
pub mod errorbar;
pub mod legend_swatch;
mod export_error;
pub mod figure;
pub mod line;
pub mod line3;
pub mod mesh;
mod numeric_data;
pub mod patch;
pub mod pie;
pub mod quiver;
pub mod reference_line;
pub mod scatter;
pub mod scatter3;
pub mod stairs;
pub mod stem;
mod stroke;
pub mod surface;
pub mod volume;

pub use area::{AreaGpuSource, AreaPlot};
pub use bar::{BarChart, PolarHistogramDisplayStyle};
pub use contour::ContourPlot;
pub use contour_fill::ContourFillPlot;
pub use errorbar::ErrorBar;
pub use export_error::PlotDataExportError;
pub use figure::{
    project_figure_for_persistence, AxesGridCell, AxesKind, AxesMetadata, Figure,
    FigureAxesContainerSelection, FigureAxesPersistenceSelection, FigureAxesSelectionKind,
    FigureDecoration, FigurePersistenceProjection, FigurePersistenceProjectionError,
    FigurePersistenceSelection, FigureProjectionRemap, LegendEntry, LegendStyle, PlotElement,
    PlotType, TextStyle,
};
pub use line::{LineGpuStyle, LineMarkerAppearance, LinePlot, LineStyle};
pub use line3::Line3Plot;
pub use mesh::{
    MeshDeformation, MeshEdgeMode, MeshFieldLocation, MeshPlot, MeshRegion, MeshScalarField,
    MeshTriangleRange, MeshVectorField,
};
pub use numeric_data::NumericPlotData;
pub use patch::{
    PatchColorSourceKind, PatchEdgeColorMode, PatchFaceColorMode, PatchPlot, PatchSourceDataOwned,
};
pub use pie::PieChart;
pub use quiver::{QuiverGpuSource, QuiverPlot};
pub use reference_line::{ReferenceLine, ReferenceLineOrientation};
pub use scatter::{MarkerStyle, ScatterPlot};
pub use scatter3::Scatter3Plot;
pub use stairs::StairsPlot;
pub use stem::StemPlot;
pub use surface::{
    ColorMap, ShadingMode, SurfaceGpuColorGridSource, SurfaceGpuSource, SurfacePlot,
    SurfaceStatistics,
};
pub use volume::*;
