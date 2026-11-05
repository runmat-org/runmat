//! Scene event tests.

mod storage_export;
mod roundtrip;
mod roundtrip_styles;
mod axes_metadata;
use super::*;
use crate::gpu::PlotReadbackError;
use crate::plots::{
    AreaPlot, BarChart, ContourFillPlot, ContourPlot, ErrorBar, Figure, Line3Plot, LinePlot,
    MeshPlot, NumericPlotData, PatchPlot, PieChart, PolarHistogramDisplayStyle, QuiverPlot,
    ReferenceLine, ReferenceLineOrientation, Scatter3Plot, ScatterPlot, StairsPlot, StemPlot,
    SurfacePlot,
};
use glam::{Vec3, Vec4};
