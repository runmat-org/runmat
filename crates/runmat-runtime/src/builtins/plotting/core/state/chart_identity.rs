//! One typed chart-family identity projection; no property dispatcher lives here.

use super::{FunctionSurfaceFunctionState, PlotChildHandleState};
use runmat_types::{standard as class, StaticClassIdentity};

impl PlotChildHandleState {
    pub fn class_identity(&self) -> StaticClassIdentity {
        self.chart_identity().0
    }

    pub(super) fn chart_identity(&self) -> (StaticClassIdentity, &'static str) {
        match self {
            Self::ColorBar(_) => (class::GRAPHICS_COLOR_BAR, "colorbar"),
            Self::Histogram(_) => (class::GRAPHICS_HISTOGRAM, "histogram"),
            Self::Histogram2(_) => (class::GRAPHICS_HISTOGRAM2, "histogram2"),
            Self::Line(_) | Self::Line3(_) => (class::GRAPHICS_LINE, "line"),
            Self::AnimatedLine(_) => (class::GRAPHICS_ANIMATED_LINE, "animatedline"),
            Self::Scatter(_) | Self::Scatter3(_) => (class::GRAPHICS_SCATTER, "scatter"),
            Self::Bar(_) => (class::GRAPHICS_BAR, "bar"),
            Self::Stem(_) => (class::GRAPHICS_STEM, "stem"),
            Self::ErrorBar(_) => (class::GRAPHICS_ERROR_BAR, "errorbar"),
            Self::Stairs(_) => (class::GRAPHICS_STAIR, "stairs"),
            Self::Quiver(_) => (class::GRAPHICS_QUIVER, "quiver"),
            Self::Image(_) => (class::GRAPHICS_IMAGE, "image"),
            Self::Heatmap(_) => (class::GRAPHICS_HEATMAP, "heatmap"),
            Self::Binscatter(_) => (class::GRAPHICS_BINSCATTER, "binscatter"),
            Self::FunctionSurface(state) => match &state.function {
                FunctionSurfaceFunctionState::Explicit(_) => {
                    (class::GRAPHICS_FUNCTION_SURFACE, "functionsurface")
                }
                FunctionSurfaceFunctionState::Parametric { .. } => (
                    class::GRAPHICS_PARAMETERIZED_FUNCTION_SURFACE,
                    "functionsurface",
                ),
            },
            Self::FunctionContour(_) => (class::GRAPHICS_FUNCTION_CONTOUR, "functioncontour"),
            Self::Area(_) => (class::GRAPHICS_AREA, "area"),
            Self::Surface(_) => (class::GRAPHICS_SURFACE, "surface"),
            Self::Patch(_) => (class::GRAPHICS_PATCH, "patch"),
            Self::Contour(_) | Self::ContourFill(_) => (class::GRAPHICS_CONTOUR, "contour"),
            Self::ReferenceLine(_) => (class::GRAPHICS_CONSTANT_LINE, "constantline"),
            Self::Pie(_) => (class::GRAPHICS_PIE, "pie"),
            Self::Text(_) => (class::GRAPHICS_TEXT, "text"),
            Self::TextScatter(_) => (class::GRAPHICS_TEXT_SCATTER, "textscatter"),
            Self::WordCloud(_) => (class::GRAPHICS_WORD_CLOUD, "wordcloud"),
            Self::StackedPlot(_) => (class::GRAPHICS_STACKED_LINE, "stackedplot"),
        }
    }
}
