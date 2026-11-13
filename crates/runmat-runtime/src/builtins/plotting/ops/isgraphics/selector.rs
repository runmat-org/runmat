//! Text is interpreted only at the public selector boundary; matching uses typed identity.
use crate::builtins::plotting::{properties::PlotHandle, state::PlotObjectKind};
use runmat_types::{standard as class, StaticClassIdentity};
pub(crate) enum GraphicsType {
    Root,
    Figure,
    Axes,
    Ruler,
    XAxis,
    YAxis,
    Legend,
    Text,
    FunctionSurface,
    Patch,
    Chart(StaticClassIdentity),
    Unknown,
}
impl GraphicsType {
    pub(super) fn parse(name: &str) -> Self {
        match name {
            "root" => Self::Root,
            "figure" => Self::Figure,
            "axes" => Self::Axes,
            "ruler" => Self::Ruler,
            "xaxis" => Self::XAxis,
            "yaxis" => Self::YAxis,
            "legend" => Self::Legend,
            "text" => Self::Text,
            "functionsurface" => Self::FunctionSurface,
            "patch" => Self::Patch,
            "histogram" => Self::Chart(class::GRAPHICS_HISTOGRAM),
            "histogram2" => Self::Chart(class::GRAPHICS_HISTOGRAM2),
            "line" => Self::Chart(class::GRAPHICS_LINE),
            "animatedline" => Self::Chart(class::GRAPHICS_ANIMATED_LINE),
            "scatter" => Self::Chart(class::GRAPHICS_SCATTER),
            "bar" => Self::Chart(class::GRAPHICS_BAR),
            "stem" => Self::Chart(class::GRAPHICS_STEM),
            "errorbar" => Self::Chart(class::GRAPHICS_ERROR_BAR),
            "stairs" => Self::Chart(class::GRAPHICS_STAIR),
            "quiver" => Self::Chart(class::GRAPHICS_QUIVER),
            "image" => Self::Chart(class::GRAPHICS_IMAGE),
            "heatmap" => Self::Chart(class::GRAPHICS_HEATMAP),
            "binscatter" => Self::Chart(class::GRAPHICS_BINSCATTER),
            "functioncontour" => Self::Chart(class::GRAPHICS_FUNCTION_CONTOUR),
            "area" => Self::Chart(class::GRAPHICS_AREA),
            "surface" => Self::Chart(class::GRAPHICS_SURFACE),
            "contour" => Self::Chart(class::GRAPHICS_CONTOUR),
            "constantline" => Self::Chart(class::GRAPHICS_CONSTANT_LINE),
            "textscatter" => Self::Chart(class::GRAPHICS_TEXT_SCATTER),
            "wordcloud" => Self::Chart(class::GRAPHICS_WORD_CLOUD),
            "colorbar" => Self::Chart(class::GRAPHICS_COLOR_BAR),
            "stackedplot" => Self::Chart(class::GRAPHICS_STACKED_LINE),
            _ => Self::Unknown,
        }
    }
    pub(super) fn matches_class(&self, identity: StaticClassIdentity) -> bool {
        match self {
            Self::Figure => identity == class::FIGURE,
            Self::Axes => identity == class::AXES,
            Self::Text => identity == class::GRAPHICS_TEXT,
            Self::FunctionSurface => {
                identity == class::GRAPHICS_FUNCTION_SURFACE
                    || identity == class::GRAPHICS_PARAMETERIZED_FUNCTION_SURFACE
            }
            // Preserve existing public pie-as-patch semantics without stringifying a class.
            Self::Patch => identity == class::GRAPHICS_PATCH || identity == class::GRAPHICS_PIE,
            Self::Chart(expected) => identity == *expected,
            _ => false,
        }
    }
    pub(super) fn matches_handle(&self, handle: &PlotHandle) -> bool {
        match handle {
            PlotHandle::Root => matches!(self, Self::Root),
            PlotHandle::Figure(_) => matches!(self, Self::Figure),
            PlotHandle::Axes(_, _) => matches!(self, Self::Axes),
            PlotHandle::Legend(_, _) => matches!(self, Self::Legend),
            PlotHandle::Ruler(_, _, axis) => {
                matches!(self, Self::Ruler)
                    || matches!(
                        (self, axis),
                        (Self::XAxis, PlotObjectKind::XAxis) | (Self::YAxis, PlotObjectKind::YAxis)
                    )
            }
            PlotHandle::Text(_, _, _) => matches!(self, Self::Text),
            PlotHandle::PlotChild(_, child) => self.matches_class(child.class_identity()),
        }
    }
}
