//! Typed registry identity matching; strings occur only at the public selector boundary.
use runmat_types::{standard as class, StaticClassIdentity};
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum AncestorKind {
    Root,
    Figure,
    Axes,
    Ruler,
    Legend,
    Text,
    Chart(StaticClassIdentity),
}
pub(super) enum TypeName {
    Kind(AncestorKind),
    FunctionSurface,
    Unknown,
}
impl TypeName {
    pub(super) fn parse(name: &str) -> Self {
        match name {
            "root" => Self::Kind(AncestorKind::Root),
            "figure" => Self::Kind(AncestorKind::Figure),
            "axes" => Self::Kind(AncestorKind::Axes),
            "ruler" | "numericruler" => Self::Kind(AncestorKind::Ruler),
            "legend" => Self::Kind(AncestorKind::Legend),
            "text" => Self::Kind(AncestorKind::Text),
            "functionsurface" => Self::FunctionSurface,
            "colorbar" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_COLOR_BAR)),
            "histogram" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_HISTOGRAM)),
            "histogram2" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_HISTOGRAM2)),
            "line" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_LINE)),
            "animatedline" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_ANIMATED_LINE)),
            "scatter" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_SCATTER)),
            "bar" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_BAR)),
            "stem" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_STEM)),
            "errorbar" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_ERROR_BAR)),
            "stairs" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_STAIR)),
            "quiver" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_QUIVER)),
            "image" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_IMAGE)),
            "heatmap" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_HEATMAP)),
            "binscatter" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_BINSCATTER)),
            "functioncontour" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_FUNCTION_CONTOUR)),
            "area" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_AREA)),
            "surface" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_SURFACE)),
            "patch" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_PATCH)),
            "contour" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_CONTOUR)),
            "constantline" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_CONSTANT_LINE)),
            "pie" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_PIE)),
            "textscatter" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_TEXT_SCATTER)),
            "wordcloud" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_WORD_CLOUD)),
            "stackedplot" => Self::Kind(AncestorKind::Chart(class::GRAPHICS_STACKED_LINE)),
            _ => Self::Unknown,
        }
    }
    pub(super) fn matches(&self, kind: AncestorKind) -> bool {
        match self {
            Self::Kind(expected) => {
                *expected == kind
                    || (*expected == AncestorKind::Text
                        && kind == AncestorKind::Chart(class::GRAPHICS_TEXT))
            }
            Self::FunctionSurface => {
                matches!(kind, AncestorKind::Chart(id) if id == class::GRAPHICS_FUNCTION_SURFACE || id == class::GRAPHICS_PARAMETERIZED_FUNCTION_SURFACE)
            }
            Self::Unknown => false,
        }
    }
}
