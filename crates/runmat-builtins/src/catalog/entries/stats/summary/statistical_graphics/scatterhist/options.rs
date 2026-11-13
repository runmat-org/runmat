//! Public option spellings decode once into identities shared with runtime.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterhistOptionKind {
    NBins,
    Group,
    Kernel,
    Legend,
    PlotGroup,
    Location,
    Direction,
    Color,
    Marker,
    MarkerSize,
    LineWidth,
    LineStyle,
    Style,
    Bandwidth,
    Parent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterhistIntegerRole {
    Structural,
    RendererFloat,
    TextOnly,
    ParentHandle,
}

impl ScatterhistOptionKind {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "nbins" => Some(Self::NBins),
            "group" => Some(Self::Group),
            "kernel" => Some(Self::Kernel),
            "legend" => Some(Self::Legend),
            "plotgroup" => Some(Self::PlotGroup),
            "location" => Some(Self::Location),
            "direction" => Some(Self::Direction),
            "color" => Some(Self::Color),
            "marker" => Some(Self::Marker),
            "markersize" => Some(Self::MarkerSize),
            "linewidth" => Some(Self::LineWidth),
            "linestyle" => Some(Self::LineStyle),
            "style" => Some(Self::Style),
            "bandwidth" => Some(Self::Bandwidth),
            "parent" => Some(Self::Parent),
            _ => None,
        }
    }

    pub const fn integer_role(self) -> ScatterhistIntegerRole {
        match self {
            Self::NBins | Self::Group | Self::Kernel | Self::Legend | Self::PlotGroup => {
                ScatterhistIntegerRole::Structural
            }
            Self::Parent => ScatterhistIntegerRole::ParentHandle,
            Self::Location | Self::Direction | Self::Marker | Self::LineStyle | Self::Style => {
                ScatterhistIntegerRole::TextOnly
            }
            Self::Color | Self::MarkerSize | Self::LineWidth | Self::Bandwidth => {
                ScatterhistIntegerRole::RendererFloat
            }
        }
    }
}
