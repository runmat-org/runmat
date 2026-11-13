//! Public boxplot name-value identities. Text is decoded only at the caller boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxplotOptionKind {
    Whisker,
    Orientation,
    Labels,
    GroupOrder,
    Positions,
    Widths,
    OutlierSize,
    Symbol,
    Colors,
    PlotStyle,
    BoxStyle,
    MedianStyle,
    Notch,
    Jitter,
    DataLim,
    ExtremeMode,
    LabelOrientation,
    LabelVerbosity,
    ColorGroup,
    FactorDirection,
    FullFactors,
    FactorGap,
    FactorSeparator,
}

impl BoxplotOptionKind {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "whisker" => Self::Whisker,
            "orientation" => Self::Orientation,
            "labels" => Self::Labels,
            "grouporder" => Self::GroupOrder,
            "positions" => Self::Positions,
            "widths" => Self::Widths,
            "outliersize" => Self::OutlierSize,
            "symbol" => Self::Symbol,
            "colors" => Self::Colors,
            "plotstyle" => Self::PlotStyle,
            "boxstyle" => Self::BoxStyle,
            "medianstyle" => Self::MedianStyle,
            "notch" => Self::Notch,
            "jitter" => Self::Jitter,
            "datalim" => Self::DataLim,
            "extrememode" => Self::ExtremeMode,
            "labelorientation" => Self::LabelOrientation,
            "labelverbosity" => Self::LabelVerbosity,
            "colorgroup" => Self::ColorGroup,
            "factordirection" => Self::FactorDirection,
            "fullfactors" => Self::FullFactors,
            "factorgap" => Self::FactorGap,
            "factorseparator" => Self::FactorSeparator,
            _ => return None,
        })
    }
}
