//! Canonical public option identity for binned-scatter charts.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinscatterOptionKind {
    NumBins,
    XLimits,
    YLimits,
    ShowEmptyBins,
    FaceAlpha,
    DisplayName,
    DisplayStyle,
    UnsupportedMatlabOption,
}

pub const BINSCATTER_MAX_NUM_BINS: usize = 250;

impl BinscatterOptionKind {
    /// Decode once at the public text boundary; all later planning uses this
    /// discriminant, not a second semantic string switch.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "numbins" => Self::NumBins,
            "xlimits" => Self::XLimits,
            "ylimits" => Self::YLimits,
            "showemptybins" => Self::ShowEmptyBins,
            "facealpha" => Self::FaceAlpha,
            "displayname" => Self::DisplayName,
            "displaystyle" => Self::DisplayStyle,
            "normalization" | "binwidth" | "xbinwidth" | "ybinwidth" | "binmethod"
            | "xbinmethod" | "ybinmethod" | "binlimits" | "xbinlimits" | "ybinlimits"
            | "xbinedges" | "ybinedges" => Self::UnsupportedMatlabOption,
            _ => return None,
        })
    }
}
