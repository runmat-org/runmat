/// Normalized user-facing order-statistics option. Text belongs only at this
/// boundary; inference and execution switch on the same typed interpretation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrderStatisticOption {
    All,
    IncludeNan,
    OmitNan,
    Midpoint,
    Inclusive,
    Exclusive,
    Approximate,
    Method,
}

impl OrderStatisticOption {
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "all" => Some(Self::All),
            "includenan" => Some(Self::IncludeNan),
            "omitnan" => Some(Self::OmitNan),
            "midpoint" | "exact" => Some(Self::Midpoint),
            "inclusive" | "linear" => Some(Self::Inclusive),
            "exclusive" => Some(Self::Exclusive),
            "approximate" => Some(Self::Approximate),
            "method" => Some(Self::Method),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantileRequestUnit {
    Probability,
    Percentage,
}

impl QuantileRequestUnit {
    pub const fn upper_bound(self) -> i128 {
        match self {
            Self::Probability => 1,
            Self::Percentage => 100,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Probability => "probabilities",
            Self::Percentage => "percentages",
        }
    }
}

/// The public tiedrank flag forms. Keep interpretation shared with catalog
/// inference; execution decodes numeric scalars into this type at admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TiedRankMode {
    Ordinary,
    Kendall,
    Ansari,
}

impl TiedRankMode {
    pub fn from_flags(first: Option<bool>, second: Option<bool>) -> Option<Self> {
        match (first, second) {
            (None | Some(false), None | Some(false)) => Some(Self::Ordinary),
            (Some(true), None) => Some(Self::Kendall),
            (Some(false), Some(true)) => Some(Self::Ansari),
            _ => None,
        }
    }
}
