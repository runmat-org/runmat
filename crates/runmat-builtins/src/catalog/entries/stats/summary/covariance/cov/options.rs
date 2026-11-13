#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CovRowsOption {
    All,
    OmitRows,
    PartialRows,
}

impl CovRowsOption {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "all" => Some(Self::All),
            "omitrows" | "omit" => Some(Self::OmitRows),
            "partialrows" | "partial" | "pairwise" => Some(Self::PartialRows),
            _ => None,
        }
    }
}
