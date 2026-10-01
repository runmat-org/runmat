/// Public mnrfit name-value identities; spellings are decoded only at the call boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MnrfitOption {
    Model,
    Link,
    Constant,
    Interactions,
    Weights,
    IterationLimit,
    MaxIter,
    Tolerance,
    TolX,
    Options,
    Display,
    EstDisp,
}

impl MnrfitOption {
    pub fn from_public(text: &str) -> Option<Self> {
        Some(match text.to_ascii_lowercase().as_str() {
            "model" => Self::Model,
            "link" => Self::Link,
            "constant" => Self::Constant,
            "interactions" => Self::Interactions,
            "weights" => Self::Weights,
            "iterationlimit" => Self::IterationLimit,
            "maxiter" => Self::MaxIter,
            "tolerance" => Self::Tolerance,
            "tolx" => Self::TolX,
            "options" => Self::Options,
            "display" => Self::Display,
            "estdisp" => Self::EstDisp,
            _ => return None,
        })
    }
}

/// Recognized fields of a public `statset`-style Options struct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MnrfitStatsetField {
    MaxIter,
    IterationLimit,
    TolX,
    Tolerance,
}

impl MnrfitStatsetField {
    pub const ALL: [Self; 4] = [
        Self::MaxIter,
        Self::IterationLimit,
        Self::TolX,
        Self::Tolerance,
    ];

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::MaxIter => "MaxIter",
            Self::IterationLimit => "IterationLimit",
            Self::TolX => "TolX",
            Self::Tolerance => "Tolerance",
        }
    }

    pub const fn is_iteration_limit(self) -> bool {
        match self {
            Self::MaxIter | Self::IterationLimit => true,
            Self::TolX | Self::Tolerance => false,
        }
    }
}
