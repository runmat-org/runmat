//! Shared, runtime-independent identity and public option policy for the four
//! statistical means. Static inference and execution both use this enum.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatisticalMeanKind {
    Geometric,
    Harmonic,
    RootMeanSquare,
    RootMeanSquaredError,
}

impl StatisticalMeanKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Geometric => "geomean",
            Self::Harmonic => "harmmean",
            Self::RootMeanSquare => "rms",
            Self::RootMeanSquaredError => "rmse",
        }
    }

    pub const fn data_arity(self) -> usize {
        match self {
            Self::RootMeanSquaredError => 2,
            Self::Geometric | Self::Harmonic | Self::RootMeanSquare => 1,
        }
    }

    pub const fn allows_missing_synonyms(self) -> bool {
        matches!(self, Self::RootMeanSquare | Self::RootMeanSquaredError)
    }

    pub const fn allows_weights(self) -> bool {
        matches!(self, Self::RootMeanSquaredError)
    }
}
