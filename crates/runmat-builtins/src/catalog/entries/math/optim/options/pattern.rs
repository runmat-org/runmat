//! Public pattern-search selector spellings. Recognizing a selector does not
//! certify an implementation: the selected runtime profile must consume it.
mod spec;
pub use spec::{OptimPatternDefault, OptimPatternScalarRule};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimPatternAlgorithm {
    Classic,
    Nups,
    NupsGps,
    NupsMads,
}
impl OptimPatternAlgorithm {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "classic" => Some(Self::Classic),
            "nups" => Some(Self::Nups),
            "nups-gps" => Some(Self::NupsGps),
            "nups-mads" => Some(Self::NupsMads),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Nups => "nups",
            Self::NupsGps => "nups-gps",
            Self::NupsMads => "nups-mads",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimPatternPoll {
    GpsMaximal,
    GpsMinimal,
    GssMaximal,
    GssMinimal,
    MadsMaximal,
    MadsMinimal,
    OrthoMadsMaximal,
    OrthoMadsMinimal,
}
impl OptimPatternPoll {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "gpspositivebasis2n" => Some(Self::GpsMaximal),
            "gpspositivebasisnp1" => Some(Self::GpsMinimal),
            "gsspositivebasis2n" => Some(Self::GssMaximal),
            "gsspositivebasisnp1" => Some(Self::GssMinimal),
            "madspositivebasis2n" => Some(Self::MadsMaximal),
            "madspositivebasisnp1" => Some(Self::MadsMinimal),
            "orthomadspositivebasis2n" => Some(Self::OrthoMadsMaximal),
            "orthomadspositivebasisnp1" => Some(Self::OrthoMadsMinimal),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::GpsMaximal => "GPSPositiveBasis2N",
            Self::GpsMinimal => "GPSPositiveBasisNp1",
            Self::GssMaximal => "GSSPositiveBasis2N",
            Self::GssMinimal => "GSSPositiveBasisNp1",
            Self::MadsMaximal => "MADSPositiveBasis2N",
            Self::MadsMinimal => "MADSPositiveBasisNp1",
            Self::OrthoMadsMaximal => "OrthoMADSPositiveBasis2N",
            Self::OrthoMadsMinimal => "OrthoMADSPositiveBasisNp1",
        }
    }
    pub const fn minimal(self) -> bool {
        matches!(
            self,
            Self::GpsMinimal | Self::GssMinimal | Self::MadsMinimal | Self::OrthoMadsMinimal
        )
    }
    pub const fn mesh_adaptive(self) -> bool {
        matches!(
            self,
            Self::MadsMaximal | Self::MadsMinimal | Self::OrthoMadsMaximal | Self::OrthoMadsMinimal
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimPatternOrder {
    Consecutive,
    Success,
    Random,
}
impl OptimPatternOrder {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "consecutive" => Some(Self::Consecutive),
            "success" => Some(Self::Success),
            "random" => Some(Self::Random),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Consecutive => "Consecutive",
            Self::Success => "Success",
            Self::Random => "Random",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimPatternSearch {
    Poll(OptimPatternPoll),
    Genetic,
    LatinHypercube,
    NelderMead,
    RbfSurrogate,
}
impl OptimPatternSearch {
    pub fn parse(value: &str) -> Option<Self> {
        if let Some(poll) = OptimPatternPoll::parse(value) {
            return Some(Self::Poll(poll));
        }
        match value.trim().to_ascii_lowercase().as_str() {
            "searchga" => Some(Self::Genetic),
            "searchlhs" => Some(Self::LatinHypercube),
            "searchneldermead" => Some(Self::NelderMead),
            "rbfsurrogate" => Some(Self::RbfSurrogate),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Poll(poll) => poll.name(),
            Self::Genetic => "searchga",
            Self::LatinHypercube => "searchlhs",
            Self::NelderMead => "searchneldermead",
            Self::RbfSurrogate => "rbfsurrogate",
        }
    }
}
