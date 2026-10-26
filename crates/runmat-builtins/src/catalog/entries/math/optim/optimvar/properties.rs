//! Public optimization-variable property identities and boundary parsing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizationVariableProperty {
    Name,
    Type,
    IndexNames,
    LowerBound,
    UpperBound,
}
impl OptimizationVariableProperty {
    pub const ALL: [Self; 5] = [
        Self::Name,
        Self::Type,
        Self::IndexNames,
        Self::LowerBound,
        Self::UpperBound,
    ];
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "Name" => Some(Self::Name),
            "Type" => Some(Self::Type),
            "IndexNames" => Some(Self::IndexNames),
            "LowerBound" => Some(Self::LowerBound),
            "UpperBound" => Some(Self::UpperBound),
            _ => None,
        }
    }
    /// Full property spelling at the public constructor name/value boundary.
    pub fn parse_name_value(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|property| property.name().eq_ignore_ascii_case(value))
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Name => "Name",
            Self::Type => "Type",
            Self::IndexNames => "IndexNames",
            Self::LowerBound => "LowerBound",
            Self::UpperBound => "UpperBound",
        }
    }
}
