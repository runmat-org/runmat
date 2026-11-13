//! Public optimization-model property identities and boundary parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationProblemProperty {
    Description,
    ObjectiveSense,
    Variables,
    Objective,
    Constraints,
}
impl OptimizationProblemProperty {
    pub const ALL: [Self; 5] = [
        Self::Description,
        Self::ObjectiveSense,
        Self::Variables,
        Self::Objective,
        Self::Constraints,
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Description => "Description",
            Self::ObjectiveSense => "ObjectiveSense",
            Self::Variables => "Variables",
            Self::Objective => "Objective",
            Self::Constraints => "Constraints",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|property| property.name() == value)
    }
    pub fn parse_name_value(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|property| property.name().eq_ignore_ascii_case(value))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationExpressionProperty {
    IndexNames,
    Variables,
}
impl OptimizationExpressionProperty {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "IndexNames" => Some(Self::IndexNames),
            "Variables" => Some(Self::Variables),
            _ => None,
        }
    }
}
