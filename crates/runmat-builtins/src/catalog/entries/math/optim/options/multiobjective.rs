//! Goal-attainment globalization, decoded only at the public option boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimMeritFunction {
    Multiobjective,
    SingleObjective,
}
impl OptimMeritFunction {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "multiobj" => Some(Self::Multiobjective),
            "singleobj" => Some(Self::SingleObjective),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Multiobjective => "multiobj",
            Self::SingleObjective => "singleobj",
        }
    }
}
