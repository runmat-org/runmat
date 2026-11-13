/// Public property keys of the data-only IntegerConstraint value class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerConstraintField {
    Integer,
    SemiContinuous,
    SemiInteger,
}
impl IntegerConstraintField {
    pub const ALL: [Self; 3] = [Self::Integer, Self::SemiContinuous, Self::SemiInteger];
    pub const NAMES: [&'static str; Self::ALL.len()] = [
        Self::Integer.name(),
        Self::SemiContinuous.name(),
        Self::SemiInteger.name(),
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Integer => "Integer",
            Self::SemiContinuous => "SemiContinuous",
            Self::SemiInteger => "SemiInteger",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|key| key.name().eq_ignore_ascii_case(name))
    }
}
