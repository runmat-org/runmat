//! Shared public spelling boundary, never a builtin-name semantic switch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TableConstructorOption {
    VariableNames,
    RowNames,
    Size,
    VariableTypes,
}
impl TableConstructorOption {
    pub fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("VariableNames") {
            Some(Self::VariableNames)
        } else if name.eq_ignore_ascii_case("RowNames") {
            Some(Self::RowNames)
        } else if name.eq_ignore_ascii_case("Size") {
            Some(Self::Size)
        } else if name.eq_ignore_ascii_case("VariableTypes") {
            Some(Self::VariableTypes)
        } else {
            None
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CategoricalOption {
    Ordinal,
    Protected,
}
impl CategoricalOption {
    pub fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Ordinal") {
            Some(Self::Ordinal)
        } else if name.eq_ignore_ascii_case("Protected") {
            Some(Self::Protected)
        } else {
            None
        }
    }
}
