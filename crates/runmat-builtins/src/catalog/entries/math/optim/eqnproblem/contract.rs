use runmat_types::StaticClassIdentity;
/// Public native equation-problem class contract.
pub const EQUATION_PROBLEM_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("optim.problemdef.EquationProblem");
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EquationProblemProperty {
    Description,
    Equations,
    Variables,
}
impl EquationProblemProperty {
    pub const ALL: [Self; 3] = [Self::Description, Self::Equations, Self::Variables];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Description => "Description",
            Self::Equations => "Equations",
            Self::Variables => "Variables",
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
