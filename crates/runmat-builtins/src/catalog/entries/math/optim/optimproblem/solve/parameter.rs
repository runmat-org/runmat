//! Public problem-solve name/value boundary. Solver identities and solver
//! options retain their existing canonical owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationSolveParameter {
    Solver,
    Options,
}
impl OptimizationSolveParameter {
    pub const ALL: [Self; 2] = [Self::Solver, Self::Options];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Solver => "Solver",
            Self::Options => "Options",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|field| field.name().eq_ignore_ascii_case(name))
    }
}
