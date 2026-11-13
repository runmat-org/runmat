//! Predefined MILP observers, distinct from continuous-solver plot contracts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MilpObserverKind {
    SaveSolutions,
    PlotBounds,
}
impl MilpObserverKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::SaveSolutions => "savemilpsolutions",
            Self::PlotBounds => "optimplotmilp",
        }
    }
}
