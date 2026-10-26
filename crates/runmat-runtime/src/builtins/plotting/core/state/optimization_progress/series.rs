use super::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in super::super) enum OptimizationSeries {
    Standard(OptimPlotKind),
    MilpPrimal,
    MilpDual,
    NonlinearFeasible,
    NonlinearInfeasible,
    NonlinearBest,
    Genetic(OptimGeneticPlotKind, OptimGeneticPlotSeries),
    Surrogate(super::surrogate::Series),
}
impl From<OptimPlotKind> for OptimizationSeries {
    fn from(kind: OptimPlotKind) -> Self {
        Self::Standard(kind)
    }
}
impl OptimizationSeries {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Standard(kind) => kind.name(),
            Self::MilpPrimal | Self::MilpDual => "optimplotmilp",
            Self::Genetic(kind, _) => kind.name(),
            Self::Surrogate(_) => "surrogateoptplot",
            _ => "optimplotfvalconstr",
        }
    }
    pub(super) fn title(self) -> &'static str {
        match self {
            Self::Standard(kind) => kind.title(),
            Self::MilpPrimal | Self::MilpDual => "MILP Objective Bounds",
            Self::Genetic(kind, _) => kind.title(),
            Self::Surrogate(_) => "Surrogate Optimization",
            _ => "Feasible Objective Value",
        }
    }
    pub(super) fn x_label(self) -> &'static str {
        match self {
            Self::Standard(kind) => kind.x_label(),
            Self::MilpPrimal | Self::MilpDual => "Time (seconds)",
            Self::Genetic(kind, _) => kind.x_label(),
            Self::Surrogate(_) => "Function evaluations",
            _ => "Iteration",
        }
    }
    pub(super) fn y_label(self) -> &'static str {
        match self {
            Self::Standard(kind) => kind.y_label(),
            Self::MilpPrimal | Self::MilpDual => "Objective bound",
            Self::Genetic(kind, _) => kind.y_label(),
            Self::Surrogate(_) => "Objective function",
            _ => "Function value",
        }
    }
    pub(super) fn is_bar(self) -> bool {
        match self {
            Self::Standard(kind) => kind.is_point(),
            Self::Genetic(kind, _) => kind.is_bar(),
            _ => false,
        }
    }
}
