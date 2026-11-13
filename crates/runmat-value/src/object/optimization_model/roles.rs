//! Typed roles for optimization model ownership, not a second arithmetic or
//! numerical implementation. Operator identities remain OperatorKind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationConstraintRelation {
    LessEqual,
    GreaterEqual,
    Equal,
}
impl OptimizationConstraintRelation {
    pub const fn from_operator(operator: runmat_types::OperatorKind) -> Option<Self> {
        match operator {
            runmat_types::OperatorKind::LessEqual => Some(Self::LessEqual),
            runmat_types::OperatorKind::GreaterEqual => Some(Self::GreaterEqual),
            runmat_types::OperatorKind::Equal => Some(Self::Equal),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationObjectiveSense {
    Minimize,
    Maximize,
}
impl OptimizationObjectiveSense {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "min" | "minimize" => Some(Self::Minimize),
            "max" | "maximize" => Some(Self::Maximize),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Minimize => "minimize",
            Self::Maximize => "maximize",
        }
    }
}
