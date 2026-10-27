//! Compilation of the original typed model graph, not numerical probing or a
//! solver callback wrapper. The original layout is independent of cancellation
//! in derived coefficient tables.
use super::*;
mod array;
mod assembly;
mod constraint_geometry;
mod engine;
pub(super) mod equation;
mod exact;
mod factors;
mod formulation;
mod functions;
mod layout;
mod linear;
mod nonlinear;
mod norm_square;
mod numerical;
mod objective;
mod operations;
mod powers;
mod proof;
mod publication;
mod quadratic;
mod report;
mod residual;
mod run;
mod selection;
mod sparse_rows;
mod terms;
#[cfg(test)]
mod tests;
mod validation;
pub(super) use publication::publish;
pub(super) use report::Products;
pub(super) use run::{solve, AlgebraicSolved, NonlinearSolved, Solved};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stop {
    Unassigned,
    NonAffine,
    UnsupportedFunction,
    NonFinite,
}
impl Stop {
    fn error(self, policy: Policy) -> crate::RuntimeError {
        policy.invalid(match self {
            Self::Unassigned => "solve requires every selected expression slot to be assigned",
            Self::NonAffine => {
                "the selected model solver does not support this nonlinear objective or constraint"
            }
            Self::UnsupportedFunction => {
                "the model contains an expression function outside the supported compiler forms"
            }
            Self::NonFinite => "the model cannot be assembled in finite solver working precision",
        })
    }
}
pub(super) type Lowered<T> = BuiltinResult<Result<T, Stop>>;

#[derive(Clone)]
struct NodeFacts {
    degree: array::Degree,
    complete: bool,
    power: Option<std::rc::Rc<powers::Exponents>>,
}
pub(super) enum Prepared {
    Algebraic(Program),
    Nonlinear(nonlinear::Model),
}
pub(super) struct Program {
    source: evaluation::Plan,
    selected: Option<selection::Selection>,
    objective: objective::ModelObjective,
    layout: layout::Layout,
    arrays: std::collections::HashMap<runmat_gc::GcHandle, array::Array>,
    facts: std::collections::HashMap<runmat_gc::GcHandle, NodeFacts>,
}

impl Program {
    fn array(&self, root: runmat_gc::GcHandle, policy: Policy) -> BuiltinResult<&array::Array> {
        self.arrays
            .get(&root)
            .ok_or_else(|| policy.invalid("original compiled model root is unavailable"))
    }
}
